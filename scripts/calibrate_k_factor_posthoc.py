from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (
    PHYSICS_TARGET_NAMES,
    PreprocessedCSIDataset,
    apply_semantic_key_mode,
    collate_fn,
)
from data.semantic_key import SemanticKey
from data.tokenizer import CaptionTokenizer
from models.encoder import CSIEncoder
from models.model import CSIClip, K_FACTOR_STRONG_BIN_LABELS
from models.text_encoder import PhysicsTextEncoder
from scripts.evaluate import (
    _decode_strong_k,
    _infer_attribute_fields,
    _infer_attribute_remap,
    _infer_semantic_key_mode,
    _infer_token_norm_mode,
    _infer_use_power_branch,
    _load_model_state_compatible,
    _physics_raw_predictions,
    _render_physical_description,
    _strong_k_targets,
    align_samples_to_checkpoint_prototypes,
    build_attribute_label_maps,
    build_prototype_bank,
    build_tokenizer,
    filter_samples_by_min_class_size,
    move_batch,
)
from scripts.pretrain import assert_checkpoint_prototype_compatibility


STRONG_K_LABELS = K_FACTOR_STRONG_BIN_LABELS


@dataclass(frozen=True)
class StrongKPredictions:
    regression: torch.Tensor
    decoded: torch.Tensor
    target: torch.Tensor
    target_label: torch.Tensor
    target_position: torch.Tensor
    probabilities: torch.Tensor
    predicted_label: torch.Tensor
    predicted_position: torch.Tensor


@dataclass(frozen=True)
class AllKPredictions:
    regression: torch.Tensor
    decoded: torch.Tensor
    raw_physics_predictions: torch.Tensor
    target: torch.Tensor
    strong_mask: torch.Tensor
    probabilities: torch.Tensor
    predicted_label: torch.Tensor
    predicted_position: torch.Tensor
    predicted_los_status: tuple[str, ...] = ()
    target_los_status: tuple[str, ...] = ()
    predicted_strong_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class LinearCalibrator:
    weights: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        z = (features - self.mean) / self.std.clamp(min=1e-6)
        design = torch.cat([torch.ones(z.shape[0], 1, dtype=z.dtype), z], dim=1)
        return design @ self.weights


@dataclass(frozen=True)
class CalibrationCandidate:
    name: str
    predictions: torch.Tensor
    gate: torch.Tensor | None = None


def _physics_target_index(name: str) -> int:
    return PHYSICS_TARGET_NAMES.index(name)


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return math.nan
    x = x.float()
    y = y.float()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt(x_centered.square().sum() * y_centered.square().sum())
    if float(denom) <= 0.0:
        return 0.0
    return float((x_centered * y_centered).sum() / denom)


def _prepare_samples(data_path: str, checkpoint: dict, min_class_size: int) -> list:
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    semantic_key_mode = _infer_semantic_key_mode(checkpoint, None)
    samples = apply_semantic_key_mode(dataset.samples, semantic_key_mode)
    samples = filter_samples_by_min_class_size(samples, min_class_size=min_class_size)
    samples, _ = align_samples_to_checkpoint_prototypes(samples, checkpoint)
    return samples


def _build_model(
    samples: list,
    checkpoint: dict,
    device: torch.device,
) -> tuple[CSIClip, CaptionTokenizer, bool, dict[str, dict[str, int]], list[SemanticKey]]:
    token_norm_mode = _infer_token_norm_mode(checkpoint, None)
    use_power_branch = _infer_use_power_branch(checkpoint, None)
    csi_delay_input_weight = checkpoint.get("model_state", {}).get(
        "csi_delay_spread_head.1.weight"
    )
    use_delay_spread_head = (
        float(checkpoint.get("args", {}).get("delay_spread_weight", 0.0)) > 0.0
        and isinstance(csi_delay_input_weight, torch.Tensor)
        and csi_delay_input_weight.ndim == 2
        and csi_delay_input_weight.shape[1] == 256
    )
    attribute_fields = _infer_attribute_fields(checkpoint, None)
    attribute_remap = _infer_attribute_remap(checkpoint)
    tokenizer = build_tokenizer(samples, checkpoint)
    prototype_keys, _, _, _ = build_prototype_bank(
        samples,
        tokenizer,
        prototype_keys_override=None,
    )
    checkpoint_keys = checkpoint.get("prototype_keys")
    if checkpoint_keys is not None:
        _, _, _, _ = build_prototype_bank(samples, tokenizer)
        samples, checkpoint_prototype_keys = align_samples_to_checkpoint_prototypes(
            samples,
            checkpoint,
        )
        prototype_keys, _, _, _ = build_prototype_bank(
            samples,
            tokenizer,
            prototype_keys_override=checkpoint_prototype_keys,
        )
    attribute_label_maps = build_attribute_label_maps(
        samples,
        attribute_fields,
        attribute_remap=attribute_remap,
    )
    model = CSIClip(
        CSIEncoder(
            d_token=8,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
        ),
        PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300)),
        num_prototypes=len(prototype_keys),
        semantic_num_classes=len(prototype_keys),
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        use_power_branch=use_power_branch,
        use_delay_spread_head=use_delay_spread_head,
        attribute_num_classes={
            field: len(label_map)
            for field, label_map in attribute_label_maps.items()
        },
    ).to(device)
    assert_checkpoint_prototype_compatibility(
        checkpoint,
        prototype_keys,
        expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
        context="posthoc calibration checkpoint",
    )
    _load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()
    return model, tokenizer, use_power_branch, attribute_label_maps, prototype_keys


@torch.no_grad()
def collect_predictions(
    data_path: str,
    checkpoint: dict,
    checkpoint_path: str,
    batch_size: int,
    device: torch.device,
    min_class_size: int,
) -> tuple[StrongKPredictions, AllKPredictions]:
    samples = _prepare_samples(data_path, checkpoint, min_class_size=min_class_size)
    model, tokenizer, use_power_branch, attribute_label_maps, prototype_keys = _build_model(
        samples,
        checkpoint,
        device,
    )
    loader = DataLoader(
        PreprocessedCSIDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
    all_regression = []
    all_decoded = []
    all_raw_physics_predictions = []
    all_probabilities = []
    all_predicted_label = []
    all_predicted_position = []
    all_k_factor_bin_logits = []
    all_semantic_logits = []
    semantic_keys: list[SemanticKey] = []
    all_physics_raw_targets = []
    all_physics_masks = []

    for batch in loader:
        batch = move_batch(batch, device)
        csi_features = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
        )
        if "k_factor_bin" in attribute_label_maps:
            all_k_factor_bin_logits.append(
                model.predict_attributes(csi_features)["k_factor_bin"].cpu()
            )
        all_semantic_logits.append(model.predict_semantic(csi_features).cpu())
        power_context = None
        if use_power_branch:
            power_context = model.encode_power_context(
                batch["tokens"],
                batch["token_mask"],
                delay_power_map=batch.get("delay_power_map"),
                delay_power_profile=batch.get("delay_power_profile"),
            )
        physics_outputs = model.predict_physics_components(
            csi_features,
            power_context=power_context,
        )
        physics_predictions = physics_outputs["final"].cpu()
        strong_k_logits = physics_outputs["k_factor_strong_bin_logits"].cpu()
        strong_k_positions = physics_outputs["k_factor_strong_position"].cpu()
        physics_raw_targets = batch["physics_raw_targets"].cpu()
        physics_masks = batch["physics_target_mask"].cpu()
        all_physics_raw_targets.append(physics_raw_targets)
        all_physics_masks.append(physics_masks)
        semantic_keys.extend(batch["semantic_keys"])

        probabilities = strong_k_logits.softmax(dim=1)
        predicted_label = strong_k_logits.argmax(dim=1)
        decoded = _decode_strong_k(predicted_label, strong_k_positions)
        raw_physics_predictions = _physics_raw_predictions(physics_predictions)
        regression = raw_physics_predictions[:, _physics_target_index("k_factor_db")]
        all_regression.append(regression)
        all_decoded.append(decoded)
        all_raw_physics_predictions.append(raw_physics_predictions)
        all_probabilities.append(probabilities)
        all_predicted_label.append(predicted_label)
        all_predicted_position.append(strong_k_positions)

    physics_raw_targets = torch.cat(all_physics_raw_targets, dim=0)
    physics_masks = torch.cat(all_physics_masks, dim=0)
    target_label, target_position, mask = _strong_k_targets(
        physics_raw_targets,
        physics_masks,
        semantic_keys,
    )
    k_idx = _physics_target_index("k_factor_db")
    target = physics_raw_targets[:, k_idx]
    k_valid_mask = physics_masks[:, k_idx].bool()
    weak_mask = k_valid_mask & torch.tensor(
        [key.k_factor_bin == "weak" for key in semantic_keys],
        dtype=torch.bool,
        device=target.device,
    )
    all_valid_mask = weak_mask | mask
    full_regression = torch.cat(all_regression, dim=0)
    predicted_strong_mask = None
    if all_k_factor_bin_logits:
        label_map = attribute_label_maps["k_factor_bin"]
        strong_idx = label_map.get("strong")
        if strong_idx is not None:
            k_factor_bin_logits = torch.cat(all_k_factor_bin_logits, dim=0)[all_valid_mask]
            predicted_strong_mask = k_factor_bin_logits.argmax(dim=1) == strong_idx
    semantic_logits = torch.cat(all_semantic_logits, dim=0)[all_valid_mask]
    semantic_predictions = semantic_logits.argmax(dim=1).tolist()
    predicted_los_status = tuple(
        prototype_keys[int(label)].los_status
        for label in semantic_predictions
    )
    valid_flags = all_valid_mask.tolist()
    target_los_status = tuple(
        key.los_status
        for key, is_valid in zip(semantic_keys, valid_flags)
        if is_valid
    )

    del checkpoint_path
    strong_predictions = StrongKPredictions(
        regression=full_regression[mask],
        decoded=torch.cat(all_decoded, dim=0)[mask],
        target=target[mask],
        target_label=target_label[mask],
        target_position=target_position[mask],
        probabilities=torch.cat(all_probabilities, dim=0)[mask],
        predicted_label=torch.cat(all_predicted_label, dim=0)[mask],
        predicted_position=torch.cat(all_predicted_position, dim=0)[mask],
    )
    all_predictions = AllKPredictions(
        regression=full_regression[all_valid_mask],
        decoded=torch.cat(all_decoded, dim=0)[all_valid_mask],
        raw_physics_predictions=torch.cat(all_raw_physics_predictions, dim=0)[all_valid_mask],
        target=target[all_valid_mask],
        strong_mask=mask[all_valid_mask],
        probabilities=torch.cat(all_probabilities, dim=0)[all_valid_mask],
        predicted_label=torch.cat(all_predicted_label, dim=0)[all_valid_mask],
        predicted_position=torch.cat(all_predicted_position, dim=0)[all_valid_mask],
        predicted_los_status=predicted_los_status,
        target_los_status=target_los_status,
        predicted_strong_mask=predicted_strong_mask,
    )
    return strong_predictions, all_predictions


def _feature_matrix(preds, names: tuple[str, ...]) -> torch.Tensor:
    very_high_idx = STRONG_K_LABELS.index("very_high")
    columns = []
    for name in names:
        if name == "regression":
            columns.append(preds.regression)
        elif name == "decoded":
            columns.append(preds.decoded)
        elif name == "p_very_high":
            columns.append(preds.probabilities[:, very_high_idx])
        elif name == "p_high":
            columns.append(preds.probabilities[:, STRONG_K_LABELS.index("high")])
        elif name == "position":
            columns.append(preds.predicted_position)
        elif name == "max_probability":
            columns.append(preds.probabilities.max(dim=1).values)
        elif name == "predicted_label":
            columns.append(preds.predicted_label.float())
        else:
            raise ValueError(f"Unknown calibration feature: {name}")
    return torch.stack(columns, dim=1)


def fit_linear_calibrator(
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
) -> LinearCalibrator:
    features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    target = target.float()
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, correction=0).clamp(min=1e-6)
    z = (features - mean) / std
    design = torch.cat([torch.ones(z.shape[0], 1), z], dim=1)
    eye = torch.eye(design.shape[1])
    eye[0, 0] = 0.0
    lhs = design.T @ design + ridge * eye
    rhs = design.T @ target
    weights = torch.linalg.solve(lhs, rhs)
    return LinearCalibrator(weights=weights, mean=mean, std=std)


def _gate(preds: StrongKPredictions, name: str) -> torch.Tensor:
    very_high_idx = STRONG_K_LABELS.index("very_high")
    p_vhigh = preds.probabilities[:, very_high_idx]
    pred_vhigh = preds.predicted_label == very_high_idx
    if name == "argmax_vhigh":
        return pred_vhigh
    if name.startswith("p") and "_reg" not in name:
        threshold = float(name[1:]) / 100.0
        return pred_vhigh | (p_vhigh >= threshold)
    if name.startswith("p") and "_reg" in name:
        left, right = name.split("_reg", 1)
        probability_threshold = float(left[1:]) / 100.0
        regression_threshold = float(right)
        return (p_vhigh >= probability_threshold) & (preds.regression >= regression_threshold)
    if name.startswith("decoded"):
        threshold = float(name.replace("decoded", ""))
        return preds.decoded >= threshold
    if name.startswith("reg"):
        threshold = float(name.replace("reg", ""))
        return preds.regression >= threshold
    raise ValueError(f"Unknown gate: {name}")


@dataclass(frozen=True)
class QuantileCalibrator:
    source: torch.Tensor
    target: torch.Tensor

    def predict(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        source = self.source.to(dtype=values.dtype)
        target = self.target.to(dtype=values.dtype)
        idx = torch.searchsorted(source, values.clamp(min=float(source[0]), max=float(source[-1])))
        idx = idx.clamp(min=1, max=source.numel() - 1)
        left_idx = idx - 1
        right_idx = idx
        left_source = source[left_idx]
        right_source = source[right_idx]
        fraction = (values - left_source) / (right_source - left_source).clamp(min=1e-6)
        return target[left_idx] + fraction.clamp(0.0, 1.0) * (target[right_idx] - target[left_idx])


def fit_quantile_calibrator(
    source_values: torch.Tensor,
    target_values: torch.Tensor,
    *,
    num_quantiles: int = 101,
) -> QuantileCalibrator:
    source_values = source_values.float()
    target_values = target_values.float()
    quantiles = torch.linspace(0.0, 1.0, steps=num_quantiles)
    source = torch.quantile(source_values, quantiles)
    target = torch.quantile(target_values, quantiles)
    source = torch.maximum(source, torch.cat([source[:1], source[:-1] + 1e-5]))
    return QuantileCalibrator(source=source, target=target)


def build_candidates(
    train: StrongKPredictions,
    eval_preds: StrongKPredictions,
    *,
    ridge: float,
) -> list[CalibrationCandidate]:
    feature_names = (
        "regression",
        "decoded",
        "p_very_high",
        "p_high",
        "position",
        "max_probability",
        "predicted_label",
    )
    global_calibrator = fit_linear_calibrator(
        _feature_matrix(train, feature_names),
        train.target,
        ridge=ridge,
    )
    high_idx = STRONG_K_LABELS.index("high")
    very_high_idx = STRONG_K_LABELS.index("very_high")
    high_vhigh_train_mask = (train.target_label == high_idx) | (train.target_label == very_high_idx)
    high_vhigh_calibrator = fit_linear_calibrator(
        _feature_matrix(train, feature_names)[high_vhigh_train_mask],
        train.target[high_vhigh_train_mask],
        ridge=ridge,
    )
    very_high_train_mask = train.target_label == very_high_idx
    very_high_calibrator = fit_linear_calibrator(
        _feature_matrix(train, feature_names)[very_high_train_mask],
        train.target[very_high_train_mask],
        ridge=ridge,
    )
    high_vhigh_source = torch.maximum(train.regression, train.decoded)[high_vhigh_train_mask]
    high_vhigh_quantile = fit_quantile_calibrator(
        high_vhigh_source,
        train.target[high_vhigh_train_mask],
    )
    very_high_source = torch.maximum(train.regression, train.decoded)[very_high_train_mask]
    very_high_quantile = fit_quantile_calibrator(
        very_high_source,
        train.target[very_high_train_mask],
    )

    eval_features = _feature_matrix(eval_preds, feature_names)
    global_pred = global_calibrator.predict(eval_features)
    high_vhigh_pred = high_vhigh_calibrator.predict(eval_features)
    very_high_pred = very_high_calibrator.predict(eval_features)
    high_vhigh_quantile_pred = high_vhigh_quantile.predict(
        torch.maximum(eval_preds.regression, eval_preds.decoded)
    )
    very_high_quantile_pred = very_high_quantile.predict(
        torch.maximum(eval_preds.regression, eval_preds.decoded)
    )
    force_vhigh_pred = torch.maximum(
        eval_preds.regression,
        45.0 + eval_preds.predicted_position.clamp(0.0, 1.0) * 25.0,
    )
    candidates = [
        CalibrationCandidate("regression", eval_preds.regression),
        CalibrationCandidate("decoded", eval_preds.decoded),
        CalibrationCandidate("max_regression_decoded", torch.maximum(eval_preds.regression, eval_preds.decoded)),
        CalibrationCandidate("global_linear", global_pred),
    ]
    gates = [
        "argmax_vhigh",
        "p20",
        "p30",
        "p40",
        "p50",
        "p20_reg35",
        "p20_reg40",
        "p20_reg45",
        "p30_reg35",
        "p30_reg40",
        "p30_reg45",
        "decoded45",
        "reg45",
    ]
    for gate_name in gates:
        gate = _gate(eval_preds, gate_name)
        candidates.extend(
            [
                CalibrationCandidate(
                    f"gate_{gate_name}_max_reg_decoded",
                    torch.where(gate, torch.maximum(eval_preds.regression, eval_preds.decoded), eval_preds.regression),
                    gate,
                ),
                CalibrationCandidate(
                    f"gate_{gate_name}_high_vhigh_linear",
                    torch.where(gate, high_vhigh_pred, eval_preds.regression),
                    gate,
                ),
                CalibrationCandidate(
                    f"gate_{gate_name}_very_high_linear",
                    torch.where(gate, very_high_pred, eval_preds.regression),
                    gate,
                ),
                CalibrationCandidate(
                    f"gate_{gate_name}_force_vhigh_position",
                    torch.where(gate, force_vhigh_pred, eval_preds.regression),
                    gate,
                ),
                CalibrationCandidate(
                    f"gate_{gate_name}_high_vhigh_quantile",
                    torch.where(gate, high_vhigh_quantile_pred, eval_preds.regression),
                    gate,
                ),
                CalibrationCandidate(
                    f"gate_{gate_name}_very_high_quantile",
                    torch.where(gate, very_high_quantile_pred, eval_preds.regression),
                    gate,
                ),
            ]
        )
    return candidates


def build_deployable_all_k_candidates(
    train: StrongKPredictions,
    eval_all: AllKPredictions,
    *,
    ridge: float,
    weak_constant: float,
) -> list[CalibrationCandidate]:
    if eval_all.predicted_strong_mask is None:
        return []

    feature_names = (
        "regression",
        "decoded",
        "p_very_high",
        "p_high",
        "position",
        "max_probability",
        "predicted_label",
    )
    train_features = _feature_matrix(train, feature_names)
    eval_features = _feature_matrix(eval_all, feature_names)
    global_calibrator = fit_linear_calibrator(
        train_features,
        train.target,
        ridge=ridge,
    )
    high_idx = STRONG_K_LABELS.index("high")
    very_high_idx = STRONG_K_LABELS.index("very_high")
    high_vhigh_mask = (train.target_label == high_idx) | (train.target_label == very_high_idx)
    high_vhigh_calibrator = fit_linear_calibrator(
        train_features[high_vhigh_mask],
        train.target[high_vhigh_mask],
        ridge=ridge,
    )
    very_high_mask = train.target_label == very_high_idx
    very_high_calibrator = fit_linear_calibrator(
        train_features[very_high_mask],
        train.target[very_high_mask],
        ridge=ridge,
    )

    weak_values = torch.full_like(eval_all.target, fill_value=float(weak_constant))

    def apply_attribute_gate(strong_values: torch.Tensor) -> torch.Tensor:
        return torch.where(eval_all.predicted_strong_mask, strong_values, weak_values)

    return [
        CalibrationCandidate(
            "attribute_gate_raw_regression",
            apply_attribute_gate(eval_all.regression),
        ),
        CalibrationCandidate(
            "attribute_gate_decoded",
            apply_attribute_gate(eval_all.decoded),
        ),
        CalibrationCandidate(
            "attribute_gate_max_regression_decoded",
            apply_attribute_gate(torch.maximum(eval_all.regression, eval_all.decoded)),
        ),
        CalibrationCandidate(
            "attribute_gate_global_linear",
            apply_attribute_gate(global_calibrator.predict(eval_features)),
        ),
        CalibrationCandidate(
            "attribute_gate_high_vhigh_linear",
            apply_attribute_gate(high_vhigh_calibrator.predict(eval_features)),
        ),
        CalibrationCandidate(
            "attribute_gate_very_high_linear",
            apply_attribute_gate(very_high_calibrator.predict(eval_features)),
        ),
    ]


def _metrics(
    predictions: torch.Tensor,
    target: torch.Tensor,
    target_label: torch.Tensor,
    gate: torch.Tensor | None,
) -> dict[str, float]:
    errors = predictions - target
    metrics = {
        "mae": float(errors.abs().mean()),
        "accuracy3": float((errors.abs() <= 3.0).float().mean()),
        "signed_mean": float(errors.mean()),
        "pearson": _safe_pearson(predictions, target),
        "pred_min": float(predictions.min()),
        "pred_max": float(predictions.max()),
    }
    for idx, label in enumerate(STRONG_K_LABELS):
        mask = target_label == idx
        if bool(mask.any()):
            label_errors = errors[mask]
            metrics[f"{label}_mae"] = float(label_errors.abs().mean())
            metrics[f"{label}_signed_mean"] = float(label_errors.mean())
            metrics[f"{label}_accuracy3"] = float((label_errors.abs() <= 3.0).float().mean())
    very_high_mask = target_label == STRONG_K_LABELS.index("very_high")
    if gate is not None and bool(very_high_mask.any()):
        metrics["gate_fraction"] = float(gate.float().mean())
        metrics["very_high_recall"] = float(gate[very_high_mask].float().mean())
        metrics["very_high_precision"] = (
            float(very_high_mask[gate].float().mean()) if bool(gate.any()) else math.nan
        )
    return metrics


def print_candidate(prefix: str, name: str, metrics: dict[str, float]) -> None:
    fields = [
        f"{prefix}_name={name}",
        f"MAE={metrics['mae']:.4f}",
        f"accuracy@3={metrics['accuracy3']:.4f}",
        f"signed_mean={metrics['signed_mean']:.4f}",
        f"pearson={metrics['pearson']:.4f}",
        f"pred_range={metrics['pred_min']:.4f},{metrics['pred_max']:.4f}",
    ]
    for label in STRONG_K_LABELS:
        key = f"{label}_mae"
        if key in metrics:
            fields.append(f"{label}_MAE={metrics[key]:.4f}")
            fields.append(f"{label}_signed={metrics[f'{label}_signed_mean']:.4f}")
    if "gate_fraction" in metrics:
        fields.extend(
            [
                f"gate_fraction={metrics['gate_fraction']:.4f}",
                f"very_high_recall={metrics['very_high_recall']:.4f}",
                f"very_high_precision={metrics['very_high_precision']:.4f}",
            ]
        )
    print(" ".join(fields))


def print_recommendations(scored: list[tuple[str, dict[str, float]]]) -> None:
    if not scored:
        return
    best_overall = min(scored, key=lambda item: item[1]["mae"])
    best_very_high = min(scored, key=lambda item: item[1].get("very_high_mae", math.inf))
    tail_candidates = [
        item
        for item in scored
        if item[1]["pred_max"] >= 58.0
    ]
    best_tail = (
        min(tail_candidates, key=lambda item: item[1]["mae"])
        if tail_candidates
        else None
    )
    print_candidate("posthoc_recommend_overall", best_overall[0], best_overall[1])
    print_candidate("posthoc_recommend_very_high", best_very_high[0], best_very_high[1])
    if best_tail is not None:
        print_candidate("posthoc_recommend_tail_preserving", best_tail[0], best_tail[1])
    else:
        print("posthoc_recommend_tail_preserving_name=none reason=no_candidate_with_pred_max_ge_58")


def _merge_all_k_with_weak_constant(
    all_preds: AllKPredictions,
    strong_predictions: torch.Tensor,
    *,
    weak_constant: float,
) -> torch.Tensor:
    strong_count = int(all_preds.strong_mask.sum().item())
    if strong_predictions.numel() != strong_count:
        raise ValueError(
            "Strong candidate length does not match all-K strong mask: "
            f"{strong_predictions.numel()} != {strong_count}"
        )
    merged = torch.full_like(all_preds.target, fill_value=float(weak_constant))
    merged[all_preds.strong_mask] = strong_predictions
    return merged


def _all_k_metrics(
    predictions: torch.Tensor,
    target: torch.Tensor,
    strong_mask: torch.Tensor,
) -> dict[str, float]:
    errors = predictions - target
    weak_mask = ~strong_mask
    metrics = {
        "count": float(target.numel()),
        "weak_count": float(weak_mask.sum()),
        "strong_count": float(strong_mask.sum()),
        "mae": float(errors.abs().mean()),
        "accuracy3": float((errors.abs() <= 3.0).float().mean()),
        "signed_mean": float(errors.mean()),
        "pearson": _safe_pearson(predictions, target),
        "pred_min": float(predictions.min()),
        "pred_max": float(predictions.max()),
    }
    for label, mask in (("weak", weak_mask), ("strong", strong_mask)):
        if bool(mask.any()):
            label_errors = errors[mask]
            metrics[f"{label}_mae"] = float(label_errors.abs().mean())
            metrics[f"{label}_accuracy3"] = float((label_errors.abs() <= 3.0).float().mean())
            metrics[f"{label}_signed_mean"] = float(label_errors.mean())
    return metrics


def print_all_k_candidate(prefix: str, name: str, metrics: dict[str, float]) -> None:
    fields = [
        f"{prefix}_name={name}",
        f"count={int(metrics['count'])}",
        f"weak_count={int(metrics['weak_count'])}",
        f"strong_count={int(metrics['strong_count'])}",
        f"MAE={metrics['mae']:.4f}",
        f"accuracy@3={metrics['accuracy3']:.4f}",
        f"signed_mean={metrics['signed_mean']:.4f}",
        f"pearson={metrics['pearson']:.4f}",
        f"pred_range={metrics['pred_min']:.4f},{metrics['pred_max']:.4f}",
    ]
    for label in ("weak", "strong"):
        key = f"{label}_mae"
        if key in metrics:
            fields.append(f"{label}_MAE={metrics[key]:.4f}")
            fields.append(f"{label}_accuracy@3={metrics[f'{label}_accuracy3']:.4f}")
            fields.append(f"{label}_signed={metrics[f'{label}_signed_mean']:.4f}")
    if "predicted_strong_fraction" in metrics:
        fields.extend(
            [
                f"predicted_strong_fraction={metrics['predicted_strong_fraction']:.4f}",
                f"weak_as_weak={metrics['weak_as_weak']:.4f}",
                f"strong_as_strong={metrics['strong_as_strong']:.4f}",
            ]
        )
    print(" ".join(fields))


def print_all_k_results(
    all_preds: AllKPredictions,
    strong_candidates: list[CalibrationCandidate],
    deployable_candidates: list[CalibrationCandidate],
    *,
    top_k: int,
    weak_constant: float,
) -> list[tuple[str, dict[str, float]]]:
    baseline_metrics = _all_k_metrics(
        all_preds.regression,
        all_preds.target,
        all_preds.strong_mask,
    )
    print_all_k_candidate("all_k_baseline", "raw_regression", baseline_metrics)
    if all_preds.predicted_strong_mask is not None:
        gated_regression = torch.where(
            all_preds.predicted_strong_mask,
            all_preds.regression,
            torch.full_like(all_preds.regression, fill_value=float(weak_constant)),
        )
        attribute_gate_metrics = _all_k_metrics(
            gated_regression,
            all_preds.target,
            all_preds.strong_mask,
        )
        weak_mask = ~all_preds.strong_mask
        attribute_gate_metrics["predicted_strong_fraction"] = float(
            all_preds.predicted_strong_mask.float().mean()
        )
        attribute_gate_metrics["weak_as_weak"] = (
            float((~all_preds.predicted_strong_mask[weak_mask]).float().mean())
            if bool(weak_mask.any())
            else math.nan
        )
        attribute_gate_metrics["strong_as_strong"] = (
            float(all_preds.predicted_strong_mask[all_preds.strong_mask].float().mean())
            if bool(all_preds.strong_mask.any())
            else math.nan
        )
        print_all_k_candidate(
            "all_k_attribute_gate_baseline",
            "k_factor_bin_gate_raw_regression",
            attribute_gate_metrics,
        )

    deployable_scored = [
        (
            candidate.name,
            _all_k_metrics(candidate.predictions, all_preds.target, all_preds.strong_mask),
        )
        for candidate in deployable_candidates
    ]
    deployable_by_mae = sorted(deployable_scored, key=lambda item: item[1]["mae"])
    for rank, (name, metrics) in enumerate(deployable_by_mae[:top_k], start=1):
        print_all_k_candidate(f"all_k_deployable_rank{rank}", name, metrics)
    if deployable_by_mae:
        print_all_k_candidate(
            "all_k_recommend_deployable",
            deployable_by_mae[0][0],
            deployable_by_mae[0][1],
        )

    scored = []
    for candidate in strong_candidates:
        merged = _merge_all_k_with_weak_constant(
            all_preds,
            candidate.predictions,
            weak_constant=weak_constant,
        )
        name = f"weak_constant_plus_{candidate.name}"
        scored.append((name, _all_k_metrics(merged, all_preds.target, all_preds.strong_mask)))

    by_mae = sorted(scored, key=lambda item: item[1]["mae"])
    for rank, (name, metrics) in enumerate(by_mae[:top_k], start=1):
        print_all_k_candidate(f"all_k_best_rank{rank}", name, metrics)
    if by_mae:
        print_all_k_candidate("all_k_recommend_overall", by_mae[0][0], by_mae[0][1])
    return deployable_by_mae


def save_final_predictions(
    output_path: str,
    all_preds: AllKPredictions,
    candidate: CalibrationCandidate,
) -> None:
    delay_idx = _physics_target_index("delay_spread_ns")
    azimuth_idx = _physics_target_index("azimuth_spread_deg")
    records = []
    texts = []
    for idx in range(candidate.predictions.numel()):
        los_status = (
            all_preds.predicted_los_status[idx]
            if idx < len(all_preds.predicted_los_status)
            else "nlos"
        )
        record = {
            "los_status": str(los_status),
            "delay_spread_ns": float(all_preds.raw_physics_predictions[idx, delay_idx]),
            "k_factor_db": float(candidate.predictions[idx]),
            "azimuth_spread_deg": float(all_preds.raw_physics_predictions[idx, azimuth_idx]),
        }
        records.append(record)
        texts.append(_render_physical_description(record))

    payload = {
        "candidate_name": candidate.name,
        "prediction": candidate.predictions.cpu(),
        "target": all_preds.target.cpu(),
        "strong_mask": all_preds.strong_mask.cpu(),
        "raw_regression": all_preds.regression.cpu(),
        "decoded": all_preds.decoded.cpu(),
        "raw_physics_predictions": all_preds.raw_physics_predictions.cpu(),
        "predicted_label": all_preds.predicted_label.cpu(),
        "predicted_position": all_preds.predicted_position.cpu(),
        "predicted_los_status": list(all_preds.predicted_los_status),
        "target_los_status": list(all_preds.target_los_status),
        "final_physical_records": records,
        "final_physical_descriptions": texts,
    }
    if all_preds.predicted_strong_mask is not None:
        payload["predicted_strong_mask"] = all_preds.predicted_strong_mask.cpu()
    torch.save(payload, output_path)
    print(f"saved_final_k_predictions={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit post-hoc K-factor calibration/gating on train predictions and report eval metrics."
    )
    parser.add_argument(
        "--train-path",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_k_factor_5bin_balanced_2000_train.pt",
    )
    parser.add_argument(
        "--eval-path",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_k_factor_5bin_balanced_2000_eval.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default="/root/autodl-tmp/CSI_model/artifacts/pretrain_k_factor_5bin_balanced_2000_power_k_heads_logstats/checkpoint_last.pt",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-class-size", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--weak-constant",
        type=float,
        default=-20.0,
        help="K-factor value assigned to weak samples when merging weak and strong predictions.",
    )
    parser.add_argument(
        "--save-predictions",
        help=(
            "Optional .pt path for final deployable all-sample K predictions. "
            "Saves the best all_k_deployable candidate."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"checkpoint={args.checkpoint}")
    print(f"train_path={args.train_path}")
    print(f"eval_path={args.eval_path}")
    train_preds, train_all_preds = collect_predictions(
        args.train_path,
        checkpoint,
        args.checkpoint,
        args.batch_size,
        device,
        min_class_size=args.min_class_size,
    )
    eval_preds, eval_all_preds = collect_predictions(
        args.eval_path,
        checkpoint,
        args.checkpoint,
        args.batch_size,
        device,
        min_class_size=args.min_class_size,
    )
    print(f"train_strong_count={train_preds.target.numel()}")
    print(f"eval_strong_count={eval_preds.target.numel()}")
    print(f"train_all_k_count={train_all_preds.target.numel()}")
    print(f"eval_all_k_count={eval_all_preds.target.numel()}")
    print(f"all_k_weak_constant={args.weak_constant:.4f}")
    candidates = build_candidates(train_preds, eval_preds, ridge=args.ridge)
    deployable_candidates = build_deployable_all_k_candidates(
        train_preds,
        eval_all_preds,
        ridge=args.ridge,
        weak_constant=args.weak_constant,
    )
    scored = [
        (
            candidate.name,
            _metrics(candidate.predictions, eval_preds.target, eval_preds.target_label, candidate.gate),
        )
        for candidate in candidates
    ]
    by_mae = sorted(scored, key=lambda item: item[1]["mae"])
    by_very_high = sorted(scored, key=lambda item: item[1].get("very_high_mae", math.inf))
    for rank, (name, metrics) in enumerate(by_mae[: args.top_k], start=1):
        print_candidate(f"posthoc_best_overall_rank{rank}", name, metrics)
    for rank, (name, metrics) in enumerate(by_very_high[: args.top_k], start=1):
        print_candidate(f"posthoc_best_very_high_rank{rank}", name, metrics)
    print_recommendations(scored)
    deployable_scored = print_all_k_results(
        eval_all_preds,
        candidates,
        deployable_candidates,
        top_k=args.top_k,
        weak_constant=args.weak_constant,
    )
    if args.save_predictions:
        if not deployable_scored:
            raise RuntimeError(
                "No deployable candidate is available. "
                "Train/evaluate with a k_factor_bin attribute classifier first."
            )
        best_name = deployable_scored[0][0]
        best_candidate = next(candidate for candidate in deployable_candidates if candidate.name == best_name)
        save_final_predictions(args.save_predictions, eval_all_preds, best_candidate)


if __name__ == "__main__":
    main()
