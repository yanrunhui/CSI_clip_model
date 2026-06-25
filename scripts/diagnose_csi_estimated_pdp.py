from __future__ import annotations

import argparse
import math
import sys
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
from models.encoder import CSIEncoder
from models.model import (
    CSIClip,
    FIRST_PATH_DELAY_POSITION_BINS,
    fuse_first_path_delay_soft_from_bin_position,
)
from models.text_encoder import PhysicsTextEncoder
from scripts.evaluate import (
    _infer_max_delay_spread_ns,
    _infer_min_class_size,
    _infer_semantic_key_mode,
    _infer_token_norm_mode,
    _infer_use_delay_specific_encoder,
    _infer_use_delay_spread_head,
    _infer_use_power_branch,
    _load_model_state_compatible,
    align_samples_to_checkpoint_prototypes,
    build_attribute_label_maps,
    build_prototype_bank,
    build_tokenizer,
    filter_samples_by_max_delay_spread,
    filter_samples_by_min_class_size,
)
from scripts.pretrain import assert_checkpoint_prototype_compatibility


FIRST_DELAY_BINS_NS = tuple(
    (label, lower, float("inf") if idx == len(FIRST_PATH_DELAY_POSITION_BINS) - 1 else upper)
    for idx, (label, lower, upper) in enumerate(FIRST_PATH_DELAY_POSITION_BINS)
)
FIRST_DELAY_BIN_LABELS = tuple(label for label, _, _ in FIRST_DELAY_BINS_NS)


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return math.nan
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) <= 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def _safe_r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    total = (targets - targets.mean()).square().sum()
    if float(total) <= 0.0:
        return 0.0
    residual = (predictions - targets).square().sum()
    return float(1.0 - residual / total)


def _ridge_label(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def parse_ridges(value: str) -> tuple[float, ...]:
    ridges = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not ridges:
        raise ValueError("--ridges must contain at least one value.")
    if any(ridge < 0.0 for ridge in ridges):
        raise ValueError("--ridges must be non-negative.")
    return ridges


def parse_floats(value: str, *, argument_name: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError(f"{argument_name} must contain at least one value.")
    return values


def parse_labels(
    value: str,
    *,
    choices: tuple[str, ...],
    argument_name: str,
) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in value.split(",") if part.strip())
    if not labels:
        raise ValueError(f"{argument_name} must contain at least one label.")
    unknown = sorted(set(labels) - set(choices))
    if unknown:
        raise ValueError(
            f"{argument_name} contains unknown labels {unknown}; "
            f"valid labels are {','.join(choices)}."
        )
    return labels


def _complex_beam_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [K, D, F], got {tuple(tokens.shape)}")
    if tokens.shape[1] % 2 != 0:
        raise ValueError(
            "tokens channel dimension must contain real/imag halves, "
            f"got D={tokens.shape[1]}."
        )
    half = tokens.shape[1] // 2
    return torch.complex(tokens[:, :half].float(), tokens[:, half:].float())


def estimated_delay_profile_from_tokens(
    tokens: torch.Tensor,
    subcarrier_spacing_hz: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    complex_tokens = _complex_beam_tokens(tokens)
    delay_response = torch.fft.ifft(complex_tokens, dim=-1)
    profile = delay_response.abs().square().sum(dim=(0, 1)).float()
    total = profile.sum()
    if float(total) > 0.0:
        profile = profile / total
    n_freq = profile.numel()
    spacing = max(float(subcarrier_spacing_hz), 1e-6)
    delay_ns = torch.arange(n_freq, dtype=torch.float32) / (n_freq * spacing) * 1e9
    return profile, delay_ns


def _weighted_quantile(delay_ns: torch.Tensor, profile: torch.Tensor, quantile: float) -> torch.Tensor:
    cdf = torch.cumsum(profile, dim=0)
    if float(cdf[-1]) <= 0.0:
        return torch.tensor(0.0)
    idx = int(torch.searchsorted(cdf, torch.tensor(float(quantile))).clamp(max=profile.numel() - 1))
    return delay_ns[idx]


def summarize_profile(
    profile: torch.Tensor,
    delay_ns: torch.Tensor,
    max_summary_delay_ns: float,
) -> torch.Tensor:
    valid = torch.isfinite(delay_ns) & (delay_ns <= max_summary_delay_ns)
    if not bool(valid.any()):
        valid = torch.ones_like(delay_ns, dtype=torch.bool)
    profile = profile[valid].float()
    delay_ns = delay_ns[valid].float()
    total = profile.sum()
    if float(total) <= 0.0:
        return torch.zeros(18, dtype=torch.float32)
    profile = profile / total
    mean_delay = (profile * delay_ns).sum()
    rms_delay = torch.sqrt((profile * (delay_ns - mean_delay).square()).sum().clamp(min=0.0))
    peak_idx = int(profile.argmax().item())
    peak_delay = delay_ns[peak_idx]
    peak_power = profile[peak_idx]
    early_100 = profile[delay_ns <= 100.0].sum()
    early_200 = profile[delay_ns <= 200.0].sum()
    early_400 = profile[delay_ns <= 400.0].sum()
    mid_400_800 = profile[(delay_ns >= 400.0) & (delay_ns < 800.0)].sum()
    tail_800 = profile[delay_ns >= 800.0].sum()
    cdf10 = _weighted_quantile(delay_ns, profile, 0.10)
    cdf25 = _weighted_quantile(delay_ns, profile, 0.25)
    cdf50 = _weighted_quantile(delay_ns, profile, 0.50)
    cdf75 = _weighted_quantile(delay_ns, profile, 0.75)
    cdf90 = _weighted_quantile(delay_ns, profile, 0.90)
    threshold = profile.max() * 0.10
    above = torch.nonzero(profile >= threshold, as_tuple=False).flatten()
    first_threshold = delay_ns[int(above[0].item())] if above.numel() else torch.tensor(0.0)
    entropy = -(profile.clamp(min=1e-12) * profile.clamp(min=1e-12).log()).sum()
    active_bins = (profile >= profile.max() * 0.05).float().sum()
    profile_std = profile.std(correction=0)
    return torch.stack(
        [
            peak_delay,
            mean_delay,
            rms_delay,
            cdf10,
            cdf25,
            cdf50,
            cdf75,
            cdf90,
            first_threshold,
            early_100,
            early_200,
            early_400,
            mid_400_800,
            tail_800,
            peak_power,
            entropy,
            active_bins,
            profile_std,
        ]
    ).float()


def downsample_profile(profile: torch.Tensor, delay_ns: torch.Tensor, bins: int, max_delay_ns: float) -> torch.Tensor:
    valid = torch.isfinite(delay_ns) & (delay_ns <= max_delay_ns)
    if not bool(valid.any()):
        return torch.zeros(bins, dtype=torch.float32)
    profile = profile[valid].float()
    delay_ns = delay_ns[valid].float()
    output = torch.zeros(bins, dtype=torch.float32)
    idx = torch.floor(delay_ns / max(max_delay_ns, 1e-6) * bins).long().clamp(0, bins - 1)
    output.scatter_add_(0, idx, profile)
    total = output.sum()
    if float(total) > 0.0:
        output = output / total
    return output


def sample_target_ns(sample) -> float:
    value = getattr(sample, "first_path_delay_s", math.nan)
    try:
        value = float(value) * 1e9
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def target_bin_label_ns(target_ns: float) -> int:
    if not math.isfinite(target_ns):
        return -1
    for class_idx, (_, lower, upper) in enumerate(FIRST_DELAY_BINS_NS):
        if target_ns >= lower and (
            target_ns <= upper
            if class_idx == len(FIRST_DELAY_BINS_NS) - 1
            else target_ns < upper
        ):
            return class_idx
    return -1


def collect_features(
    path: str,
    *,
    profile_bins: int,
    max_profile_delay_ns: float,
    max_summary_delay_ns: float,
    limit_samples: int | None,
) -> dict[str, torch.Tensor | list[str]]:
    dataset = PreprocessedCSIDataset.from_pt(path)
    samples = dataset.samples[:limit_samples] if limit_samples is not None else dataset.samples
    return collect_features_from_samples(
        samples,
        source_name=path,
        profile_bins=profile_bins,
        max_profile_delay_ns=max_profile_delay_ns,
        max_summary_delay_ns=max_summary_delay_ns,
    )


def collect_features_from_samples(
    samples,
    *,
    source_name: str,
    profile_bins: int,
    max_profile_delay_ns: float,
    max_summary_delay_ns: float,
) -> dict[str, torch.Tensor | list[str]]:
    summary_features = []
    profile_features = []
    targets = []
    bin_labels = []
    los_labels = []
    skipped = 0
    skipped_out_of_bins = 0
    for sample in samples:
        target = sample_target_ns(sample)
        if not math.isfinite(target):
            skipped += 1
            continue
        bin_label = target_bin_label_ns(target)
        if bin_label < 0:
            skipped_out_of_bins += 1
            continue
        profile, delay_ns = estimated_delay_profile_from_tokens(
            sample.tokens,
            getattr(sample, "subcarrier_spacing_hz", 1.0),
        )
        summary_features.append(
            summarize_profile(profile, delay_ns, max_summary_delay_ns=max_summary_delay_ns)
        )
        profile_features.append(
            downsample_profile(
                profile,
                delay_ns,
                bins=profile_bins,
                max_delay_ns=max_profile_delay_ns,
            )
        )
        targets.append(float(target))
        bin_labels.append(bin_label)
        los_labels.append(str(sample.semantic_key.los_status))
    if not targets:
        raise ValueError(f"No valid first_path_delay_s targets found in {source_name}.")
    print(f"loaded_path={source_name}")
    print(f"loaded_samples={len(samples)}")
    print(f"valid_samples={len(targets)}")
    print(f"skipped_missing_target={skipped}")
    print(f"skipped_out_of_bins={skipped_out_of_bins}")
    return {
        "summary": torch.stack(summary_features, dim=0),
        "profile": torch.stack(profile_features, dim=0),
        "both": torch.cat(
            [torch.stack(summary_features, dim=0), torch.stack(profile_features, dim=0)],
            dim=1,
        ),
        "target": torch.tensor(targets, dtype=torch.float32),
        "bin_label": torch.tensor(bin_labels, dtype=torch.long),
        "los": los_labels,
    }


def _checkpoint_filtered_samples(
    path: str,
    checkpoint: dict,
    *,
    limit_samples: int | None,
) -> tuple[list, list]:
    dataset = PreprocessedCSIDataset.from_pt(path)
    samples = dataset.samples[:limit_samples] if limit_samples is not None else dataset.samples
    semantic_key_mode = _infer_semantic_key_mode(checkpoint, None)
    min_class_size = _infer_min_class_size(checkpoint, None)
    max_delay_spread_ns = _infer_max_delay_spread_ns(checkpoint, None)
    samples = apply_semantic_key_mode(samples, semantic_key_mode)
    samples = filter_samples_by_min_class_size(samples, min_class_size=min_class_size)
    samples = filter_samples_by_max_delay_spread(samples, max_delay_spread_ns)
    return align_samples_to_checkpoint_prototypes(samples, checkpoint)


def _build_checkpoint_model_and_loader(
    samples: list,
    checkpoint: dict,
    *,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str,
    prototype_keys_override: list | None,
) -> tuple[CSIClip, DataLoader]:
    tokenizer = build_tokenizer(samples, checkpoint)
    prototype_keys, _, _, _ = build_prototype_bank(
        samples,
        tokenizer,
        prototype_keys_override=prototype_keys_override,
    )
    attribute_fields = tuple(
        str(field)
        for field in checkpoint.get("args", {}).get("attribute_classifier_fields", ())
    )
    attribute_label_maps = (
        build_attribute_label_maps(samples, attribute_fields)
        if attribute_fields
        else {}
    )
    model = CSIClip(
        CSIEncoder(
            d_token=8,
            d_model=384,
            d_clip=256,
            token_norm_mode=_infer_token_norm_mode(checkpoint, None),
        ),
        PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300)),
        num_prototypes=len(prototype_keys),
        semantic_num_classes=len(prototype_keys),
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        use_power_branch=_infer_use_power_branch(checkpoint, None),
        use_delay_spread_head=_infer_use_delay_spread_head(checkpoint),
        use_delay_specific_encoder=_infer_use_delay_specific_encoder(checkpoint),
        attribute_num_classes={
            field: len(label_map)
            for field, label_map in attribute_label_maps.items()
        },
    ).to(device)
    assert_checkpoint_prototype_compatibility(
        checkpoint,
        prototype_keys,
        expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
        context=f"late-fusion checkpoint {checkpoint_path}",
    )
    _load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()
    loader = DataLoader(
        PreprocessedCSIDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
    return model, loader


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def extract_model_first_delay_outputs(
    model: CSIClip,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor | list[str]]:
    logits = []
    positions = []
    soft_fused_raw = []
    targets = []
    labels = []
    los_labels = []
    first_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
    for batch in loader:
        batch = _move_batch(batch, device)
        csi_features_raw = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
        )
        first_path_delay_context = None
        if hasattr(model, "encode_first_path_delay_context"):
            first_path_delay_context = model.encode_first_path_delay_context(
                batch["tokens"],
                batch["token_mask"],
                subcarrier_spacing=batch.get("subcarrier_spacing"),
            )
        physics_outputs = model.predict_physics_components(
            csi_features_raw,
            first_path_delay_context=first_path_delay_context,
        )
        raw_target = batch["physics_raw_targets"][:, first_delay_idx]
        mask = batch["physics_target_mask"][:, first_delay_idx].bool() & torch.isfinite(raw_target)
        if not bool(mask.any()):
            continue
        batch_targets = raw_target[mask].detach().cpu()
        batch_labels = torch.tensor(
            [target_bin_label_ns(float(value)) for value in batch_targets.tolist()],
            dtype=torch.long,
        )
        valid_bin_mask = batch_labels >= 0
        if not bool(valid_bin_mask.any()):
            continue
        logits.append(
            physics_outputs["first_path_delay_bin_logits"][mask][valid_bin_mask].detach().cpu()
        )
        positions.append(
            physics_outputs["first_path_delay_bin_position"][mask][valid_bin_mask].detach().cpu()
        )
        soft_fused_raw.append(
            physics_outputs["first_path_delay_bin_soft_fused_raw"][mask][valid_bin_mask].detach().cpu()
        )
        targets.append(batch_targets[valid_bin_mask])
        labels.append(batch_labels[valid_bin_mask])
        selected_indices = torch.nonzero(mask, as_tuple=False).squeeze(1).detach().cpu()
        selected_indices = selected_indices[valid_bin_mask]
        los_labels.extend(
            str(batch["semantic_keys"][int(idx)].los_status)
            for idx in selected_indices.tolist()
        )
    if not logits:
        raise ValueError("No valid first_path_delay bin targets found for late fusion.")
    return {
        "logits": torch.cat(logits, dim=0),
        "positions": torch.cat(positions, dim=0),
        "soft_fused_raw": torch.cat(soft_fused_raw, dim=0),
        "target": torch.cat(targets, dim=0),
        "bin_label": torch.cat(labels, dim=0),
        "los": los_labels,
    }


def scale_aux_logits_to_model_logits(
    model_logits: torch.Tensor,
    aux_logits: torch.Tensor,
) -> torch.Tensor:
    model_centered = model_logits.float() - model_logits.float().mean(dim=1, keepdim=True)
    aux_centered = aux_logits.float() - aux_logits.float().mean(dim=1, keepdim=True)
    model_scale = model_centered.std(correction=0).clamp(min=1e-6)
    aux_scale = aux_centered.std(correction=0).clamp(min=1e-6)
    return aux_centered * (model_scale / aux_scale)


def build_hybrid_late_fusion_logits(
    *,
    aux_logits: torch.Tensor,
    raw_aux_logits: torch.Tensor,
    fusion_mode: str,
    tail_indices: torch.Tensor,
    tail_prob_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tail_probs = torch.softmax(raw_aux_logits.float(), dim=1)[:, tail_indices].sum(dim=1)
    if fusion_mode == "global":
        gate = torch.ones(aux_logits.shape[0], dtype=torch.bool)
        return aux_logits, gate, tail_probs

    hybrid_logits = torch.zeros_like(aux_logits)
    hybrid_logits[:, tail_indices] = aux_logits[:, tail_indices]
    gate = torch.ones(aux_logits.shape[0], dtype=torch.bool)
    if fusion_mode == "tail-gated":
        gate = tail_probs >= float(tail_prob_threshold)
        hybrid_logits = hybrid_logits * gate.to(dtype=hybrid_logits.dtype).unsqueeze(1)
    elif fusion_mode == "tail-argmax":
        predicted_bins = raw_aux_logits.float().argmax(dim=1)
        gate = (predicted_bins.unsqueeze(1) == tail_indices.unsqueeze(0)).any(dim=1)
        hybrid_logits = hybrid_logits * gate.to(dtype=hybrid_logits.dtype).unsqueeze(1)
    elif fusion_mode != "tail-only":
        raise ValueError(f"Unknown late-fusion mode: {fusion_mode}")
    return hybrid_logits, gate, tail_probs


def print_prediction_delta_metrics(
    prefix: str,
    predictions: torch.Tensor,
    model_predictions: torch.Tensor,
    soft_predictions: torch.Tensor,
    model_soft_predictions: torch.Tensor,
) -> None:
    changed = predictions.long() != model_predictions.long()
    soft_delta = (soft_predictions.float() - model_soft_predictions.float()).abs()
    print(
        f"{prefix}_changed_from_model "
        f"count={int(changed.sum().item())} "
        f"rate={_format_float(float(changed.float().mean()))} "
        f"soft_abs_delta_mean={_format_float(float(soft_delta.mean()))}"
    )


def print_delay_prediction_metrics(
    prefix: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    los_labels: list[str],
) -> None:
    errors = predictions.float() - targets.float()
    los_mask = torch.tensor([label == "los" for label in los_labels], dtype=torch.bool)
    nlos_mask = ~los_mask
    print(f"{prefix}_soft_fused_MAE={_format_float(float(errors.abs().mean()))}")
    print(f"{prefix}_soft_fused_signed_mean={_format_float(float(errors.mean()))}")
    for name, mask in (("los", los_mask), ("nlos", nlos_mask)):
        print(f"{prefix}_soft_fused_{name}_count={int(mask.sum().item())}")
        if not bool(mask.any()):
            print(f"{prefix}_soft_fused_{name}_MAE=nan")
            print(f"{prefix}_soft_fused_{name}_signed_mean=nan")
            continue
        group_errors = errors[mask]
        print(f"{prefix}_soft_fused_{name}_MAE={_format_float(float(group_errors.abs().mean()))}")
        print(f"{prefix}_soft_fused_{name}_signed_mean={_format_float(float(group_errors.mean()))}")


def run_late_fusion_diagnostic(
    *,
    checkpoint_path: str,
    train_path: str,
    eval_path: str,
    profile_bins: int,
    max_profile_delay_ns: float,
    max_summary_delay_ns: float,
    limit_train: int | None,
    limit_eval: int | None,
    batch_size: int,
    device: torch.device,
    feature_mode: str,
    ridge: float,
    bin_class_weight: str,
    alphas: tuple[float, ...],
    normalize_logits: bool,
    fusion_mode: str,
    tail_labels: tuple[str, ...],
    tail_prob_threshold: float,
    print_confusion: bool,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_samples, _ = _checkpoint_filtered_samples(
        train_path,
        checkpoint,
        limit_samples=limit_train,
    )
    eval_samples, eval_prototype_keys = _checkpoint_filtered_samples(
        eval_path,
        checkpoint,
        limit_samples=limit_eval,
    )
    train_features = collect_features_from_samples(
        train_samples,
        source_name=train_path,
        profile_bins=profile_bins,
        max_profile_delay_ns=max_profile_delay_ns,
        max_summary_delay_ns=max_summary_delay_ns,
    )
    eval_features = collect_features_from_samples(
        eval_samples,
        source_name=eval_path,
        profile_bins=profile_bins,
        max_profile_delay_ns=max_profile_delay_ns,
        max_summary_delay_ns=max_summary_delay_ns,
    )
    model, loader = _build_checkpoint_model_and_loader(
        eval_samples,
        checkpoint,
        batch_size=batch_size,
        device=device,
        checkpoint_path=checkpoint_path,
        prototype_keys_override=eval_prototype_keys,
    )
    model_outputs = extract_model_first_delay_outputs(model, loader, device)
    if model_outputs["bin_label"].shape[0] != eval_features["bin_label"].shape[0]:
        raise ValueError(
            "Late-fusion sample count mismatch: "
            f"model={model_outputs['bin_label'].shape[0]} "
            f"estimated_pdp={eval_features['bin_label'].shape[0]}."
        )
    if not torch.equal(model_outputs["bin_label"], eval_features["bin_label"]):
        raise ValueError("Late-fusion target bin labels are not aligned between model and PDP features.")
    _, pdp_logits = fit_ridge_classifier_predict(
        train_features[feature_mode],
        train_features["bin_label"],
        eval_features[feature_mode],
        ridge=ridge,
        num_classes=len(FIRST_DELAY_BINS_NS),
        class_weight=bin_class_weight,
    )
    model_logits = model_outputs["logits"].float()
    pdp_logits = pdp_logits.float()
    pdp_logits_for_fusion = (
        scale_aux_logits_to_model_logits(model_logits, pdp_logits)
        if normalize_logits
        else pdp_logits
    )
    label_to_idx = {label: idx for idx, label in enumerate(FIRST_DELAY_BIN_LABELS)}
    tail_indices = torch.tensor([label_to_idx[label] for label in tail_labels], dtype=torch.long)
    hybrid_pdp_logits, hybrid_gate, pdp_tail_probs = build_hybrid_late_fusion_logits(
        aux_logits=pdp_logits_for_fusion,
        raw_aux_logits=pdp_logits,
        fusion_mode=fusion_mode,
        tail_indices=tail_indices,
        tail_prob_threshold=tail_prob_threshold,
    )
    pdp_tail_argmax = (pdp_logits.argmax(dim=1).unsqueeze(1) == tail_indices.unsqueeze(0)).any(dim=1)
    print("late_fusion_logit_normalization=" + ("pdp_centered_scaled_to_model_std" if normalize_logits else "none"))
    print(
        f"late_fusion_feature_mode={feature_mode} "
        f"ridge={_format_float(ridge)} "
        f"bin_class_weight={bin_class_weight}"
    )
    print(
        f"late_fusion_mode={fusion_mode} "
        f"tail_labels={','.join(tail_labels)} "
        f"tail_prob_threshold={_format_float(tail_prob_threshold)}"
    )
    print(
        "late_fusion_hybrid_gate "
        f"count={int(hybrid_gate.sum().item())} "
        f"rate={_format_float(float(hybrid_gate.float().mean()))} "
        f"pdp_tail_argmax_count={int(pdp_tail_argmax.sum().item())} "
        f"pdp_tail_prob_mean={_format_float(float(pdp_tail_probs.mean()))} "
        f"pdp_tail_prob_max={_format_float(float(pdp_tail_probs.max()))}"
    )
    model_predictions = model_logits.argmax(dim=1)
    print_bin_classification_metrics(
        "late_fusion_model_only",
        model_predictions,
        model_outputs["bin_label"],
        model_outputs["los"],
        print_confusion=print_confusion,
    )
    print_delay_prediction_metrics(
        "late_fusion_model_only",
        model_outputs["soft_fused_raw"],
        model_outputs["target"],
        model_outputs["los"],
    )
    print_bin_classification_metrics(
        "late_fusion_pdp_only",
        pdp_logits.argmax(dim=1),
        model_outputs["bin_label"],
        model_outputs["los"],
        print_confusion=print_confusion,
    )
    for alpha in alphas:
        fused_logits = model_logits + float(alpha) * hybrid_pdp_logits
        fused_predictions = fused_logits.argmax(dim=1)
        fused_soft_raw = fuse_first_path_delay_soft_from_bin_position(
            fused_logits,
            model_outputs["positions"],
        )
        alpha_label = _ridge_label(alpha)
        prefix = f"late_fusion_alpha{alpha_label}"
        print_prediction_delta_metrics(
            prefix,
            fused_predictions,
            model_predictions,
            fused_soft_raw,
            model_outputs["soft_fused_raw"],
        )
        print_bin_classification_metrics(
            prefix,
            fused_predictions,
            model_outputs["bin_label"],
            model_outputs["los"],
            print_confusion=print_confusion,
        )
        print_delay_prediction_metrics(
            prefix,
            fused_soft_raw,
            model_outputs["target"],
            model_outputs["los"],
        )


def _standardize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = train_x.double()
    eval_x = eval_x.double()
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, correction=0).clamp(min=1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def fit_ridge_predict(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor:
    train_z, eval_z = _standardize(train_x, eval_x)
    train_y = train_y.double()
    design = torch.cat(
        [torch.ones(train_z.shape[0], 1, dtype=train_z.dtype), train_z],
        dim=1,
    )
    eval_design = torch.cat(
        [torch.ones(eval_z.shape[0], 1, dtype=eval_z.dtype), eval_z],
        dim=1,
    )
    penalty = torch.eye(design.shape[1], dtype=design.dtype)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + float(ridge) * penalty
    rhs = design.T @ train_y
    try:
        weights = torch.linalg.solve(lhs, rhs)
    except torch.linalg.LinAlgError:
        weights = torch.linalg.pinv(lhs) @ rhs
    return (eval_design @ weights).float()


def fit_ridge_classifier_predict(
    train_x: torch.Tensor,
    train_labels: torch.Tensor,
    eval_x: torch.Tensor,
    *,
    ridge: float,
    num_classes: int,
    class_weight: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_z, eval_z = _standardize(train_x, eval_x)
    train_labels = train_labels.long()
    design = torch.cat(
        [torch.ones(train_z.shape[0], 1, dtype=train_z.dtype), train_z],
        dim=1,
    )
    eval_design = torch.cat(
        [torch.ones(eval_z.shape[0], 1, dtype=eval_z.dtype), eval_z],
        dim=1,
    )
    targets = torch.zeros(train_z.shape[0], num_classes, dtype=train_z.dtype)
    targets.scatter_(1, train_labels.unsqueeze(1), 1.0)
    if class_weight == "balanced":
        counts = torch.bincount(train_labels, minlength=num_classes).to(dtype=train_z.dtype)
        weights = train_labels.numel() / (num_classes * counts.clamp(min=1.0))
        sample_weights = weights[train_labels].sqrt().unsqueeze(1)
        design = design * sample_weights
        targets = targets * sample_weights
    penalty = torch.eye(design.shape[1], dtype=design.dtype)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + float(ridge) * penalty
    rhs = design.T @ targets
    try:
        weights = torch.linalg.solve(lhs, rhs)
    except torch.linalg.LinAlgError:
        weights = torch.linalg.pinv(lhs) @ rhs
    logits = (eval_design @ weights).float()
    return logits.argmax(dim=1), logits


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    los_labels: list[str],
) -> dict[str, float]:
    predictions = predictions.float()
    targets = targets.float()
    errors = predictions - targets
    los_mask = torch.tensor([label == "los" for label in los_labels], dtype=torch.bool)
    metrics = {
        "count": float(targets.numel()),
        "MAE": float(errors.abs().mean()),
        "RMSE": float(torch.sqrt(errors.square().mean())),
        "signed_mean": float(errors.mean()),
        "pearson": _safe_pearson(predictions, targets),
        "R2": _safe_r2(predictions, targets),
    }
    for name, mask in (("los", los_mask), ("nlos", ~los_mask)):
        metrics[f"{name}_count"] = float(mask.sum().item())
        if bool(mask.any()):
            group_errors = errors[mask]
            metrics[f"{name}_MAE"] = float(group_errors.abs().mean())
            metrics[f"{name}_signed_mean"] = float(group_errors.mean())
        else:
            metrics[f"{name}_MAE"] = math.nan
            metrics[f"{name}_signed_mean"] = math.nan
    return metrics


def print_metrics(
    prefix: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    los_labels: list[str],
    *,
    output_detail: str,
) -> None:
    predictions = predictions.float()
    targets = targets.float()
    errors = predictions - targets
    los_mask = torch.tensor([label == "los" for label in los_labels], dtype=torch.bool)
    metrics = compute_metrics(predictions, targets, los_labels)
    print(
        f"{prefix} "
        f"count={int(metrics['count'])} "
        f"MAE={_format_float(metrics['MAE'])} "
        f"nlos_MAE={_format_float(metrics['nlos_MAE'])} "
        f"nlos_signed={_format_float(metrics['nlos_signed_mean'])} "
        f"los_MAE={_format_float(metrics['los_MAE'])} "
        f"pearson={_format_float(metrics['pearson'])} "
        f"R2={_format_float(metrics['R2'])}"
    )
    if output_detail == "compact":
        return

    print(f"{prefix}_RMSE={_format_float(metrics['RMSE'])}")
    print(f"{prefix}_signed_mean={_format_float(metrics['signed_mean'])}")
    for name, mask in (("los", los_mask), ("nlos", ~los_mask)):
        print(f"{prefix}_{name}_count={int(mask.sum().item())}")
        if not bool(mask.any()):
            print(f"{prefix}_{name}_MAE=nan")
            print(f"{prefix}_{name}_signed_mean=nan")
            continue
        group_errors = errors[mask]
        print(f"{prefix}_{name}_MAE={_format_float(float(group_errors.abs().mean()))}")
        print(f"{prefix}_{name}_signed_mean={_format_float(float(group_errors.mean()))}")
    if output_detail != "bins":
        return

    for bin_idx, (label, lower, upper) in enumerate(FIRST_DELAY_BINS_NS):
        upper_mask = targets <= upper if bin_idx == len(FIRST_DELAY_BINS_NS) - 1 else targets < upper
        bin_mask = (targets >= lower) & upper_mask
        for group_name, group_mask in (
            ("", bin_mask),
            ("_los", bin_mask & los_mask),
            ("_nlos", bin_mask & ~los_mask),
        ):
            name = f"{prefix}_bin_{label}{group_name}"
            print(f"{name}_count={int(group_mask.sum().item())}")
            if not bool(group_mask.any()):
                print(f"{name}_MAE=nan")
                print(f"{name}_signed_mean=nan")
                continue
            bin_errors = errors[group_mask]
            print(f"{name}_MAE={_format_float(float(bin_errors.abs().mean()))}")
            print(f"{name}_signed_mean={_format_float(float(bin_errors.mean()))}")


def _safe_accuracy(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return math.nan
    return float((predictions[mask] == labels[mask]).float().mean())


def print_bin_classification_metrics(
    prefix: str,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    los_labels: list[str],
    *,
    print_confusion: bool,
) -> None:
    predictions = predictions.long()
    labels = labels.long()
    los_mask = torch.tensor([label == "los" for label in los_labels], dtype=torch.bool)
    num_classes = len(FIRST_DELAY_BINS_NS)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true_label, pred_label in zip(labels.tolist(), predictions.tolist()):
        confusion[int(true_label), int(pred_label)] += 1
    class_counts = confusion.sum(dim=1)
    prediction_counts = confusion.sum(dim=0)
    class_accuracy = torch.full((num_classes,), float("nan"), dtype=torch.float32)
    nonempty = class_counts > 0
    class_accuracy[nonempty] = confusion.diag()[nonempty].float() / class_counts[nonempty].float()
    macro_accuracy = (
        float(class_accuracy[nonempty].mean())
        if bool(nonempty.any())
        else math.nan
    )
    accuracy = float((predictions == labels).float().mean())
    nlos_accuracy = _safe_accuracy(predictions, labels, ~los_mask)
    los_accuracy = _safe_accuracy(predictions, labels, los_mask)
    print(
        f"{prefix} "
        f"count={labels.numel()} "
        f"accuracy={_format_float(accuracy)} "
        f"macro_accuracy={_format_float(macro_accuracy)} "
        f"nlos_accuracy={_format_float(nlos_accuracy)} "
        f"los_accuracy={_format_float(los_accuracy)}"
    )
    print(
        f"{prefix}_label_order="
        + ",".join(label for label, _, _ in FIRST_DELAY_BINS_NS)
    )
    for class_idx, (label, _, _) in enumerate(FIRST_DELAY_BINS_NS):
        class_mask = labels == class_idx
        nlos_class_mask = class_mask & ~los_mask
        print(
            f"{prefix}_bin_{label} "
            f"count={int(class_counts[class_idx].item())} "
            f"pred_count={int(prediction_counts[class_idx].item())} "
            f"accuracy={_format_float(float(class_accuracy[class_idx]))} "
            f"nlos_count={int(nlos_class_mask.sum().item())} "
            f"nlos_accuracy={_format_float(_safe_accuracy(predictions, labels, nlos_class_mask))}"
        )
    if not print_confusion:
        return
    for true_idx, (true_label, _, _) in enumerate(FIRST_DELAY_BINS_NS):
        row = ",".join(
            f"{pred_label}:{int(confusion[true_idx, pred_idx].item())}"
            for pred_idx, (pred_label, _, _) in enumerate(FIRST_DELAY_BINS_NS)
        )
        print(f"{prefix}_confusion_true_{true_label}={row}")


def print_feature_correlations(features: torch.Tensor, targets: torch.Tensor) -> None:
    names = (
        "peak_delay",
        "mean_delay",
        "rms_delay",
        "cdf10",
        "cdf25",
        "cdf50",
        "cdf75",
        "cdf90",
        "first_threshold",
        "early_100",
        "early_200",
        "early_400",
        "mid_400_800",
        "tail_800",
        "peak_power",
        "entropy",
        "active_bins",
        "profile_std",
    )
    for idx, name in enumerate(names):
        print(
            f"summary_feature_{name}_pearson="
            f"{_format_float(_safe_pearson(features[:, idx], targets))}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a delay-domain PDP from CSI tokens and linear/ridge-probe "
            "its first_path_delay_ns information."
        )
    )
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--profile-bins", type=int, default=64)
    parser.add_argument("--max-profile-delay-ns", type=float, default=3000.0)
    parser.add_argument("--max-summary-delay-ns", type=float, default=3000.0)
    parser.add_argument("--ridges", default="0,0.01,0.1,1,10,100")
    parser.add_argument(
        "--probe-task",
        choices=("bin-classification", "regression", "both", "late-fusion"),
        default="bin-classification",
        help=(
            "Which diagnostic to run. bin-classification is the default because "
            "it tests whether estimated PDP can separate first-delay bins."
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=("summary", "profile", "both", "all"),
        default="all",
    )
    parser.add_argument(
        "--target-space",
        choices=("raw", "log1p", "both"),
        default="both",
    )
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for checkpoint late-fusion diagnostics.",
    )
    parser.add_argument(
        "--output-detail",
        choices=("compact", "groups", "bins"),
        default="compact",
        help=(
            "compact prints one summary line per probe. groups adds LoS/NLoS "
            "details. bins also prints per first-path-delay bin diagnostics."
        ),
    )
    parser.add_argument(
        "--print-feature-correlations",
        action="store_true",
        help="Print correlations for individual estimated-PDP summary features.",
    )
    parser.add_argument(
        "--bin-class-weight",
        choices=("none", "balanced"),
        default="balanced",
        help="Class weighting for the ridge bin classifier.",
    )
    parser.add_argument(
        "--print-bin-confusion",
        action="store_true",
        help="Print the full first-delay-bin confusion matrix for each classification probe.",
    )
    parser.add_argument(
        "--late-fusion-feature-mode",
        choices=("summary", "profile", "both"),
        default="both",
        help="Estimated-PDP feature set used by the late-fusion classifier.",
    )
    parser.add_argument(
        "--late-fusion-ridge",
        type=float,
        default=0.0,
        help="Ridge strength for the estimated-PDP classifier used in late fusion.",
    )
    parser.add_argument(
        "--late-fusion-alphas",
        default="0,0.05,0.1,0.2,0.5,1.0",
        help="Comma-separated alpha values for model_logits + alpha * estimated_pdp_logits.",
    )
    parser.add_argument(
        "--late-fusion-mode",
        choices=("global", "tail-only", "tail-gated", "tail-argmax"),
        default="global",
        help=(
            "How to use estimated-PDP logits in late fusion. global keeps the "
            "old behavior, tail-only lets PDP affect only selected delay bins, "
            "tail-gated does that only when PDP assigns enough probability "
            "to those tail bins, and tail-argmax gates on PDP's predicted bin."
        ),
    )
    parser.add_argument(
        "--late-fusion-tail-labels",
        default="1040_1280",
        help=(
            "Comma-separated first-path-delay labels that estimated PDP may "
            "modify in tail-only/tail-gated modes."
        ),
    )
    parser.add_argument(
        "--late-fusion-tail-prob-threshold",
        type=float,
        default=0.4,
        help="Minimum PDP probability mass on tail labels for tail-gated fusion.",
    )
    parser.add_argument(
        "--disable-late-fusion-logit-normalization",
        action="store_true",
        help="Disable centering/scaling estimated-PDP logits to the model-logit scale before fusion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if args.probe_task == "late-fusion":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --probe-task late-fusion.")
        if not 0.0 <= args.late_fusion_tail_prob_threshold <= 1.0:
            raise ValueError("--late-fusion-tail-prob-threshold must be in [0, 1].")
        run_late_fusion_diagnostic(
            checkpoint_path=args.checkpoint,
            train_path=args.train_path,
            eval_path=args.eval_path,
            profile_bins=args.profile_bins,
            max_profile_delay_ns=args.max_profile_delay_ns,
            max_summary_delay_ns=args.max_summary_delay_ns,
            limit_train=args.limit_train,
            limit_eval=args.limit_eval,
            batch_size=args.batch_size,
            device=device,
            feature_mode=args.late_fusion_feature_mode,
            ridge=args.late_fusion_ridge,
            bin_class_weight=args.bin_class_weight,
            alphas=parse_floats(args.late_fusion_alphas, argument_name="--late-fusion-alphas"),
            normalize_logits=not args.disable_late_fusion_logit_normalization,
            fusion_mode=args.late_fusion_mode,
            tail_labels=parse_labels(
                args.late_fusion_tail_labels,
                choices=FIRST_DELAY_BIN_LABELS,
                argument_name="--late-fusion-tail-labels",
            ),
            tail_prob_threshold=args.late_fusion_tail_prob_threshold,
            print_confusion=args.print_bin_confusion,
        )
        return

    train = collect_features(
        args.train_path,
        profile_bins=args.profile_bins,
        max_profile_delay_ns=args.max_profile_delay_ns,
        max_summary_delay_ns=args.max_summary_delay_ns,
        limit_samples=args.limit_train,
    )
    eval_data = collect_features(
        args.eval_path,
        profile_bins=args.profile_bins,
        max_profile_delay_ns=args.max_profile_delay_ns,
        max_summary_delay_ns=args.max_summary_delay_ns,
        limit_samples=args.limit_eval,
    )
    print(
        "first_path_delay_bin_order="
        + ",".join(
            f"{label}:{lower:g}-{upper:g}" if upper != float("inf") else f"{label}:{lower:g}-inf"
            for label, lower, upper in FIRST_DELAY_BINS_NS
        )
    )
    if args.print_feature_correlations:
        print_feature_correlations(eval_data["summary"], eval_data["target"])
    feature_modes = (
        ("summary", "profile", "both")
        if args.feature_mode == "all"
        else (args.feature_mode,)
    )
    ridges = parse_ridges(args.ridges)
    if args.probe_task in ("bin-classification", "both"):
        for feature_mode in feature_modes:
            for ridge in ridges:
                predictions, _ = fit_ridge_classifier_predict(
                    train[feature_mode],
                    train["bin_label"],
                    eval_data[feature_mode],
                    ridge=ridge,
                    num_classes=len(FIRST_DELAY_BINS_NS),
                    class_weight=args.bin_class_weight,
                )
                prefix = (
                    f"estimated_pdp_bin_probe_{feature_mode}_ridge"
                    f"{_ridge_label(ridge)}_{args.bin_class_weight}"
                )
                print_bin_classification_metrics(
                    prefix,
                    predictions,
                    eval_data["bin_label"],
                    eval_data["los"],
                    print_confusion=args.print_bin_confusion,
                )

    if args.probe_task not in ("regression", "both"):
        return

    target_spaces = ("raw", "log1p") if args.target_space == "both" else (args.target_space,)
    for feature_mode in feature_modes:
        for target_space in target_spaces:
            train_target = train["target"]
            if target_space == "log1p":
                train_target = torch.log1p(train_target.clamp(min=0.0))
            for ridge in ridges:
                predictions = fit_ridge_predict(
                    train[feature_mode],
                    train_target,
                    eval_data[feature_mode],
                    ridge=ridge,
                )
                if target_space == "log1p":
                    predictions = torch.expm1(predictions).clamp(min=0.0)
                prefix = (
                    f"estimated_pdp_probe_{feature_mode}_{target_space}_ridge"
                    f"{_ridge_label(ridge)}"
                )
                print_metrics(
                    prefix,
                    predictions,
                    eval_data["target"],
                    eval_data["los"],
                    output_detail=args.output_detail,
                )


if __name__ == "__main__":
    main()
