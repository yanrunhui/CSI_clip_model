from __future__ import annotations

import argparse
import math
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (
    PHYSICS_TARGET_NAMES,
    PHYSICS_TARGET_OFFSETS,
    PHYSICS_TARGET_SCALES,
    PreprocessedCSIDataset,
    apply_semantic_key_mode,
    collate_fn,
    semantic_key_mode_choices,
)
from data.semantic_key import implied_attribute_value_filters
from models.encoder import CSIEncoder
from models.model import CSIClip
from models.text_encoder import PhysicsTextEncoder
from scripts.evaluate import (
    _infer_attribute_fields,
    _infer_attribute_remap,
    _infer_filter_attribute_values,
    _infer_max_delay_spread_ns,
    _infer_min_class_size,
    _infer_semantic_key_mode,
    _infer_token_norm_mode,
    _infer_use_delay_spread_head,
    _infer_use_power_branch,
    _load_model_state_compatible,
    align_samples_to_checkpoint_prototypes,
    build_attribute_label_maps,
    build_prototype_bank,
    build_tokenizer,
    filter_samples_by_attribute_values,
    filter_samples_by_max_delay_spread,
    filter_samples_by_min_class_size,
    parse_attribute_value_filters,
)
from scripts.pretrain import assert_checkpoint_prototype_compatibility


DELAY_SPREAD_BINS = (
    ("0_25", 0.0, 25.0),
    ("25_50", 25.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("100_200", 100.0, 200.0),
    ("200_400", 200.0, 400.0),
    ("400_plus", 400.0, float("inf")),
)


def _physics_target_index(name: str) -> int:
    return PHYSICS_TARGET_NAMES.index(name)


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


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _ridge_label(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def prepare_samples(
    data_path: str,
    checkpoint: dict | None,
    *,
    split_name: str,
    semantic_key_mode_override: str | None,
    min_class_size_override: int | None,
    filter_attribute_values_override: dict[str, tuple[str, ...]] | None,
    max_delay_spread_ns_override: float | None,
    limit_samples: int | None,
):
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    semantic_key_mode = _infer_semantic_key_mode(checkpoint, semantic_key_mode_override)
    min_class_size = _infer_min_class_size(checkpoint, min_class_size_override)
    attribute_fields = _infer_attribute_fields(checkpoint)
    attribute_remap = _infer_attribute_remap(checkpoint)
    filter_attribute_values = _infer_filter_attribute_values(
        checkpoint,
        filter_attribute_values_override,
    )
    filter_attribute_values = {
        **implied_attribute_value_filters(attribute_fields, attribute_remap),
        **filter_attribute_values,
    }
    max_delay_spread_ns = _infer_max_delay_spread_ns(
        checkpoint,
        max_delay_spread_ns_override,
    )

    samples = apply_semantic_key_mode(dataset.samples, semantic_key_mode)
    samples = filter_samples_by_min_class_size(samples, min_class_size)
    samples = filter_samples_by_attribute_values(samples, filter_attribute_values)
    samples = filter_samples_by_max_delay_spread(samples, max_delay_spread_ns)
    if limit_samples is not None:
        if limit_samples <= 0:
            raise ValueError("--limit-train/--limit-eval must be positive.")
        samples = samples[:limit_samples]
    samples, checkpoint_prototype_keys = align_samples_to_checkpoint_prototypes(
        samples,
        checkpoint,
    )
    print(f"{split_name}_path={data_path}")
    print(f"{split_name}_samples={len(samples)}")
    print(f"{split_name}_semantic_key_mode={semantic_key_mode}")
    print(f"{split_name}_min_class_size={min_class_size}")
    print(f"{split_name}_max_delay_spread_ns={max_delay_spread_ns}")
    return samples, checkpoint_prototype_keys


def build_model(
    samples,
    checkpoint: dict | None,
    checkpoint_prototype_keys,
    device: torch.device,
) -> tuple[CSIClip, object]:
    token_norm_mode = _infer_token_norm_mode(checkpoint, None)
    use_power_branch = _infer_use_power_branch(checkpoint, None)
    use_delay_spread_head = _infer_use_delay_spread_head(checkpoint)
    attribute_fields = _infer_attribute_fields(checkpoint)
    attribute_remap = _infer_attribute_remap(checkpoint)
    tokenizer = build_tokenizer(samples, checkpoint)
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
    if checkpoint is not None:
        assert_checkpoint_prototype_compatibility(
            checkpoint,
            prototype_keys,
            expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
            context="CSI delay linear probe checkpoint",
        )
        _load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()
    print(f"token_norm_mode={token_norm_mode}")
    print(f"use_power_branch={use_power_branch}")
    print(f"use_delay_spread_head={use_delay_spread_head}")
    return model, tokenizer


@torch.no_grad()
def collect_features(
    model: CSIClip,
    tokenizer,
    samples,
    *,
    batch_size: int,
    device: torch.device,
    include_head_predictions: bool,
) -> dict[str, torch.Tensor]:
    loader = DataLoader(
        PreprocessedCSIDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
    delay_idx = _physics_target_index("delay_spread_ns")
    raw_features = []
    norm_features = []
    targets = []
    head_predictions = []
    for batch in loader:
        moved = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        csi_raw = model.encode_csi(
            moved["tokens"],
            moved["beam_positions"],
            moved["token_mask"],
            moved["freq_bin"],
            moved["bw_bin"],
            moved["subcarrier_spacing"],
            normalize=False,
        )
        mask = moved["physics_target_mask"][:, delay_idx]
        if not bool(mask.any()):
            continue
        raw_features.append(csi_raw[mask].cpu())
        norm_features.append(F.normalize(csi_raw[mask], dim=-1).cpu())
        targets.append(moved["physics_raw_targets"][mask, delay_idx].cpu())
        if include_head_predictions:
            head_norm = model.csi_delay_spread_head(csi_raw).squeeze(-1)
            head_raw = (
                head_norm * PHYSICS_TARGET_SCALES[delay_idx].to(device)
                + PHYSICS_TARGET_OFFSETS[delay_idx].to(device)
            )
            head_predictions.append(head_raw[mask].cpu())

    if not raw_features:
        raise ValueError("No valid delay_spread_ns targets were found.")
    output = {
        "raw": torch.cat(raw_features, dim=0),
        "normalized": torch.cat(norm_features, dim=0),
        "target": torch.cat(targets, dim=0),
    }
    if include_head_predictions:
        output["head_prediction"] = torch.cat(head_predictions, dim=0)
    return output


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


def print_metrics(prefix: str, predictions: torch.Tensor, targets: torch.Tensor) -> None:
    predictions = predictions.float()
    targets = targets.float()
    errors = predictions - targets
    print(f"{prefix}_count={targets.numel()}")
    print(f"{prefix}_MAE={_format_float(float(errors.abs().mean()))}")
    print(f"{prefix}_RMSE={_format_float(float(torch.sqrt(errors.square().mean())))}")
    print(f"{prefix}_signed_mean={_format_float(float(errors.mean()))}")
    print(f"{prefix}_pearson={_format_float(_safe_pearson(predictions, targets))}")
    print(f"{prefix}_R2={_format_float(_safe_r2(predictions, targets))}")
    print(
        f"{prefix}_pred_range="
        f"{_format_float(float(predictions.min()))},{_format_float(float(predictions.max()))}"
    )
    print(
        f"{prefix}_target_range="
        f"{_format_float(float(targets.min()))},{_format_float(float(targets.max()))}"
    )
    for bin_idx, (label, lower, upper) in enumerate(DELAY_SPREAD_BINS):
        upper_mask = targets <= upper if bin_idx == len(DELAY_SPREAD_BINS) - 1 else targets < upper
        mask = (targets >= lower) & upper_mask
        count = int(mask.sum().item())
        print(f"{prefix}_bin_{label}_count={count}")
        if count == 0:
            print(f"{prefix}_bin_{label}_MAE=nan")
            print(f"{prefix}_bin_{label}_signed_mean=nan")
            print(f"{prefix}_bin_{label}_pred_range=nan,nan")
            continue
        bin_predictions = predictions[mask]
        bin_targets = targets[mask]
        bin_errors = bin_predictions - bin_targets
        print(f"{prefix}_bin_{label}_MAE={_format_float(float(bin_errors.abs().mean()))}")
        print(f"{prefix}_bin_{label}_signed_mean={_format_float(float(bin_errors.mean()))}")
        print(
            f"{prefix}_bin_{label}_pred_range="
            f"{_format_float(float(bin_predictions.min()))},{_format_float(float(bin_predictions.max()))}"
        )


def parse_ridges(value: str) -> tuple[float, ...]:
    ridges = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not ridges:
        raise ValueError("--ridges must contain at least one value.")
    if any(ridge < 0.0 for ridge in ridges):
        raise ValueError("--ridges must be non-negative.")
    return ridges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linear-probe CSI encoder features for delay_spread_ns prediction."
    )
    parser.add_argument(
        "--train-path",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_400k_coarse_k_cap5000_drop8_23421_train.pt",
    )
    parser.add_argument(
        "--eval-path",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_400k_coarse_k_cap5000_drop8_23421_test.pt",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridges", default="0,0.01,0.1,1,10")
    parser.add_argument(
        "--feature-mode",
        choices=("raw", "normalized", "both"),
        default="both",
    )
    parser.add_argument(
        "--target-space",
        choices=("raw", "log1p", "both"),
        default="both",
    )
    parser.add_argument(
        "--semantic-key-mode",
        choices=semantic_key_mode_choices(),
        help="Defaults to checkpoint args.",
    )
    parser.add_argument("--min-class-size", type=int, help="Defaults to checkpoint args.")
    parser.add_argument(
        "--filter-attribute-values",
        action="append",
        help="FIELD=VALUE[,VALUE...] filters. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--max-delay-spread-ns",
        type=float,
        help="Defaults to checkpoint args.",
    )
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    filter_attribute_values = (
        parse_attribute_value_filters(args.filter_attribute_values)
        if args.filter_attribute_values is not None
        else None
    )
    train_samples, checkpoint_prototype_keys = prepare_samples(
        args.train_path,
        checkpoint,
        split_name="train",
        semantic_key_mode_override=args.semantic_key_mode,
        min_class_size_override=args.min_class_size,
        filter_attribute_values_override=filter_attribute_values,
        max_delay_spread_ns_override=args.max_delay_spread_ns,
        limit_samples=args.limit_train,
    )
    eval_samples, _ = prepare_samples(
        args.eval_path,
        checkpoint,
        split_name="eval",
        semantic_key_mode_override=args.semantic_key_mode,
        min_class_size_override=args.min_class_size,
        filter_attribute_values_override=filter_attribute_values,
        max_delay_spread_ns_override=args.max_delay_spread_ns,
        limit_samples=args.limit_eval,
    )
    model, tokenizer = build_model(
        train_samples,
        checkpoint,
        checkpoint_prototype_keys,
        device,
    )
    include_head_predictions = _infer_use_delay_spread_head(checkpoint)
    train = collect_features(
        model,
        tokenizer,
        train_samples,
        batch_size=args.batch_size,
        device=device,
        include_head_predictions=False,
    )
    eval_data = collect_features(
        model,
        tokenizer,
        eval_samples,
        batch_size=args.batch_size,
        device=device,
        include_head_predictions=include_head_predictions,
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"train_valid_delay_count={train['target'].numel()}")
    print(f"eval_valid_delay_count={eval_data['target'].numel()}")
    print(
        "delay_spread_bin_order="
        + ",".join(
            f"{label}:{lower:g}-{upper:g}" if upper != float("inf") else f"{label}:{lower:g}-inf"
            for label, lower, upper in DELAY_SPREAD_BINS
        )
    )
    if include_head_predictions:
        print_metrics(
            "checkpoint_csi_delay_head",
            eval_data["head_prediction"],
            eval_data["target"],
        )

    feature_modes = ("raw", "normalized") if args.feature_mode == "both" else (args.feature_mode,)
    target_spaces = ("raw", "log1p") if args.target_space == "both" else (args.target_space,)
    for feature_mode in feature_modes:
        for target_space in target_spaces:
            train_target = train["target"]
            if target_space == "log1p":
                train_target = torch.log1p(train_target.clamp(min=0.0))
            for ridge in parse_ridges(args.ridges):
                predictions = fit_ridge_predict(
                    train[feature_mode],
                    train_target,
                    eval_data[feature_mode],
                    ridge=ridge,
                )
                if target_space == "log1p":
                    predictions = torch.expm1(predictions).clamp(min=0.0)
                prefix = (
                    f"linear_probe_{feature_mode}_{target_space}_ridge{_ridge_label(ridge)}"
                )
                print_metrics(prefix, predictions, eval_data["target"])


if __name__ == "__main__":
    main()
