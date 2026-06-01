from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset
from scripts.diagnose_k_factor_information import (
    _balanced_class_weights,
    _classification_metrics,
    _distribution_features,
    _finite,
    _format_float,
    _physics_features,
    _sanitize_features,
    _standardize,
    _token_power_features,
)


DS_BIN_LABELS = ("low", "moderate", "high")


@dataclass(frozen=True)
class RegressionMatrix:
    names: list[str]
    values: torch.Tensor
    targets: torch.Tensor
    bins: list[str]


@dataclass(frozen=True)
class ClassificationMatrix:
    names: list[str]
    values: torch.Tensor
    labels: torch.Tensor


def _delay_spread_ns(sample) -> float:
    return _finite(getattr(sample, "delay_spread_s", math.nan)) * 1e9


def _delay_spread_bin(sample) -> str | None:
    label = getattr(getattr(sample, "semantic_key", None), "ds_bin", None)
    return label if label in DS_BIN_LABELS else None


def _feature_groups(sample) -> dict[str, tuple[list[str], list[float]]]:
    physics_names, physics_values = _physics_features(sample)
    profile_names, profile_values = _distribution_features(sample)
    none_names, none_values = _token_power_features(sample, "none")
    rms_names, rms_values = _token_power_features(sample, "rms")
    std_names, std_values = _token_power_features(sample, "std")

    physics_no_delay_names = []
    physics_no_delay_values = []
    for name, value in zip(physics_names, physics_values, strict=True):
        if name != "delay_spread_ns":
            physics_no_delay_names.append(name)
            physics_no_delay_values.append(value)

    groups = {
        "physics_no_delay": (physics_no_delay_names, physics_no_delay_values),
        "profile_shape": (profile_names, profile_values),
        "token_power_none": (none_names, none_values),
        "token_power_rms": (rms_names, rms_values),
        "token_power_std": (std_names, std_values),
    }
    groups["profile_plus_physics"] = (
        profile_names + physics_no_delay_names,
        profile_values + physics_no_delay_values,
    )
    groups["profile_plus_tokens"] = (
        profile_names + none_names,
        profile_values + none_values,
    )
    groups["all_no_delay"] = (
        physics_no_delay_names + profile_names + none_names,
        physics_no_delay_values + profile_values + none_values,
    )
    return groups


def build_regression_matrix(samples, group_name: str) -> RegressionMatrix:
    rows: list[list[float]] = []
    targets: list[float] = []
    bins: list[str] = []
    feature_names: list[str] | None = None
    for sample in samples:
        target = _delay_spread_ns(sample)
        if not math.isfinite(target):
            continue
        groups = _feature_groups(sample)
        if group_name not in groups:
            raise ValueError(f"Unknown feature group: {group_name}")
        names, values = groups[group_name]
        if feature_names is None:
            feature_names = list(names)
        rows.append(_sanitize_features(values))
        targets.append(target)
        bins.append(_delay_spread_bin(sample) or "unknown")
    if not rows or feature_names is None:
        raise ValueError(f"No valid delay_spread samples for group={group_name}")
    return RegressionMatrix(
        names=feature_names,
        values=torch.tensor(rows, dtype=torch.float32),
        targets=torch.tensor(targets, dtype=torch.float32),
        bins=bins,
    )


def build_classification_matrix(samples, group_name: str) -> ClassificationMatrix:
    rows: list[list[float]] = []
    labels: list[int] = []
    feature_names: list[str] | None = None
    for sample in samples:
        label = _delay_spread_bin(sample)
        if label is None:
            continue
        groups = _feature_groups(sample)
        if group_name not in groups:
            raise ValueError(f"Unknown feature group: {group_name}")
        names, values = groups[group_name]
        if feature_names is None:
            feature_names = list(names)
        rows.append(_sanitize_features(values))
        labels.append(DS_BIN_LABELS.index(label))
    if not rows or feature_names is None:
        raise ValueError(f"No valid delay_spread bin samples for group={group_name}")
    return ClassificationMatrix(
        names=feature_names,
        values=torch.tensor(rows, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
    )


def _safe_r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    total = (targets - targets.mean()).square().sum()
    if float(total) <= 0.0:
        return 0.0
    residual = (predictions - targets).square().sum()
    return float(1.0 - residual / total)


def train_linear_regressor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[torch.nn.Linear, torch.Tensor, dict[str, float]]:
    torch.manual_seed(seed)
    train_z, eval_z = _standardize(train_x, eval_x)
    target_mean = train_y.mean()
    target_std = train_y.std(correction=0).clamp(min=1e-6)
    train_target_z = (train_y - target_mean) / target_std
    model = torch.nn.Linear(train_z.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        predictions = model(train_z).squeeze(1)
        loss = F.smooth_l1_loss(predictions, train_target_z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        eval_predictions = model(eval_z).squeeze(1) * target_std + target_mean
        errors = eval_predictions - eval_y
        metrics = {
            "mae": float(errors.abs().mean()),
            "rmse": float(torch.sqrt(errors.square().mean())),
            "signed_mean": float(errors.mean()),
            "r2": _safe_r2(eval_predictions, eval_y),
            "pred_min": float(eval_predictions.min()),
            "pred_max": float(eval_predictions.max()),
        }
    return model, eval_predictions, metrics


def train_linear_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, float | list[list[int]]]:
    torch.manual_seed(seed)
    train_z, eval_z = _standardize(train_x, eval_x)
    model = torch.nn.Linear(train_z.shape[1], len(DS_BIN_LABELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = _balanced_class_weights(train_y, len(DS_BIN_LABELS))
    for _ in range(epochs):
        logits = model(train_z)
        loss = F.cross_entropy(logits, train_y, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return _classification_metrics(model(eval_z), eval_y, len(DS_BIN_LABELS))


def _top_regression_features(model: torch.nn.Linear, names: list[str], limit: int = 8) -> str:
    weights = model.weight.detach().squeeze(0)
    top = torch.topk(weights.abs(), k=min(limit, weights.numel())).indices.tolist()
    return ",".join(f"{names[idx]}:{float(weights[idx]):.3f}" for idx in top)


def _print_distribution(samples, prefix: str) -> None:
    values = torch.tensor(
        [_delay_spread_ns(sample) for sample in samples if math.isfinite(_delay_spread_ns(sample))],
        dtype=torch.float32,
    )
    bins = Counter(_delay_spread_bin(sample) or "unknown" for sample in samples)
    print(f"{prefix}_count={values.numel()}")
    if values.numel() == 0:
        return
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    print(
        f"{prefix}_delay_spread_ns_range={float(values.min()):.4f},{float(values.max()):.4f}"
    )
    print(
        f"{prefix}_delay_spread_ns_mean={float(values.mean()):.4f} "
        f"std={float(values.std(correction=0)):.4f}"
    )
    print(
        f"{prefix}_delay_spread_ns_quantiles="
        + ",".join(f"{float(value):.4f}" for value in quantiles)
    )
    print(
        f"{prefix}_ds_bin_histogram="
        + ",".join(f"{label}:{bins[label]}" for label in sorted(bins))
    )


def run_diagnostics(
    train_samples,
    eval_samples,
    *,
    groups: list[str],
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> None:
    _print_distribution(train_samples, "train")
    _print_distribution(eval_samples, "eval")
    print(f"ds_bin_label_order={','.join(DS_BIN_LABELS)}")

    best_regression: tuple[str, dict[str, float]] | None = None
    for group in groups:
        train_matrix = build_regression_matrix(train_samples, group)
        eval_matrix = build_regression_matrix(eval_samples, group)
        model, predictions, metrics = train_linear_regressor(
            train_matrix.values,
            train_matrix.targets,
            eval_matrix.values,
            eval_matrix.targets,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
        )
        if best_regression is None or metrics["mae"] < best_regression[1]["mae"]:
            best_regression = (group, metrics)
        fields = [
            f"regression_group={group}",
            f"dim={train_matrix.values.shape[1]}",
            f"train_samples={train_matrix.values.shape[0]}",
            f"eval_samples={eval_matrix.values.shape[0]}",
            f"MAE={metrics['mae']:.4f}",
            f"RMSE={metrics['rmse']:.4f}",
            f"signed_mean={metrics['signed_mean']:.4f}",
            f"R2={metrics['r2']:.4f}",
            f"pred_range={metrics['pred_min']:.4f},{metrics['pred_max']:.4f}",
        ]
        print(" ".join(fields))
        for label in (*DS_BIN_LABELS, "unknown"):
            mask = torch.tensor([item == label for item in eval_matrix.bins], dtype=torch.bool)
            if bool(mask.any()):
                errors = predictions[mask] - eval_matrix.targets[mask]
                print(
                    f"regression_group={group} bin={label} "
                    f"count={int(mask.sum())} MAE={float(errors.abs().mean()):.4f} "
                    f"signed_mean={float(errors.mean()):.4f}"
                )
        print(f"regression_group={group} top_abs_features={_top_regression_features(model, train_matrix.names)}")

        train_cls = build_classification_matrix(train_samples, group)
        eval_cls = build_classification_matrix(eval_samples, group)
        cls_metrics = train_linear_classifier(
            train_cls.values,
            train_cls.labels,
            eval_cls.values,
            eval_cls.labels,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
        )
        print(
            f"classification_group={group} "
            f"acc={_format_float(cls_metrics['acc'])} "
            f"balanced_acc={_format_float(cls_metrics['balanced_acc'])} "
            f"macro_f1={_format_float(cls_metrics['macro_f1'])} "
            f"confusion={cls_metrics['confusion']}"
        )

    if best_regression is not None:
        group, metrics = best_regression
        print(
            f"diagnosis_best_regression_group={group} "
            f"MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}"
        )
        if metrics["r2"] < 0.20:
            print(
                "diagnosis_hint=linear_features_do_not_explain_delay_spread_well;"
                "check_label_noise_or_need_stronger_non_linear_head"
            )
        elif group in {"profile_shape", "profile_plus_tokens", "all_no_delay"}:
            print(
                "diagnosis_hint=delay_profile_features_are_predictive;"
                "delay_spread_head_should_use_power_context_and_profile_features"
            )
        else:
            print(
                "diagnosis_hint=delay_spread_signal_exists_but_may_be_carried_by_correlated_physics;"
                "compare_profile_shape_vs_physics_no_delay_before changing loss weights"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose delay_spread_ns distribution and simple-feature predictability."
    )
    parser.add_argument(
        "--train-path",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_400k_coarse_k_cap5000_drop8_23421_train.pt",
    )
    parser.add_argument(
        "--eval-path",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_400k_coarse_k_cap5000_drop8_23421_test.pt",
    )
    parser.add_argument(
        "--groups",
        default=(
            "physics_no_delay,profile_shape,token_power_std,token_power_rms,"
            "token_power_none,profile_plus_physics,profile_plus_tokens,all_no_delay"
        ),
        help="Comma-separated feature groups.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_samples = PreprocessedCSIDataset.from_pt(args.train_path).samples
    eval_samples = PreprocessedCSIDataset.from_pt(args.eval_path).samples
    if args.limit_train is not None:
        train_samples = train_samples[: args.limit_train]
    if args.limit_eval is not None:
        eval_samples = eval_samples[: args.limit_eval]
    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    print(f"train_path={args.train_path}")
    print(f"eval_path={args.eval_path}")
    print(f"train_samples={len(train_samples)} eval_samples={len(eval_samples)}")
    run_diagnostics(
        train_samples,
        eval_samples,
        groups=groups,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
