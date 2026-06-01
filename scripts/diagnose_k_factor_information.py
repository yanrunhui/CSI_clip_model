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

STRONG_K_BINS = (
    ("low", 3.0, 15.0),
    ("mid", 15.0, 30.0),
    ("high", 30.0, 45.0),
    ("very_high", 45.0, 70.0),
)


@dataclass(frozen=True)
class FeatureMatrix:
    names: list[str]
    values: torch.Tensor
    labels: torch.Tensor


def _finite(value: float | int) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _safe_log10_power(dbw: float) -> float:
    return 10.0 ** (dbw / 10.0) if math.isfinite(dbw) else math.nan


def _sanitize_features(values: list[float]) -> list[float]:
    return [float(value) if math.isfinite(float(value)) else 0.0 for value in values]


def _k_strong_bin(k_factor_db: float) -> str | None:
    if not math.isfinite(k_factor_db):
        return None
    for idx, (label, lower, upper) in enumerate(STRONG_K_BINS):
        upper_ok = k_factor_db <= upper if idx == len(STRONG_K_BINS) - 1 else k_factor_db < upper
        if k_factor_db >= lower and upper_ok:
            return label
    return None


def _normalize_tokens(tokens: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    tokens = tokens.float()
    if mode == "none":
        return tokens
    if tokens.numel() == 0:
        return tokens
    if mode == "rms":
        rms = torch.sqrt(tokens.square().mean().clamp(min=eps ** 2))
        return tokens / rms
    if mode == "std":
        centered = tokens - tokens.mean()
        std = torch.sqrt(centered.square().mean().clamp(min=eps ** 2))
        return centered / std
    raise ValueError(f"Unknown token normalization mode: {mode}")


def _topk_mean(values: torch.Tensor, k: int) -> torch.Tensor:
    if values.numel() == 0:
        return torch.tensor(0.0)
    k = min(k, int(values.numel()))
    return torch.topk(values.reshape(-1), k=k).values.mean()


def _token_power_features(sample, mode: str) -> tuple[list[str], list[float]]:
    tokens = _normalize_tokens(sample.tokens[: sample.n_tokens], mode)
    if tokens.numel() == 0:
        names = [
            f"token_{mode}_abs_mean",
            f"token_{mode}_abs_std",
            f"token_{mode}_abs_max",
            f"token_{mode}_power_mean",
            f"token_{mode}_power_std",
            f"token_{mode}_power_max",
            f"token_{mode}_rms",
            f"token_{mode}_beam_power_std",
            f"token_{mode}_beam_power_max",
            f"token_{mode}_beam_peak_over_top3",
            f"token_{mode}_freq_rms_std",
            f"token_{mode}_freq_rms_max",
            f"token_{mode}_delay_power_std",
            f"token_{mode}_delay_power_max",
            f"token_{mode}_delay_peak_over_top3",
        ]
        return names, [0.0] * len(names)

    abs_tokens = tokens.abs()
    power = tokens.square()
    beam_power = power.mean(dim=(1, 2))
    freq_rms = torch.sqrt(power.mean(dim=(0, 1)).clamp(min=1e-12))

    delay_power = torch.zeros(1, dtype=tokens.dtype)
    if tokens.shape[1] % 2 == 0:
        half = tokens.shape[1] // 2
        complex_tokens = torch.complex(tokens[:, :half, :], tokens[:, half : half * 2, :])
        delay_tokens = torch.fft.ifft(complex_tokens, dim=-1)
        delay_power = delay_tokens.abs().square().mean(dim=1)

    top3_beam = _topk_mean(beam_power, 3)
    top3_delay = _topk_mean(delay_power, 3)
    names = [
        f"token_{mode}_abs_mean",
        f"token_{mode}_abs_std",
        f"token_{mode}_abs_max",
        f"token_{mode}_power_mean",
        f"token_{mode}_power_std",
        f"token_{mode}_power_max",
        f"token_{mode}_rms",
        f"token_{mode}_beam_power_std",
        f"token_{mode}_beam_power_max",
        f"token_{mode}_beam_peak_over_top3",
        f"token_{mode}_freq_rms_std",
        f"token_{mode}_freq_rms_max",
        f"token_{mode}_delay_power_std",
        f"token_{mode}_delay_power_max",
        f"token_{mode}_delay_peak_over_top3",
    ]
    values = [
        float(abs_tokens.mean()),
        float(abs_tokens.std(correction=0)),
        float(abs_tokens.max()),
        float(power.mean()),
        float(power.std(correction=0)),
        float(power.max()),
        float(torch.sqrt(power.mean().clamp(min=1e-12))),
        float(beam_power.std(correction=0)),
        float(beam_power.max()),
        float(beam_power.max() / top3_beam.clamp(min=1e-12)),
        float(freq_rms.std(correction=0)),
        float(freq_rms.max()),
        float(delay_power.std(correction=0)),
        float(delay_power.max()),
        float(delay_power.max() / top3_delay.clamp(min=1e-12)),
    ]
    return names, values


def _distribution_features(sample) -> tuple[list[str], list[float]]:
    profile = sample.delay_power_profile.float().flatten()
    profile_sum = profile.sum().clamp(min=1e-12)
    profile_norm = profile / profile_sum
    idx = torch.arange(profile_norm.numel(), dtype=torch.float32)
    profile_center = float((profile_norm * idx).sum())
    profile_spread = float(torch.sqrt((profile_norm * (idx - profile_center).square()).sum()))
    entropy = float(-(profile_norm * (profile_norm + 1e-12).log()).sum())
    thirds = torch.chunk(profile_norm, 3)

    delay_map = sample.delay_power_map.float()
    map_sum = delay_map.sum().clamp(min=1e-12)
    map_norm = delay_map / map_sum
    delay_marginal = map_norm.sum(dim=1)
    power_marginal = map_norm.sum(dim=0)
    delay_idx = torch.arange(delay_marginal.numel(), dtype=torch.float32)
    power_idx = torch.arange(power_marginal.numel(), dtype=torch.float32)
    delay_center = float((delay_marginal * delay_idx).sum())
    power_center = float((power_marginal * power_idx).sum())
    map_entropy = float(-(map_norm * (map_norm + 1e-12).log()).sum())

    names = [
        "profile_peak",
        "profile_top3_sum",
        "profile_entropy",
        "profile_center",
        "profile_spread",
        "profile_early_sum",
        "profile_mid_sum",
        "profile_late_sum",
        "delay_map_peak",
        "delay_map_entropy",
        "delay_map_delay_center",
        "delay_map_power_center",
        "delay_map_delay_peak",
        "delay_map_power_peak",
    ]
    values = [
        float(profile_norm.max()),
        float(torch.topk(profile_norm, k=min(3, profile_norm.numel())).values.sum()),
        entropy,
        profile_center,
        profile_spread,
        float(thirds[0].sum()),
        float(thirds[1].sum()) if len(thirds) > 1 else 0.0,
        float(thirds[2].sum()) if len(thirds) > 2 else 0.0,
        float(map_norm.max()),
        map_entropy,
        delay_center,
        power_center,
        float(delay_marginal.max()),
        float(power_marginal.max()),
    ]
    return names, values


def _physics_features(sample) -> tuple[list[str], list[float]]:
    first_power = _finite(sample.first_path_power_dbw)
    names = [
        "n_paths",
        "delay_spread_ns",
        "azimuth_spread_deg",
        "first_path_delay_ns",
        "first_path_power_dbw",
        "first_path_power_linear",
        "reflection_count",
        "diffraction_count",
    ]
    values = [
        _finite(sample.n_paths),
        _finite(sample.delay_spread_s) * 1e9,
        _finite(sample.azimuth_spread_deg),
        _finite(sample.first_path_delay_s) * 1e9,
        first_power,
        _safe_log10_power(first_power),
        _finite(sample.reflection_count),
        _finite(sample.diffraction_count),
    ]
    return names, values


def _feature_groups(sample) -> dict[str, tuple[list[str], list[float]]]:
    physics_names, physics_values = _physics_features(sample)
    profile_names, profile_values = _distribution_features(sample)
    none_names, none_values = _token_power_features(sample, "none")
    rms_names, rms_values = _token_power_features(sample, "rms")
    std_names, std_values = _token_power_features(sample, "std")

    first_power_idx = physics_names.index("first_path_power_dbw")
    first_power = ([physics_names[first_power_idx]], [physics_values[first_power_idx]])

    groups = {
        "first_path_power": first_power,
        "physics_no_k": (physics_names, physics_values),
        "profile_shape": (profile_names, profile_values),
        "token_power_none": (none_names, none_values),
        "token_power_rms": (rms_names, rms_values),
        "token_power_std": (std_names, std_values),
    }
    groups["power_all"] = (
        groups["first_path_power"][0] + profile_names + none_names,
        groups["first_path_power"][1] + profile_values + none_values,
    )
    groups["stdtoken_plus_power"] = (
        std_names + groups["power_all"][0],
        std_values + groups["power_all"][1],
    )
    groups["all_no_k"] = (
        physics_names + profile_names + none_names,
        physics_values + profile_values + none_values,
    )
    return groups


def _task_label(sample, task: str) -> int | None:
    label = _k_strong_bin(_finite(sample.k_factor_db))
    if task == "high_vs_very_high":
        if label == "high":
            return 0
        if label == "very_high":
            return 1
        return None
    if task == "strong_4way":
        labels = [item[0] for item in STRONG_K_BINS]
        return labels.index(label) if label in labels else None
    raise ValueError(f"Unknown task: {task}")


def build_feature_matrix(samples, task: str, group_name: str) -> FeatureMatrix:
    rows: list[list[float]] = []
    labels: list[int] = []
    feature_names: list[str] | None = None
    for sample in samples:
        label = _task_label(sample, task)
        if label is None:
            continue
        groups = _feature_groups(sample)
        if group_name not in groups:
            raise ValueError(f"Unknown feature group: {group_name}")
        names, values = groups[group_name]
        if feature_names is None:
            feature_names = list(names)
        rows.append(_sanitize_features(values))
        labels.append(label)
    if not rows or feature_names is None:
        raise ValueError(f"No samples left for task={task} group={group_name}")
    return FeatureMatrix(
        names=feature_names,
        values=torch.tensor(rows, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
    )


def _standardize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, correction=0).clamp(min=1e-6)
    train_z = (train_x - mean) / std
    eval_z = (eval_x - mean) / std
    train_z = torch.nan_to_num(train_z, nan=0.0, posinf=0.0, neginf=0.0)
    eval_z = torch.nan_to_num(eval_z, nan=0.0, posinf=0.0, neginf=0.0)
    return train_z, eval_z


def _balanced_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.mean()


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.long()
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc)


def _classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, float | list[list[int]]]:
    preds = logits.argmax(dim=1)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for target, pred in zip(labels.tolist(), preds.tolist(), strict=True):
        confusion[target, pred] += 1
    acc = float((preds == labels).float().mean())
    per_class_acc = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()
    precision = confusion.diag().float() / confusion.sum(dim=0).clamp(min=1).float()
    recall = per_class_acc
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    out: dict[str, float | list[list[int]]] = {
        "acc": acc,
        "balanced_acc": float(per_class_acc.mean()),
        "macro_f1": float(f1.mean()),
        "confusion": confusion.tolist(),
    }
    if num_classes == 2:
        out["auc"] = _roc_auc(logits[:, 1] - logits[:, 0], labels)
    return out


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
) -> tuple[torch.nn.Linear, dict[str, float | list[list[int]]]]:
    torch.manual_seed(seed)
    num_classes = int(max(train_y.max(), eval_y.max()).item()) + 1
    train_z, eval_z = _standardize(train_x, eval_x)
    model = torch.nn.Linear(train_z.shape[1], num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = _balanced_class_weights(train_y, num_classes)
    for _ in range(epochs):
        logits = model(train_z)
        loss = F.cross_entropy(logits, train_y, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        metrics = _classification_metrics(model(eval_z), eval_y, num_classes)
    return model, metrics


def _format_float(value: float | object) -> str:
    if not isinstance(value, float):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _top_linear_features(model: torch.nn.Linear, names: list[str], limit: int = 8) -> tuple[str, str]:
    if model.out_features != 2:
        return "", ""
    weights = (model.weight[1] - model.weight[0]).detach()
    top_pos = torch.topk(weights, k=min(limit, weights.numel())).indices.tolist()
    top_neg = torch.topk(-weights, k=min(limit, weights.numel())).indices.tolist()
    positive = ",".join(f"{names[idx]}:{float(weights[idx]):.3f}" for idx in top_pos)
    negative = ",".join(f"{names[idx]}:{float(weights[idx]):.3f}" for idx in top_neg)
    return positive, negative


def run_task(
    train_samples,
    eval_samples,
    *,
    task: str,
    groups: list[str],
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, dict[str, float | list[list[int]]]]:
    task_labels = [label for sample in eval_samples if (label := _task_label(sample, task)) is not None]
    print(
        f"task={task} eval_label_histogram="
        + ",".join(f"{label}:{count}" for label, count in sorted(Counter(task_labels).items()))
    )
    results: dict[str, dict[str, float | list[list[int]]]] = {}
    for group in groups:
        train_matrix = build_feature_matrix(train_samples, task, group)
        eval_matrix = build_feature_matrix(eval_samples, task, group)
        model, metrics = train_linear_classifier(
            train_matrix.values,
            train_matrix.labels,
            eval_matrix.values,
            eval_matrix.labels,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
        )
        results[group] = metrics
        fields = [
            f"feature_group={group}",
            f"dim={train_matrix.values.shape[1]}",
            f"train_samples={train_matrix.values.shape[0]}",
            f"eval_samples={eval_matrix.values.shape[0]}",
            f"acc={_format_float(metrics['acc'])}",
            f"balanced_acc={_format_float(metrics['balanced_acc'])}",
            f"macro_f1={_format_float(metrics['macro_f1'])}",
        ]
        if "auc" in metrics:
            fields.append(f"auc={_format_float(metrics['auc'])}")
        fields.append(f"confusion={metrics['confusion']}")
        print(" ".join(fields))
        positive, negative = _top_linear_features(model, train_matrix.names)
        if positive:
            print(f"feature_group={group} top_very_high_features={positive}")
            print(f"feature_group={group} top_high_features={negative}")
    return results


def print_diagnosis(results: dict[str, dict[str, float | list[list[int]]]]) -> None:
    if not results:
        return
    std_acc = float(results.get("token_power_std", {}).get("balanced_acc", math.nan))
    raw_acc = float(results.get("token_power_none", {}).get("balanced_acc", math.nan))
    first_acc = float(results.get("first_path_power", {}).get("balanced_acc", math.nan))
    power_acc = float(results.get("power_all", {}).get("balanced_acc", math.nan))
    if not math.isfinite(std_acc) or not math.isfinite(power_acc):
        return
    print(
        "diagnosis_high_vs_very_high="
        f"token_power_std_bal={std_acc:.4f},"
        f"token_power_raw_bal={raw_acc:.4f},"
        f"first_path_power_bal={first_acc:.4f},"
        f"power_all_bal={power_acc:.4f}"
    )
    if power_acc - std_acc >= 0.10:
        print(
            "diagnosis_hint=power_features_separate_better_than_std_tokens;"
            "current_std_encoder_likely_discards_or_hides_useful_power_information"
        )
    elif power_acc >= 0.75 and raw_acc >= std_acc:
        print(
            "diagnosis_hint=power_features_are_predictive;"
            "wire_power_context_into_k_factor_heads_or_reduce_amplitude_normalization"
        )
    else:
        print(
            "diagnosis_hint=power_features_do_not_strongly_separate_high_vs_very_high;"
            "the_error_is_less_likely_to_be_only_missing_power_information"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether simple power/physics statistics separate K-factor bins."
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
        "--task",
        choices=("high_vs_very_high", "strong_4way", "all"),
        default="all",
    )
    parser.add_argument(
        "--groups",
        default=(
            "first_path_power,physics_no_k,profile_shape,"
            "token_power_std,token_power_rms,token_power_none,power_all,all_no_k"
        ),
        help="Comma-separated feature groups.",
    )
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_samples = PreprocessedCSIDataset.from_pt(args.train_path).samples
    eval_samples = PreprocessedCSIDataset.from_pt(args.eval_path).samples
    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    tasks = (
        ("high_vs_very_high", "strong_4way")
        if args.task == "all"
        else (args.task,)
    )
    print(f"train_path={args.train_path}")
    print(f"eval_path={args.eval_path}")
    print(f"train_samples={len(train_samples)} eval_samples={len(eval_samples)}")
    for task in tasks:
        results = run_task(
            train_samples,
            eval_samples,
            task=task,
            groups=groups,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        if task == "high_vs_very_high":
            print_diagnosis(results)


if __name__ == "__main__":
    main()
