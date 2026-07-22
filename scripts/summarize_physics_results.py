from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_physics_baselines import BASELINE_MODEL_NAMES, TARGET_SPECS  # noqa: E402


FULL_MODEL_METRICS = (
    ("first_path_delay", "MAE", "first_path_delay_context_MAE"),
    ("first_path_delay_los", "MAE", "first_path_delay_los_MAE"),
    ("first_path_delay_nlos", "MAE", "first_path_delay_nlos_MAE"),
    ("delay_spread", "MAE", "delay_spread_MAE"),
    ("k_factor", "MAE", "k_factor_db_MAE"),
    ("strong_k", "MAE", "strong_k_MAE"),
    ("first_path_power_base", "MAE", "base_first_power_MAE"),
    ("first_path_power_enhanced", "MAE", "enhanced_first_power_MAE"),
    ("first_path_power_los", "MAE", "los_first_power_MAE"),
    ("first_path_power_nlos", "MAE", "nlos_first_power_MAE"),
    ("los_delay", "MAE", "los_delay_context_MAE"),
    ("los_angle", "MAE", "los_angle_MAE"),
    ("first_path_angle_los", "MAE", "first_path_angle_los_MAE"),
    ("first_path_angle_nlos", "MAE", "first_path_angle_nlos_MAE"),
    ("reflection_count", "MAE", "reflection_count_head_MAE"),
    ("reflection_count", "accuracy", "reflection_count_head_accuracy"),
    ("reflection_count", "adjacent_accuracy", "reflection_count_head_adjacent_accuracy"),
    ("reflection_count", "pearson", "reflection_count_head_pearson"),
    ("n_paths", "MAE", "n_paths_MAE"),
    ("azimuth_spread", "MAE", "azimuth_spread_MAE"),
)


def parse_metric_lines(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics
    pattern = re.compile(r"^([A-Za-z0-9_@.-]+)=([^=\n]+)$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            metrics[match.group(1)] = match.group(2)
    return metrics


def parse_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return f"{mean:.6g}", ""
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return f"{mean:.6g}", f"{math.sqrt(variance):.6g}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def detect_seed_dirs(baseline_root: Path) -> list[Path]:
    seed_dirs = sorted(
        [path for path in baseline_root.glob("seed_*") if path.is_dir()],
        key=lambda path: int(path.name.split("_", 1)[1]) if path.name.split("_", 1)[1].isdigit() else path.name,
    )
    return seed_dirs or [baseline_root]


def expected_checkpoint_name(model_name: str, target_name: str) -> str:
    return f"{model_name}_{target_name}.pt"


def load_baseline_rows(baseline_root: Path, seeds: list[int] | None) -> tuple[list[dict[str, str]], list[str]]:
    seed_dirs = detect_seed_dirs(baseline_root)
    if seeds is not None:
        allowed = {f"seed_{seed}" for seed in seeds}
        seed_dirs = [path for path in seed_dirs if path.name in allowed]

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for seed_dir in seed_dirs:
        if seed_dir.name.startswith("seed_") and seed_dir.name[5:].isdigit():
            seed = seed_dir.name[5:]
        else:
            seed = ""
        for model_name in BASELINE_MODEL_NAMES:
            for target_name in TARGET_SPECS:
                checkpoint_path = seed_dir / expected_checkpoint_name(model_name, target_name)
                if not checkpoint_path.exists():
                    missing.append(str(checkpoint_path))
                    continue
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                metrics = checkpoint.get("metrics", {})
                args = checkpoint.get("args", {})
                row = {
                    "method": model_name,
                    "target": target_name,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path),
                    "count": str(metrics.get("count", "")),
                    "MAE": str(metrics.get("MAE", "")),
                    "RMSE": str(metrics.get("RMSE", "")),
                    "signed_mean": str(metrics.get("signed_mean", "")),
                    "pearson": str(metrics.get("pearson", "")),
                    "accuracy": str(metrics.get("accuracy", "")),
                    "adjacent_accuracy": str(metrics.get("adjacent_accuracy", "")),
                    "epochs": str(args.get("epochs", "")),
                    "batch_size": str(args.get("batch_size", "")),
                    "lr": str(args.get("lr", "")),
                    "weight_decay": str(args.get("weight_decay", "")),
                }
                rows.append(row)
    return rows, missing


def aggregate_baseline_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metrics = ("MAE", "RMSE", "signed_mean", "pearson", "accuracy", "adjacent_accuracy")
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["target"]), []).append(row)

    aggregate_rows: list[dict[str, str]] = []
    for (method, target), group_rows in sorted(groups.items()):
        aggregate = {
            "method": method,
            "target": target,
            "seeds": ",".join(row["seed"] for row in group_rows),
            "n_seeds": str(len(group_rows)),
        }
        for metric in metrics:
            values = [
                parsed
                for row in group_rows
                if (parsed := parse_float(row.get(metric, ""))) is not None
            ]
            mean, std = mean_std(values)
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
        aggregate_rows.append(aggregate)
    return aggregate_rows


def _seed_from_path(path: Path, fallback: int) -> str:
    for part in path.parts:
        if part.startswith("seed_") and part[5:].isdigit():
            return part[5:]
    return str(fallback)


def load_full_model_rows(paths: list[Path] | None) -> list[dict[str, str]]:
    if not paths:
        return []
    rows = []
    for path_idx, path in enumerate(paths):
        metrics = parse_metric_lines(path)
        seed = _seed_from_path(path, path_idx)
        for target, metric, source_name in FULL_MODEL_METRICS:
            if source_name not in metrics:
                continue
            rows.append(
                {
                    "method": "full_multitask",
                    "target": target,
                    "metric": metric,
                    "source_metric": source_name,
                    "seed": seed,
                    "value": metrics[source_name],
                    "evaluate_output": str(path),
                }
            )
    return rows


def aggregate_full_model_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["target"], row["metric"], row["source_metric"]), []).append(row)

    aggregate_rows = []
    for (target, metric, source_metric), group_rows in sorted(groups.items()):
        values = [
            parsed
            for row in group_rows
            if (parsed := parse_float(row.get("value", ""))) is not None
        ]
        mean, std = mean_std(values)
        aggregate_rows.append(
            {
                "method": "full_multitask",
                "target": target,
                "metric": metric,
                "source_metric": source_metric,
                "seeds": ",".join(row["seed"] for row in group_rows),
                "n_seeds": str(len(group_rows)),
                "mean": mean,
                "std": std,
            }
        )
    return aggregate_rows


def comparison_rows(
    baseline_aggregate_rows: list[dict[str, str]],
    full_aggregate_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in baseline_aggregate_rows:
        rows.append(
            {
                "method": row["method"],
                "target": row["target"],
                "metric": "MAE",
                "mean": row.get("MAE_mean", ""),
                "std": row.get("MAE_std", ""),
                "n_seeds": row.get("n_seeds", ""),
            }
        )
    for row in full_aggregate_rows:
        rows.append(
            {
                "method": row["method"],
                "target": row["target"],
                "metric": row["metric"],
                "mean": row["mean"],
                "std": row["std"],
                "n_seeds": row["n_seeds"],
                "source_metric": row["source_metric"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        default="artifacts/baselines_d2los_100k_upa8x8_nf128_los50k_nlos50k",
        help="Directory containing seed_*/ baseline checkpoint files.",
    )
    parser.add_argument(
        "--full-eval-output",
        nargs="+",
        help="Optional evaluate_output.txt from the full model.",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--output-dir",
        help="Where to write summary files. Defaults to --baseline-root.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write summaries even if expected baseline checkpoints are missing.",
    )
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    output_dir = Path(args.output_dir) if args.output_dir else baseline_root
    baseline_rows, missing = load_baseline_rows(baseline_root, args.seeds)
    if missing and not args.allow_missing:
        print("missing_baseline_checkpoints=" + ",".join(missing), file=sys.stderr)
        raise SystemExit(1)

    baseline_by_seed_csv = output_dir / "baseline_by_seed_summary.csv"
    baseline_by_seed_json = baseline_by_seed_csv.with_suffix(".json")
    baseline_aggregate_csv = output_dir / "baseline_aggregate_summary.csv"
    baseline_aggregate_json = baseline_aggregate_csv.with_suffix(".json")
    full_model_by_seed_csv = output_dir / "full_model_by_seed_summary.csv"
    full_model_by_seed_json = full_model_by_seed_csv.with_suffix(".json")
    full_model_csv = output_dir / "full_model_aggregate_summary.csv"
    full_model_json = full_model_csv.with_suffix(".json")
    comparison_csv = output_dir / "physics_comparison_summary.csv"
    comparison_json = comparison_csv.with_suffix(".json")

    aggregate_rows = aggregate_baseline_rows(baseline_rows)
    full_rows = load_full_model_rows(
        [Path(path) for path in args.full_eval_output]
        if args.full_eval_output
        else None
    )
    full_aggregate_rows = aggregate_full_model_rows(full_rows)
    rows_for_comparison = comparison_rows(aggregate_rows, full_aggregate_rows)

    write_csv(baseline_by_seed_csv, baseline_rows)
    write_json(baseline_by_seed_json, baseline_rows)
    write_csv(baseline_aggregate_csv, aggregate_rows)
    write_json(baseline_aggregate_json, aggregate_rows)
    if full_rows:
        write_csv(full_model_by_seed_csv, full_rows)
        write_json(full_model_by_seed_json, full_rows)
        write_csv(full_model_csv, full_aggregate_rows)
        write_json(full_model_json, full_aggregate_rows)
    write_csv(comparison_csv, rows_for_comparison)
    write_json(comparison_json, rows_for_comparison)

    print(f"baseline_rows={len(baseline_rows)}")
    print(f"missing_baseline_checkpoints={len(missing)}")
    print(f"saved_baseline_by_seed_csv={baseline_by_seed_csv}")
    print(f"saved_baseline_aggregate_csv={baseline_aggregate_csv}")
    if full_rows:
        print(f"saved_full_model_by_seed_csv={full_model_by_seed_csv}")
        print(f"saved_full_model_csv={full_model_csv}")
    print(f"saved_comparison_csv={comparison_csv}")


if __name__ == "__main__":
    main()
