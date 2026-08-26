from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PRIMARY_METRICS = (
    "first_delay_mae_ns",
    "first_angle_mae_deg",
    "first_power_mae_db",
    "k_factor_mae_db",
    "reflection_mae",
    "numeric_accuracy",
    "los_delay_mae_ns",
    "nlos_delay_mae_ns",
    "los_angle_mae_deg",
    "nlos_angle_mae_deg",
)
ERROR_METRICS = tuple(metric for metric in PRIMARY_METRICS if metric != "numeric_accuracy")
DEGRADATION_METRICS = tuple(
    f"{metric}_relative_degradation_pct" for metric in ERROR_METRICS
) + ("numeric_accuracy_drop_percentage_points",)
AVERAGED_METRICS = (*PRIMARY_METRICS, *DEGRADATION_METRICS, "actual_snr_mean_db")


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else math.nan


def finite_std(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.stdev(finite) if len(finite) > 1 else 0.0 if finite else math.nan


def optional_float(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return math.nan
    return float(value)


def condition_sort_key(row: dict[str, Any]) -> tuple[bool, float]:
    target_snr = optional_float(row.get("target_snr_db"))
    return math.isfinite(target_snr), -target_snr if math.isfinite(target_snr) else 0.0


def aggregate_by_model_seed(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Average noise realizations within each model, then summarize model means."""
    if not runs:
        raise ValueError("No robustness runs were provided")
    required = {"model_seed", "condition", "target_snr_db", *PRIMARY_METRICS}
    missing = sorted(required - set().union(*(row.keys() for row in runs)))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        model_seed = str(run["model_seed"]).strip()
        condition = str(run["condition"]).strip()
        if not model_seed or not condition:
            raise ValueError("Every run must have non-empty model_seed and condition")
        grouped[(condition, model_seed)].append(run)

    model_means: list[dict[str, Any]] = []
    for (condition, model_seed), rows in grouped.items():
        target_values = {
            optional_float(row.get("target_snr_db")) for row in rows
        }
        finite_targets = {value for value in target_values if math.isfinite(value)}
        if len(finite_targets) > 1:
            raise ValueError(
                f"Inconsistent target_snr_db for condition={condition}, model_seed={model_seed}"
            )
        target_snr = next(iter(finite_targets)) if finite_targets else None
        noise_seeds = {
            str(row.get("noise_seed", "")).strip()
            for row in rows
            if str(row.get("noise_seed", "")).strip()
        }
        result: dict[str, Any] = {
            "condition": condition,
            "target_snr_db": target_snr,
            "model_seed": model_seed,
            "noise_run_count": len(rows),
            "noise_seed_count": len(noise_seeds),
        }
        for metric in AVERAGED_METRICS:
            values = [optional_float(row.get(metric)) for row in rows]
            result[metric] = finite_mean(values)
        model_means.append(result)

    model_means.sort(
        key=lambda row: (*condition_sort_key(row), str(row["model_seed"]))
    )

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_means:
        by_condition[str(row["condition"])].append(row)

    summaries: list[dict[str, Any]] = []
    for condition, rows in by_condition.items():
        model_seeds = {str(row["model_seed"]) for row in rows}
        result = {
            "condition": condition,
            "target_snr_db": rows[0]["target_snr_db"],
            "model_seed_count": len(model_seeds),
            "noise_runs_per_model_min": min(int(row["noise_run_count"]) for row in rows),
            "noise_runs_per_model_max": max(int(row["noise_run_count"]) for row in rows),
        }
        for metric in AVERAGED_METRICS:
            values = [optional_float(row.get(metric)) for row in rows]
            result[f"{metric}_mean"] = finite_mean(values)
            result[f"{metric}_std"] = finite_std(values)
        summaries.append(result)
    summaries.sort(key=condition_sort_key)
    return model_means, summaries


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Average noise realizations within each trained model seed, then report "
            "mean and sample standard deviation across model seeds."
        )
    )
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to the directory containing --runs-csv.",
    )
    args = parser.parse_args()
    if not args.runs_csv.is_file():
        raise FileNotFoundError(args.runs_csv)
    output_dir = args.output_dir or args.runs_csv.parent
    model_means, summaries = aggregate_by_model_seed(read_csv(args.runs_csv))
    means_path = output_dir / "noise_robustness_model_seed_means.csv"
    summary_path = output_dir / "noise_robustness_model_seed_summary.csv"
    write_csv(means_path, model_means)
    write_csv(summary_path, summaries)
    print(f"saved_model_seed_means={means_path.resolve()}")
    print(f"saved_model_seed_summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
