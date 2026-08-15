from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev


KEY_METRICS = (
    "physical_description_los_status_accuracy",
    "los_angle_MAE",
    "first_path_angle_MAE",
    "first_path_angle_los_MAE",
    "first_path_angle_nlos_MAE",
    "first_path_delay_context_MAE",
    "first_path_delay_los_MAE",
    "first_path_delay_nlos_MAE",
    "los_delay_context_MAE",
    "delay_spread_MAE",
    "azimuth_spread_MAE",
    "k_factor_db_MAE",
    "strong_k_MAE",
    "base_first_power_MAE",
    "final_first_power_MAE",
    "los_final_first_power_MAE",
    "nlos_final_first_power_MAE",
    "reflection_count_head_MAE",
    "reflection_count_head_accuracy",
    "reflection_count_head_adjacent_accuracy",
    "reflection_path_count_head_MAE",
    "reflection_path_count_head_exact_accuracy",
    "reflection_path_count_head_within_1_accuracy",
    "n_paths_MAE",
    "text_description_factual_accuracy",
    "text_categorical_macro_f1",
    "text_slot_f1",
    "text_hallucination_rate",
    "text_numerical_slot_accuracy",
    "text_physical_consistency_rate",
    "text_physical_consistency_violation_rate",
)

TEXT_METRICS = (
    (
        "text_description_factual_accuracy",
        "description_factual_accuracy",
        "primary_factual_metrics",
    ),
    (
        "text_categorical_macro_f1",
        "categorical_macro_f1",
        "all_categorical_fields",
    ),
    ("text_slot_f1", "slot_f1", "all_numeric_slots"),
    ("text_hallucination_rate", "hallucination_rate", "all_numeric_slots"),
    (
        "text_numerical_slot_accuracy",
        "numerical_slot_accuracy",
        "all_numeric_slots",
    ),
    (
        "text_physical_consistency_rate",
        "physical_consistency_rate",
        "predicted_description",
    ),
    (
        "text_physical_consistency_violation_rate",
        "physical_consistency_violation_rate",
        "predicted_description",
    ),
)


def parse_config(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--config must use NAME=TYPE, got {value!r}.")
    name, evaluation_type = value.split("=", 1)
    return name.strip(), evaluation_type.strip()


def parse_metrics(path: Path) -> dict[str, float]:
    pattern = re.compile(r"^([A-Za-z0-9_@.-]+)=([^=\n]+)$")
    metrics = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if math.isfinite(value):
            metrics[match.group(1)] = value
    return metrics


def parse_text_metrics(path: Path) -> dict[str, float]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row.get("value", ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                rows[(row.get("metric", ""), row.get("field", ""))] = value
    metrics = {}
    for output_name, metric, field in TEXT_METRICS:
        value = rows.get((metric, field))
        if value is not None:
            metrics[output_name] = value
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    if len(args.seeds) < 2:
        raise ValueError("At least two seeds are required for sample SD.")
    configurations = [parse_config(value) for value in args.config]
    details = []
    aggregates = []
    for config_name, evaluation_type in configurations:
        seed_metrics = {}
        for seed in args.seeds:
            evaluation_dir = args.root / f"seed_{seed}" / "evaluation" / config_name
            physics_path = evaluation_dir / "evaluate_output.txt"
            text_path = (
                evaluation_dir / "text_metrics" / "signal_description_text_metrics.csv"
            )
            if not physics_path.exists():
                raise FileNotFoundError(physics_path)
            if not text_path.exists():
                raise FileNotFoundError(text_path)
            seed_metrics[seed] = {
                **parse_metrics(physics_path),
                **parse_text_metrics(text_path),
            }
        for metric in KEY_METRICS:
            values = []
            for seed in args.seeds:
                value = seed_metrics[seed].get(metric)
                details.append(
                    {
                        "row_type": "seed",
                        "configuration": config_name,
                        "evaluation_type": evaluation_type,
                        "metric": metric,
                        "seed": seed,
                        "value": "" if value is None else value,
                    }
                )
                if value is not None:
                    values.append(value)
            if values:
                aggregates.append(
                    {
                        "row_type": "mean_std",
                        "configuration": config_name,
                        "evaluation_type": evaluation_type,
                        "metric": metric,
                        "seeds": ",".join(str(seed) for seed in args.seeds),
                        "n": len(values),
                        "mean": mean(values),
                        "sample_std": stdev(values) if len(values) > 1 else math.nan,
                    }
                )

    output_prefix = args.output_prefix or args.root / "cross_array_3seed_summary"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    fieldnames = (
        "row_type",
        "configuration",
        "evaluation_type",
        "metric",
        "seed",
        "seeds",
        "n",
        "value",
        "mean",
        "sample_std",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)
        writer.writerows(aggregates)
    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {"details": details, "aggregates": aggregates},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for row in aggregates:
        if row["metric"] not in {
            "los_angle_MAE",
            "first_path_angle_los_MAE",
            "first_path_angle_nlos_MAE",
            "first_path_delay_los_MAE",
            "first_path_delay_nlos_MAE",
            "k_factor_db_MAE",
            "text_description_factual_accuracy",
            "text_hallucination_rate",
        }:
            continue
        print(
            f"{row['configuration']}_{row['metric']}="
            f"{row['mean']:.4f} +/- {row['sample_std']:.4f}"
        )
    print(f"saved_cross_array_summary_csv={csv_path}")
    print(f"saved_cross_array_summary_json={json_path}")


if __name__ == "__main__":
    main()
