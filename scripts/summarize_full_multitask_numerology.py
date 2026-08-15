from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev


PHYSICS_METRICS = (
    ("physical_description_los_status_accuracy", "up"),
    ("k_factor_db_MAE", "down"),
    ("strong_k_MAE", "down"),
    ("first_path_angle_MAE", "down"),
    ("first_path_angle_los_MAE", "down"),
    ("first_path_angle_nlos_MAE", "down"),
    ("base_first_power_MAE", "down"),
    ("final_first_power_MAE", "down"),
    ("los_final_first_power_MAE", "down"),
    ("nlos_final_first_power_MAE", "down"),
    ("azimuth_spread_MAE", "down"),
    ("delay_spread_MAE", "down"),
    ("first_path_delay_context_MAE", "down"),
    ("first_path_delay_los_MAE", "down"),
    ("first_path_delay_nlos_MAE", "down"),
    ("los_delay_context_MAE", "down"),
    ("los_angle_MAE", "down"),
    ("reflection_count_head_MAE", "down"),
    ("reflection_count_head_accuracy", "up"),
    ("reflection_count_head_adjacent_accuracy", "up"),
    ("reflection_path_count_head_MAE", "down"),
    ("reflection_path_count_head_exact_accuracy", "up"),
    ("reflection_path_count_head_within_1_accuracy", "up"),
    ("n_paths_MAE", "down"),
)

TEXT_METRICS = (
    ("description_factual_accuracy", "primary_factual_metrics", "up"),
    ("categorical_macro_f1", "all_categorical_fields", "up"),
    ("slot_f1", "all_numeric_slots", "up"),
    ("hallucination_rate", "all_numeric_slots", "down"),
    ("numerical_slot_accuracy", "all_numeric_slots", "up"),
    ("physical_consistency_rate", "predicted_description", "up"),
    ("physical_consistency_violation_rate", "predicted_description", "down"),
)


def finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_key_value_metrics(path: Path) -> dict[str, float]:
    pattern = re.compile(r"^([A-Za-z0-9_@.-]+)=([^=\n]+)$")
    metrics = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        value = finite_float(match.group(2))
        if value is not None:
            metrics[match.group(1)] = value
    return metrics


def parse_text_metrics(path: Path) -> dict[tuple[str, str], float]:
    metrics = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = finite_float(row.get("value"))
            if value is not None:
                metrics[(str(row.get("metric", "")), str(row.get("field", "")))] = value
    return metrics


def aggregate_rows(detail_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict]] = {}
    for row in detail_rows:
        key = (
            row["configuration"],
            row["evaluation_type"],
            row["section"],
            row["metric"],
            row["field"],
            row["direction"],
        )
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for (
        configuration,
        evaluation_type,
        section,
        metric,
        field,
        direction,
    ), rows in grouped.items():
        values = [float(row["value"]) for row in rows]
        aggregates.append(
            {
                "row_type": "mean_std",
                "configuration": configuration,
                "evaluation_type": evaluation_type,
                "section": section,
                "metric": metric,
                "field": field,
                "direction": direction,
                "seed": "",
                "seeds": ",".join(str(row["seed"]) for row in rows),
                "n": len(values),
                "value": "",
                "mean": mean(values),
                "sample_std": stdev(values) if len(values) > 1 else math.nan,
            }
        )
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--configuration",
        action="append",
        required=True,
        help="Evaluation directory and type as NAME=TYPE.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    if len(args.seeds) < 2:
        raise ValueError("At least two seeds are required for sample SD.")

    configurations = []
    for value in args.configuration:
        if "=" not in value:
            raise ValueError(f"--configuration must use NAME=TYPE, got {value!r}.")
        name, evaluation_type = value.split("=", 1)
        configurations.append((name.strip(), evaluation_type.strip()))

    details = []
    for configuration, evaluation_type in configurations:
        for seed in args.seeds:
            evaluation_dir = args.root / f"seed_{seed}" / "evaluation" / configuration
            physics_path = evaluation_dir / "evaluate_output.txt"
            text_path = (
                evaluation_dir / "text_metrics" / "signal_description_text_metrics.csv"
            )
            if not physics_path.exists():
                raise FileNotFoundError(physics_path)
            if not text_path.exists():
                raise FileNotFoundError(text_path)
            physics = parse_key_value_metrics(physics_path)
            text = parse_text_metrics(text_path)
            for metric, direction in PHYSICS_METRICS:
                value = physics.get(metric)
                if value is None:
                    continue
                details.append(
                    {
                        "row_type": "seed",
                        "configuration": configuration,
                        "evaluation_type": evaluation_type,
                        "section": "physics",
                        "metric": metric,
                        "field": "",
                        "direction": direction,
                        "seed": seed,
                        "seeds": "",
                        "n": "",
                        "value": value,
                        "mean": "",
                        "sample_std": "",
                    }
                )
            for metric, field, direction in TEXT_METRICS:
                value = text.get((metric, field))
                if value is None:
                    continue
                details.append(
                    {
                        "row_type": "seed",
                        "configuration": configuration,
                        "evaluation_type": evaluation_type,
                        "section": "text",
                        "metric": metric,
                        "field": field,
                        "direction": direction,
                        "seed": seed,
                        "seeds": "",
                        "n": "",
                        "value": value,
                        "mean": "",
                        "sample_std": "",
                    }
                )

    aggregates = aggregate_rows(details)
    output_prefix = (
        args.output_prefix or args.root / "full_multitask_nf128_3seed_summary"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    fieldnames = (
        "row_type",
        "configuration",
        "evaluation_type",
        "section",
        "metric",
        "field",
        "direction",
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
            {"details": details, "aggregates": aggregates}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )

    headline_metrics = {
        "physical_description_los_status_accuracy",
        "k_factor_db_MAE",
        "first_path_angle_MAE",
        "final_first_power_MAE",
        "azimuth_spread_MAE",
        "delay_spread_MAE",
        "reflection_count_head_accuracy",
        "reflection_path_count_head_exact_accuracy",
        "description_factual_accuracy",
        "hallucination_rate",
    }
    for row in aggregates:
        if row["metric"] not in headline_metrics:
            continue
        field_suffix = f"_{row['field']}" if row["field"] else ""
        print(
            f"{row['configuration']}_{row['section']}_{row['metric']}{field_suffix}="
            f"{float(row['mean']):.6g} +/- {float(row['sample_std']):.6g}"
        )
    print(f"saved_full_multitask_summary_csv={csv_path}")
    print(f"saved_full_multitask_summary_json={json_path}")


if __name__ == "__main__":
    main()
