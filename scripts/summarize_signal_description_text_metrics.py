from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_METRICS = (
    ("description_factual_accuracy", "primary_factual_metrics"),
    ("description_factual_accuracy", "delay_numeric_slots"),
    ("slot_f1", "all_numeric_slots"),
    ("slot_f1", "delay_numeric_slots"),
    ("hallucination_rate", "all_numeric_slots"),
    ("hallucination_rate", "delay_numeric_slots"),
    ("numerical_slot_accuracy", "all_numeric_slots"),
    ("numerical_slot_accuracy", "delay_numeric_slots"),
    ("physical_consistency_rate", "predicted_description"),
    ("physical_consistency_violation_rate", "predicted_description"),
    ("numerical_mae", "first_path_delay_ns"),
    ("numerical_accuracy@50", "first_path_delay_ns"),
    ("numerical_mae", "los_delay_ns"),
    ("numerical_accuracy@50", "los_delay_ns"),
    ("numerical_mae", "reflection_path_count"),
)


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def load_metric_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            (row["metric"], row["field"]): row
            for row in csv.DictReader(f)
        }


def format_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Metric CSV path, optionally named as NAME=PATH.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    named_paths = [parse_input(value) for value in args.input]
    loaded = [(name, path, load_metric_rows(path)) for name, path in named_paths]
    metric_keys = list(DEFAULT_METRICS)
    for _, _, rows in loaded:
        for key in rows:
            if key not in metric_keys:
                metric_keys.append(key)

    detail_rows: list[dict[str, str]] = []
    aggregate_rows: list[dict[str, str]] = []
    for metric, field in metric_keys:
        values = []
        for name, path, rows in loaded:
            row = rows.get((metric, field))
            value = parse_float(row["value"]) if row is not None else None
            detail_rows.append(
                {
                    "group": name,
                    "source": str(path),
                    "metric": metric,
                    "field": field,
                    "value": "" if value is None else f"{value:.12g}",
                }
            )
            if value is not None:
                values.append(value)
        if values:
            mean = sum(values) / len(values)
            aggregate_rows.append(
                {
                    "group": "mean_std",
                    "source": "",
                    "metric": metric,
                    "field": field,
                    "value": format_float(mean),
                    "std": format_float(sample_std(values)),
                    "n": str(len(values)),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("group", "source", "metric", "field", "value", "std", "n")
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in detail_rows:
            writer.writerow({**{"std": "", "n": ""}, **row})
        for row in aggregate_rows:
            writer.writerow(row)
    print(f"saved_signal_description_summary_csv={output_csv}")

    output_json = Path(args.output_json) if args.output_json else output_csv.with_suffix(".json")
    output_json.write_text(
        json.dumps(
            {"details": detail_rows, "aggregate": aggregate_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved_signal_description_summary_json={output_json}")


if __name__ == "__main__":
    main()
