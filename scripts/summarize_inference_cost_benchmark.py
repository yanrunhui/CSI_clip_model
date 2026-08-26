from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PANEL_A_TOLERANCES = {
    "first_path_delay_ns": 50.0,
    "first_path_angle_deg": 15.0,
    "first_path_power_dbw": 5.0,
    "k_factor_db": 3.0,
    "reflection_count": 1.0,
}

PANEL_B_CONTINUOUS_TOLERANCES = {
    "first_path_delay_ns": 50.0,
    "los_delay_ns": 50.0,
    "delay_spread_ns": 50.0,
    "first_path_angle_deg": 15.0,
    "los_angle_deg": 15.0,
    "angle_spread_deg": 15.0,
    "first_path_power_dbw": 5.0,
    "k_factor_db": 3.0,
}

BASELINE_FIELD_MAP = {
    "first_path_delay": "first_path_delay_ns",
    "first_path_angle": "first_path_angle_deg",
    "first_path_power": "first_path_power_dbw",
    "k_factor": "k_factor_db",
    "reflection_count": "reflection_count",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def geometric_mean(values: list[float]) -> float:
    if not values or any(value < 0.0 for value in values):
        return math.nan
    if any(value == 0.0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def score_maes(maes: dict[str, float], tolerances: dict[str, float]) -> float:
    missing = [field for field in tolerances if field not in maes]
    if missing:
        raise ValueError(f"Missing MAE fields: {', '.join(missing)}")
    components = [
        1.0 / (1.0 + maes[field] / tolerance)
        for field, tolerance in tolerances.items()
    ]
    return 100.0 * geometric_mean(components)


def parse_text_metrics(path: Path) -> dict[tuple[str, str], float]:
    metrics: dict[tuple[str, str], float] = {}
    for row in read_csv(path):
        value = finite_float(row.get("value"))
        if value is not None:
            metrics[(row.get("metric", ""), row.get("field", ""))] = value
    return metrics


def parse_baseline_maes(path: Path) -> dict[str, float]:
    result = {}
    for row in read_csv(path):
        if row.get("metric") != "MAE":
            continue
        field = BASELINE_FIELD_MAP.get(row.get("target", ""))
        value = finite_float(row.get("value"))
        if field and value is not None:
            result[field] = value
    return result


def metric(metrics: dict[tuple[str, str], float], name: str, field: str) -> float:
    key = (name, field)
    if key not in metrics:
        raise ValueError(f"Missing text metric {name}/{field}")
    return metrics[key]


def panel_b_scores(metrics: dict[tuple[str, str], float]) -> dict[str, float]:
    maes = {
        field: metric(metrics, "numerical_mae", field)
        for field in PANEL_B_CONTINUOUS_TOLERANCES
    }
    continuous = score_maes(maes, PANEL_B_CONTINUOUS_TOLERANCES)
    los_component = 0.5 * (
        metric(metrics, "attribute_accuracy", "los_status")
        + metric(metrics, "categorical_macro_f1", "los_status")
    )
    structural = 100.0 * geometric_mean(
        [
            los_component,
            metric(metrics, "numerical_accuracy@1", "path_count"),
            metric(metrics, "numerical_accuracy@1", "reflection_count"),
            metric(metrics, "numerical_accuracy@1", "reflection_path_count"),
        ]
    )
    text = 100.0 * geometric_mean(
        [
            metric(
                metrics,
                "description_factual_accuracy",
                "primary_factual_metrics",
            ),
            metric(metrics, "numerical_slot_accuracy", "all_numeric_slots"),
            metric(metrics, "slot_f1", "all_numeric_slots"),
            1.0 - metric(metrics, "hallucination_rate", "all_numeric_slots"),
            metric(metrics, "numerical_slot_accuracy", "delay_numeric_slots"),
        ]
    )
    consistency = metric(
        metrics, "physical_consistency_rate", "predicted_description"
    )
    return {
        "continuous_score": continuous,
        "structural_score": structural,
        "text_score": text,
        "complete_score": 0.50 * continuous + 0.20 * structural + 0.30 * text,
        "physical_consistency": consistency,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def measured_sample_identities(cost_summary_path: Path) -> list[tuple[int, str, str]]:
    latency_path = cost_summary_path.with_name("per_sample_latency.csv")
    rows = read_csv(latency_path)
    by_repeat: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for row in rows:
        identity = (
            int(row["sample_index"]),
            str(row.get("group_id", "")),
            str(row.get("config_key", "")),
        )
        by_repeat[int(row.get("repeat", -1))].append(identity)
    if not by_repeat or sorted(by_repeat) != list(range(len(by_repeat))):
        raise ValueError(f"Invalid repeat ids in benchmark latency file: {latency_path}")
    reference: list[tuple[int, str, str]] | None = None
    for repeat, identities in sorted(by_repeat.items()):
        identities.sort(key=lambda identity: identity[0])
        if [identity[0] for identity in identities] != list(range(len(identities))):
            raise ValueError(
                f"Missing or duplicate sample_index values in repeat {repeat}: {latency_path}"
            )
        if any(not group_id or not config_key for _, group_id, config_key in identities):
            raise ValueError(
                "Missing group_id/config_key values in benchmark latency file: "
                f"{latency_path}"
            )
        if reference is None:
            reference = identities
        elif identities != reference:
            raise ValueError(
                f"Sample identities changed between repeats in {latency_path}: "
                f"repeat={repeat}"
            )
    return reference or []


def aggregate_rows(rows: list[dict[str, Any]], numeric_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    result = []
    for model, model_rows in grouped.items():
        aggregate: dict[str, Any] = {
            "row_type": "mean_std",
            "model": model,
            "seeds": ",".join(str(row["seed"]) for row in model_rows),
            "n": len(model_rows),
        }
        for count_field in ("cost_sample_count", "quality_sample_count"):
            values = {int(row[count_field]) for row in model_rows if count_field in row}
            if len(values) == 1:
                aggregate[count_field] = values.pop()
        for field in numeric_fields:
            values = [
                value
                for row in model_rows
                if (value := finite_float(row.get(field))) is not None
            ]
            aggregate[f"{field}_mean"] = statistics.mean(values) if values else math.nan
            aggregate[f"{field}_sample_std"] = (
                statistics.stdev(values) if len(values) > 1 else math.nan
            )
            if field.endswith("score"):
                aggregate[f"{field}_seed_std"] = aggregate[
                    f"{field}_sample_std"
                ]
        result.append(aggregate)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("models", [])
    if not entries:
        raise ValueError("Manifest must contain a non-empty models list.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    panel_a_rows: list[dict[str, Any]] = []
    panel_b_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    reference_identities: list[tuple[int, str, str]] | None = None
    reference_data_sha256: str | None = None
    reference_model = ""
    for entry in entries:
        model = str(entry["model"])
        seed = int(entry.get("seed", 0))
        cost_path = Path(entry["cost_summary"])
        cost = dict(json.loads(cost_path.read_text(encoding="utf-8")))
        identities = measured_sample_identities(cost_path)
        if int(cost.get("samples_per_repeat", -1)) != len(identities):
            raise ValueError(
                f"Sample-count mismatch in {cost_path}: "
                f"summary={cost.get('samples_per_repeat')} identities={len(identities)}"
            )
        if bool(cost.get("cpu_offload", False)):
            raise ValueError(f"CPU offload is not allowed: {cost_path}")
        data_sha256 = str(cost.get("data_sha256", ""))
        if len(data_sha256) != 64:
            raise ValueError(f"Missing data_sha256 in {cost_path}")
        quality_sample_count = int(entry.get("quality_sample_count", -1))
        if quality_sample_count <= 0:
            raise ValueError(
                f"Missing quality_sample_count for model={model} seed={seed}"
            )
        quality_test_sha256 = str(entry.get("quality_test_data_sha256", ""))
        if quality_test_sha256 and quality_test_sha256 != data_sha256:
            raise ValueError(
                "Cost/quality test-data SHA256 mismatch: "
                f"model={model} seed={seed} cost={data_sha256} "
                f"quality={quality_test_sha256}"
            )
        if reference_identities is None:
            reference_identities = identities
            reference_data_sha256 = data_sha256
            reference_model = model
        elif identities != reference_identities or data_sha256 != reference_data_sha256:
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(reference_identities, identities)
                    )
                    if left != right
                ),
                min(len(reference_identities), len(identities)),
            )
            raise ValueError(
                "Benchmark sample identity mismatch: "
                f"reference={reference_model} model={model} index={mismatch} "
                f"reference_count={len(reference_identities)} model_count={len(identities)} "
                f"reference_sha256={reference_data_sha256} model_sha256={data_sha256}"
            )
        cost.update(
            {
                "cost_sample_count": len(identities),
                "quality_sample_count": quality_sample_count,
                "quality_reused": bool(entry.get("quality_reused", False)),
                "quality_metrics_source": str(
                    entry.get("quality_metrics_source", entry["metrics"])
                ),
                "quality_metrics_sha256": str(
                    entry.get("quality_metrics_sha256", "")
                ),
            }
        )
        cost_rows.append(cost)
        metric_format = entry.get("metric_format", "text")
        metrics_path = Path(entry["metrics"])
        if metric_format == "baseline":
            maes = parse_baseline_maes(metrics_path)
            text_metrics = None
        elif metric_format == "text":
            text_metrics = parse_text_metrics(metrics_path)
            maes = {
                field: metric(text_metrics, "numerical_mae", field)
                for field in PANEL_A_TOLERANCES
            }
        else:
            raise ValueError(f"Unsupported metric_format={metric_format!r}")
        common_score = score_maes(maes, PANEL_A_TOLERANCES)
        if not math.isfinite(common_score):
            raise ValueError(f"Non-finite Panel A score for model={model} seed={seed}")
        panel_a_rows.append(
            {
                "row_type": "seed",
                "model": model,
                "seed": seed,
                "common_physical_score": common_score,
                "cost_sample_count": len(identities),
                "quality_sample_count": quality_sample_count,
                "median_latency_ms": cost["median_latency_ms"],
                "p95_latency_ms": cost["p95_latency_ms"],
                "peak_allocated_gb": cost["peak_allocated_gb"],
                "peak_reserved_gb": cost["peak_reserved_gb"],
                "repeat_median_std_ms": cost["repeat_median_std_ms"],
                "latency_error_lower_ms": max(
                    0.0,
                    float(cost["median_latency_ms"])
                    - float(cost["repeat_median_std_ms"]),
                ),
                "latency_error_upper_ms": (
                    float(cost["median_latency_ms"])
                    + float(cost["repeat_median_std_ms"])
                ),
                "total_parameters": cost["total_parameters"],
                "trainable_parameters": cost["trainable_parameters"],
                **{f"{field}_MAE": maes[field] for field in PANEL_A_TOLERANCES},
            }
        )
        if bool(entry.get("panel_b", False)):
            if text_metrics is None:
                raise ValueError(f"Panel B model {model} requires text metrics.")
            scores = panel_b_scores(text_metrics)
            if any(not math.isfinite(value) for value in scores.values()):
                raise ValueError(
                    f"Non-finite Panel B score for model={model} seed={seed}: {scores}"
                )
            panel_b_rows.append(
                {
                    "row_type": "seed",
                    "model": model,
                    "seed": seed,
                    **scores,
                    "cost_sample_count": len(identities),
                    "quality_sample_count": quality_sample_count,
                    "median_latency_ms": cost["median_latency_ms"],
                    "p95_latency_ms": cost["p95_latency_ms"],
                    "peak_allocated_gb": cost["peak_allocated_gb"],
                    "peak_reserved_gb": cost["peak_reserved_gb"],
                    "repeat_median_std_ms": cost["repeat_median_std_ms"],
                    "latency_error_lower_ms": max(
                        0.0,
                        float(cost["median_latency_ms"])
                        - float(cost["repeat_median_std_ms"]),
                    ),
                    "latency_error_upper_ms": (
                        float(cost["median_latency_ms"])
                        + float(cost["repeat_median_std_ms"])
                    ),
                    "total_parameters": cost["total_parameters"],
                    "trainable_parameters": cost["trainable_parameters"],
                }
            )

    panel_a_numeric = [
        "common_physical_score",
        "median_latency_ms",
        "p95_latency_ms",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "repeat_median_std_ms",
        "latency_error_lower_ms",
        "latency_error_upper_ms",
        "total_parameters",
        *[f"{field}_MAE" for field in PANEL_A_TOLERANCES],
    ]
    panel_b_numeric = [
        "continuous_score",
        "structural_score",
        "text_score",
        "complete_score",
        "physical_consistency",
        "median_latency_ms",
        "p95_latency_ms",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "repeat_median_std_ms",
        "latency_error_lower_ms",
        "latency_error_upper_ms",
        "total_parameters",
    ]
    panel_a_output = panel_a_rows + aggregate_rows(panel_a_rows, panel_a_numeric)
    panel_b_output = panel_b_rows + aggregate_rows(panel_b_rows, panel_b_numeric)
    write_rows(output / "panel_a_common_physical_scores.csv", panel_a_output)
    write_rows(output / "panel_b_complete_scores.csv", panel_b_output)
    write_rows(output / "cost_summary.csv", cost_rows)
    result = {
        "manifest": str(manifest_path),
        "panel_a_tolerances": PANEL_A_TOLERANCES,
        "panel_b_continuous_tolerances": PANEL_B_CONTINUOUS_TOLERANCES,
        "panel_a": panel_a_output,
        "panel_b": panel_b_output,
        "cost": cost_rows,
        "verified_common_sample_count": len(reference_identities or []),
        "cost_sample_count": len(reference_identities or []),
        "quality_sample_count": int(manifest.get("quality_sample_count", -1)),
        "verified_test_data_sha256": reference_data_sha256,
    }
    (output / "benchmark_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (output / "benchmark_config.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "panel_a_tolerances": PANEL_A_TOLERANCES,
                "panel_b_continuous_tolerances": PANEL_B_CONTINUOUS_TOLERANCES,
                "panel_b_weights": {
                    "continuous": 0.50,
                    "structural": 0.20,
                    "text": 0.30,
                },
                "sample_identity_fields": [
                    "sample_index",
                    "group_id",
                    "config_key",
                ],
                "verified_test_data_sha256": reference_data_sha256,
                "cost_sample_count": len(reference_identities or []),
                "quality_sample_count": int(
                    manifest.get("quality_sample_count", -1)
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved_panel_a={output / 'panel_a_common_physical_scores.csv'}")
    print(f"saved_panel_b={output / 'panel_b_complete_scores.csv'}")
    print(f"saved_cost_summary={output / 'cost_summary.csv'}")
    print(f"saved_benchmark_results={output / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
