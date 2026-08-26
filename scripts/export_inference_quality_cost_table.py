from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PANEL_B_METRIC_COLUMNS = {
    "first_path_delay_mae_ns": ("numerical_mae", "first_path_delay_ns"),
    "first_path_angle_mae_deg": ("numerical_mae", "first_path_angle_deg"),
    "numeric_slot_accuracy": ("numerical_slot_accuracy", "all_numeric_slots"),
    "description_factuality": (
        "description_factual_accuracy",
        "primary_factual_metrics",
    ),
    "numeric_slot_f1": ("slot_f1", "all_numeric_slots"),
    "hallucination_rate": ("hallucination_rate", "all_numeric_slots"),
}

PANEL_A_FIELDS = (
    "first_path_delay_ns",
    "first_path_angle_deg",
    "first_path_power_dbw",
    "k_factor_db",
    "reflection_count",
)

BASELINE_TARGET_MAP = {
    "first_path_delay": "first_path_delay_ns",
    "first_path_angle": "first_path_angle_deg",
    "first_path_power": "first_path_power_dbw",
    "k_factor": "k_factor_db",
    "reflection_count": "reflection_count",
}

STRICT_RECORD_KEYS = (
    ("entire_record_accuracy", "all_factual_fields"),
    ("description_entire_record_accuracy", "all_factual_fields"),
    ("entire_record_factual_accuracy", "all_factual_fields"),
    ("entire_record_factual_accuracy", "primary_factual_metrics"),
)


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_metric_csv(path: Path) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = finite_float(row.get("value"))
            if value is not None:
                result[(row.get("metric", ""), row.get("field", ""))] = value
    return result


def read_baseline_maes(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") != "MAE":
                continue
            field = BASELINE_TARGET_MAP.get(row.get("target", ""))
            value = finite_float(row.get("value"))
            if field is not None and value is not None:
                result[field] = value
    return result


def required_metric(
    metrics: dict[tuple[str, str], float],
    key: tuple[str, str],
    source: Path,
) -> float:
    if key not in metrics:
        raise ValueError(f"Missing metric {key[0]}/{key[1]} in {source}")
    return metrics[key]


def strict_record_accuracy(metrics: dict[tuple[str, str], float]) -> float | None:
    for key in STRICT_RECORD_KEYS:
        if key in metrics:
            return metrics[key]
    return None


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def aggregate(
    rows: list[dict[str, Any]],
    numeric_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    output = []
    for model, model_rows in grouped.items():
        current: dict[str, Any] = {
            "model": model,
            "seeds": ",".join(str(row["seed"]) for row in model_rows),
            "seed_count": len(model_rows),
            "quality_sample_count": min(
                int(row["quality_sample_count"]) for row in model_rows
            ),
            "cost_sample_count": min(
                int(row["cost_sample_count"]) for row in model_rows
            ),
            "cost_repeat_count": min(
                int(row["cost_repeat_count"]) for row in model_rows
            ),
        }
        for field in numeric_fields:
            values = [
                value
                for row in model_rows
                if (value := finite_float(row.get(field))) is not None
            ]
            current[f"{field}_mean"] = statistics.mean(values) if values else ""
            current[f"{field}_sample_std"] = sample_std(values) if values else ""
        output.append(current)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def format_mean_std(
    row: dict[str, Any],
    field: str,
    *,
    scale: float = 1.0,
    digits: int = 2,
) -> str:
    mean = finite_float(row.get(f"{field}_mean"))
    std = finite_float(row.get(f"{field}_sample_std"))
    if mean is None:
        return "--"
    formatted = f"{scale * mean:.{digits}f}"
    if std is not None:
        formatted += rf" $\pm$ {scale * std:.{digits}f}"
    return formatted


def write_panel_a_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Direct comparison on the five physical quantities shared by all methods. No composite score is used.}",
        r"\label{tab:common_physics_cost_direct}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Model & Delay (ns)$\downarrow$ & Angle ($^\circ$)$\downarrow$ & Power (dB)$\downarrow$ & K-factor (dB)$\downarrow$ & Refl. count$\downarrow$ & Median (ms)$\downarrow$ & P95 (ms)$\downarrow$ & VRAM (GB)$\downarrow$ & Params$\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        parameters = finite_float(row.get("total_parameters_mean"))
        if parameters is None:
            parameter_text = "--"
        elif parameters >= 1e9:
            parameter_text = f"{parameters / 1e9:.2f}B"
        else:
            parameter_text = f"{parameters / 1e6:.1f}M"
        cells = [
            latex_escape(str(row["model"])),
            format_mean_std(row, "first_path_delay_ns_mae"),
            format_mean_std(row, "first_path_angle_deg_mae"),
            format_mean_std(row, "first_path_power_dbw_mae"),
            format_mean_std(row, "k_factor_db_mae"),
            format_mean_std(row, "reflection_count_mae"),
            format_mean_std(row, "median_latency_ms", digits=1),
            format_mean_std(row, "p95_latency_ms", digits=1),
            format_mean_std(row, "peak_allocated_gb"),
            parameter_text,
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_panel_b_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Direct comparison of physical accuracy, text factuality, and inference cost. No composite score is used.}",
        r"\label{tab:quality_cost_direct}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Model & Delay MAE (ns)$\downarrow$ & Angle MAE ($^\circ$)$\downarrow$ & Numeric acc. (\%)$\uparrow$ & Factuality (\%)$\uparrow$ & Entire record (\%)$\uparrow$ & Halluc. (\%)$\downarrow$ & Median (ms)$\downarrow$ & P95 (ms)$\downarrow$ & VRAM (GB)$\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            latex_escape(str(row["model"])),
            format_mean_std(row, "first_path_delay_mae_ns"),
            format_mean_std(row, "first_path_angle_mae_deg"),
            format_mean_std(row, "numeric_slot_accuracy", scale=100.0),
            format_mean_std(row, "description_factuality", scale=100.0),
            format_mean_std(row, "entire_record_accuracy", scale=100.0),
            format_mean_std(row, "hallucination_rate", scale=100.0, digits=3),
            format_mean_std(row, "median_latency_ms", digits=1),
            format_mean_std(row, "p95_latency_ms", digits=1),
            format_mean_std(row, "peak_allocated_gb"),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export direct quality/cost results without a composite score."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--panel-a-output-csv", required=True)
    parser.add_argument("--panel-a-output-tex")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-tex")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    panel_a_rows = []
    panel_b_rows = []
    for entry in manifest.get("models", []):
        metrics_path = Path(entry["metrics"])
        cost_path = Path(entry["cost_summary"])
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        median_latency = finite_float(cost.get("median_latency_ms"))
        if median_latency is None:
            raise ValueError(f"Missing finite median_latency_ms in {cost_path}")
        common: dict[str, Any] = {
            "model": str(entry["model"]),
            "seed": int(entry.get("seed", 0)),
            "quality_sample_count": int(entry.get("quality_sample_count", -1)),
            "cost_sample_count": int(cost.get("samples_per_repeat", -1)),
            "cost_repeat_count": int(
                cost.get("repeat_count", cost.get("repeats", -1))
            ),
            "median_latency_ms": median_latency,
            "p95_latency_ms": float(cost["p95_latency_ms"]),
            "peak_allocated_gb": float(cost["peak_allocated_gb"]),
            "total_parameters": float(cost["total_parameters"]),
        }
        metric_format = str(entry.get("metric_format", "text"))
        if metric_format == "text":
            text_metrics = read_metric_csv(metrics_path)
            maes = {
                field: required_metric(
                    text_metrics,
                    ("numerical_mae", field),
                    metrics_path,
                )
                for field in PANEL_A_FIELDS
            }
        elif metric_format == "baseline":
            text_metrics = None
            maes = read_baseline_maes(metrics_path)
            missing = [field for field in PANEL_A_FIELDS if field not in maes]
            if missing:
                raise ValueError(
                    f"Missing baseline MAEs {', '.join(missing)} in {metrics_path}"
                )
        else:
            raise ValueError(f"Unsupported metric_format={metric_format!r}")

        panel_a_rows.append(
            {
                **common,
                **{f"{field}_mae": maes[field] for field in PANEL_A_FIELDS},
            }
        )

        if bool(entry.get("panel_b", False)):
            if text_metrics is None:
                raise ValueError(
                    f"Panel-B model {entry['model']} requires text metrics."
                )
            panel_b_row = {
                **common,
                "entire_record_accuracy": strict_record_accuracy(text_metrics),
            }
            for output_name, metric_key in PANEL_B_METRIC_COLUMNS.items():
                panel_b_row[output_name] = required_metric(
                    text_metrics, metric_key, metrics_path
                )
            panel_b_rows.append(panel_b_row)

    if not panel_a_rows:
        raise ValueError("Manifest contains no Panel-A models.")
    if not panel_b_rows:
        raise ValueError("Manifest contains no Panel-B text models.")

    cost_fields = [
        "median_latency_ms",
        "p95_latency_ms",
        "peak_allocated_gb",
        "total_parameters",
    ]
    panel_a_aggregated = aggregate(
        panel_a_rows,
        [*[f"{field}_mae" for field in PANEL_A_FIELDS], *cost_fields],
    )
    panel_b_aggregated = aggregate(
        panel_b_rows,
        [*PANEL_B_METRIC_COLUMNS, "entire_record_accuracy", *cost_fields],
    )

    panel_a_csv = Path(args.panel_a_output_csv)
    write_csv(panel_a_csv, panel_a_aggregated)
    panel_a_tex = (
        Path(args.panel_a_output_tex)
        if args.panel_a_output_tex
        else panel_a_csv.with_suffix(".tex")
    )
    write_panel_a_latex(panel_a_tex, panel_a_aggregated)

    output_csv = Path(args.output_csv)
    write_csv(output_csv, panel_b_aggregated)

    output_tex = Path(args.output_tex) if args.output_tex else output_csv.with_suffix(".tex")
    write_panel_b_latex(output_tex, panel_b_aggregated)
    print(f"saved_panel_a_quality_cost_table_csv={panel_a_csv}")
    print(f"saved_panel_a_quality_cost_table_tex={panel_a_tex}")
    print(f"saved_quality_cost_table_csv={output_csv}")
    print(f"saved_quality_cost_table_tex={output_tex}")
    if all(
        row.get("entire_record_accuracy_mean", "") == ""
        for row in panel_b_aggregated
    ):
        print(
            "entire_record_accuracy=unavailable; reporting description factuality "
            "instead of mislabeling aggregate factuality as strict record accuracy"
        )


if __name__ == "__main__":
    main()
