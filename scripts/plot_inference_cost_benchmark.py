from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_aggregates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("row_type") == "mean_std"]


def finite(value: str, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parameter_label(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.2g}B"
    return f"{value / 1e6:.1f}M"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    root = Path(args.input_dir)
    panels = [
        (
            read_aggregates(root / "panel_a_common_physical_scores.csv"),
            "common_physical_score",
            "Common Physical Score",
            "Panel A: Shared physical capability and inference cost",
        ),
        (
            read_aggregates(root / "panel_b_complete_scores.csv"),
            "complete_score",
            "Complete CSI-to-text Score",
            "Panel B: Complete physical-text capability and inference cost",
        ),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    colors = plt.cm.tab10.colors
    for axis, (rows, score_field, ylabel, title) in zip(axes, panels):
        plotted_points = []
        for index, row in enumerate(rows):
            x = finite(row.get("median_latency_ms_mean", ""))
            y = finite(row.get(f"{score_field}_mean", ""))
            yerr = finite(row.get(f"{score_field}_sample_std", ""), 0.0)
            xerr = finite(row.get("repeat_median_std_ms_mean", ""), 0.0)
            vram = finite(row.get("peak_allocated_gb_mean", ""), 0.1)
            parameters = finite(row.get("total_parameters_mean", ""), 0.0)
            size = 55.0 + 24.0 * max(vram, 0.0)
            axis.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor=colors[index % len(colors)],
                capsize=3,
                alpha=0.8,
            )
            axis.scatter(
                x,
                y,
                s=size,
                color=colors[index % len(colors)],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
            )
            label = f"{row['model']} ({parameter_label(parameters)})"
            if score_field == "complete_score":
                p95 = finite(row.get("p95_latency_ms_mean", ""))
                consistency = finite(row.get("physical_consistency_mean", ""))
                label += f"\nP95 {p95:.1f} ms, cons. {100 * consistency:.1f}%"
            axis.annotate(label, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8)
            plotted_points.append((x, y))
        frontier = []
        best_score = -math.inf
        for x, y in sorted(plotted_points):
            if y > best_score:
                frontier.append((x, y))
                best_score = y
        if len(frontier) > 1:
            axis.plot(
                [point[0] for point in frontier],
                [point[1] for point in frontier],
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label="Pareto frontier",
            )
        for vram in (4, 12, 24):
            axis.scatter(
                [],
                [],
                s=55.0 + 24.0 * vram,
                color="#888888",
                alpha=0.45,
                label=f"{vram} GB VRAM",
            )
        axis.set_xscale("log")
        axis.set_xlabel("Median end-to-end latency (ms/sample, log scale)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", linestyle=":", alpha=0.35)
        axis.legend(title="Marker area / frontier", fontsize=8, title_fontsize=8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved_benchmark_figure={output}")


if __name__ == "__main__":
    main()
