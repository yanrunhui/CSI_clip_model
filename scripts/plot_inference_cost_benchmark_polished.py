from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Style:
    label: str
    color: str
    marker: str


STYLES = {
    "Full model": Style("Full model", "#1F4E79", "star"),
    "CSI encoder + single-task": Style("CSI encoder + single-task", "#4C78A8", "square"),
    "PDP/IFFT + MLP": Style("PDP/IFFT + MLP", "#2A9D8F", "square"),
    "Flattened CSI + MLP": Style("Flattened CSI + MLP", "#6C757D", "square"),
    "Transformer without branches": Style("Transformer w/o branches", "#8E6CBB", "square"),
    "CNN baseline": Style("CNN baseline", "#9C755F", "square"),
    "Qwen3-1.7B": Style("Qwen3-1.7B", "#E69F00", "circle"),
    "Qwen3.5-2B updated direct decoder": Style("Qwen3.5-2B", "#009E73", "circle"),
    "DeepSeek-R1-Qwen3-8B": Style("DeepSeek-R1-Qwen3-8B", "#D55E00", "circle"),
    "InternLM3-8B-Instruct": Style("InternLM3-8B", "#0072B2", "circle"),
}

PANEL_A_OFFSETS = {
    "Full model": (12, -17),
    "CSI encoder + single-task": (12, 25),
    "PDP/IFFT + MLP": (12, -14),
    "Flattened CSI + MLP": (12, 29),
    "Transformer without branches": (12, 22),
    "CNN baseline": (12, -15),
    "Qwen3-1.7B": (-126, 27),
    "Qwen3.5-2B updated direct decoder": (-126, -18),
    "DeepSeek-R1-Qwen3-8B": (15, -3),
    "InternLM3-8B-Instruct": (15, -18),
}

PANEL_B_OFFSETS = {
    "Full model": (15, 4),
    "Qwen3-1.7B": (-122, 30),
    "Qwen3.5-2B updated direct decoder": (-122, -19),
    "DeepSeek-R1-Qwen3-8B": (16, -2),
    "InternLM3-8B-Instruct": (16, -14),
}


def read_aggregates(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("row_type") == "mean_std"]


def finite(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def sx(value: float, left: float, width: float, domain: tuple[float, float]) -> float:
    lo, hi = map(math.log10, domain)
    return left + width * (math.log10(value) - lo) / (hi - lo)


def sy(value: float, top: float, height: float, domain: tuple[float, float]) -> float:
    lo, hi = domain
    return top + height * (hi - value) / (hi - lo)


def marker_radius(vram_gb: float) -> float:
    return 5.5 + 3.8 * math.sqrt(max(vram_gb, 0.0))


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def text(x: float, y: float, value: str, size: float = 14, weight: int = 400,
         anchor: str = "start", fill: str = "#222222", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}"{transform}>{esc(value)}</text>'
    )


def marker(x: float, y: float, radius: float, style: Style) -> str:
    common = f'fill="{style.color}" stroke="#FFFFFF" stroke-width="1.4" opacity="0.96"'
    if style.marker == "square":
        return f'<rect x="{x-radius:.1f}" y="{y-radius:.1f}" width="{2*radius:.1f}" height="{2*radius:.1f}" rx="1.5" {common}/>'
    if style.marker == "star":
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            r = radius * (1.0 if index % 2 == 0 else 0.43)
            points.append(f"{x + r * math.cos(angle):.1f},{y + r * math.sin(angle):.1f}")
        return f'<polygon points="{" ".join(points)}" {common}/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" {common}/>'


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    best = -math.inf
    for x, y in sorted(points):
        if y > best:
            result.append((x, y))
            best = y
    return result


def draw_panel(
    rows: list[dict[str, str]],
    score_field: str,
    title: str,
    ylabel: str,
    left: float,
    top: float,
    width: float,
    height: float,
    xdomain: tuple[float, float],
    ydomain: tuple[float, float],
    xticks: list[float],
    yticks: list[float],
    offsets: dict[str, tuple[int, int]],
) -> list[str]:
    out = [text(left, top - 28, title, 18, 600)]
    out.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" height="{height:.1f}" fill="#FCFCFC" stroke="#666666" stroke-width="1"/>')

    low_decade = math.floor(math.log10(xdomain[0]))
    high_decade = math.ceil(math.log10(xdomain[1]))
    for decade in range(low_decade, high_decade + 1):
        for multiplier in range(1, 10):
            value = multiplier * (10 ** decade)
            if not xdomain[0] <= value <= xdomain[1]:
                continue
            x = sx(value, left, width, xdomain)
            major = multiplier == 1
            out.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top+height:.1f}" stroke="{"#D8D8D8" if major else "#EEEEEE"}" stroke-width="{0.9 if major else 0.55}"/>')
    for value in yticks:
        y = sy(value, top, height, ydomain)
        out.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left+width:.1f}" y2="{y:.1f}" stroke="#DEDEDE" stroke-width="0.8"/>')
        out.append(text(left - 12, y + 5, f"{value:g}", 13, anchor="end", fill="#444444"))
    for value in xticks:
        x = sx(value, left, width, xdomain)
        label = f"10^{int(math.log10(value))}"
        out.append(text(x, top + height + 24, label, 13, anchor="middle", fill="#444444"))

    out.append(text(left + width / 2, top + height + 55, "Median end-to-end latency (ms/sample; log scale)", 14, 500, anchor="middle"))
    out.append(text(left - 60, top + height / 2, ylabel, 14, 500, anchor="middle", rotate=-90))

    plotted = []
    row_points = []
    for row in rows:
        xvalue = finite(row.get("median_latency_ms_mean"))
        yvalue = finite(row.get(f"{score_field}_mean"))
        if not (math.isfinite(xvalue) and math.isfinite(yvalue)):
            continue
        x = sx(xvalue, left, width, xdomain)
        y = sy(yvalue, top, height, ydomain)
        row_points.append((row, xvalue, yvalue, x, y))
        plotted.append((xvalue, yvalue))

    frontier = pareto(plotted)
    if len(frontier) > 1:
        points = " ".join(f"{sx(x,left,width,xdomain):.1f},{sy(y,top,height,ydomain):.1f}" for x, y in frontier)
        out.append(f'<polyline points="{points}" fill="none" stroke="#4D4D4D" stroke-width="1.6" stroke-dasharray="7 5" opacity="0.75"/>')

    for row, xvalue, yvalue, x, y in row_points:
        model = row["model"]
        style = STYLES.get(model, Style(model, "#555555", "circle"))
        yerr = finite(row.get(f"{score_field}_seed_std"), 0.0)
        if yerr > 0:
            low = sy(yvalue - yerr, top, height, ydomain)
            high = sy(yvalue + yerr, top, height, ydomain)
            out.extend([
                f'<line x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" y2="{high:.1f}" stroke="{style.color}" stroke-width="1.5"/>',
                f'<line x1="{x-5:.1f}" y1="{low:.1f}" x2="{x+5:.1f}" y2="{low:.1f}" stroke="{style.color}" stroke-width="1.5"/>',
                f'<line x1="{x-5:.1f}" y1="{high:.1f}" x2="{x+5:.1f}" y2="{high:.1f}" stroke="{style.color}" stroke-width="1.5"/>',
            ])
        radius = marker_radius(finite(row.get("peak_allocated_gb_mean"), 0.0))
        out.append(marker(x, y, radius, style))
        dx, dy = offsets.get(model, (10, -10))
        tx, ty = x + dx, y + dy
        label = style.label
        out.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{tx:.1f}" y2="{ty-4:.1f}" stroke="{style.color}" stroke-width="0.9" opacity="0.55"/>')
        out.append(text(tx, ty, label, 13, 500))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True, help="SVG output path")
    args = parser.parse_args()

    root = Path(args.input_dir)
    panel_a = read_aggregates(root / "panel_a_common_physical_scores.csv")
    panel_b = read_aggregates(root / "panel_b_complete_scores.csv")

    width, height = 2000, 1010
    left_a, left_b, top, plot_w, plot_h = 105, 1110, 145, 790, 590
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
        text(105, 52, "CSI inference quality–cost trade-off", 27, 700),
        text(105, 84, "Latency: 200 common samples · Quality: 1,000 samples · Bars: seed SD (quality)", 14, fill="#555555"),
    ]
    svg += draw_panel(
        panel_a, "common_physical_score", "A  Shared physical capability vs. inference cost",
        "Common physical score (higher is better)", left_a, top, plot_w, plot_h,
        (0.75, 4.2e4), (0, 59), [1, 10, 100, 1000, 10000], [0, 10, 20, 30, 40, 50], PANEL_A_OFFSETS,
    )
    svg += draw_panel(
        panel_b, "complete_score", "B  Complete physical–text capability vs. inference cost",
        "Complete CSI-to-text score (higher is better)", left_b, top, plot_w, plot_h,
        (8.5, 4.2e4), (38.5, 81), [10, 100, 1000, 10000], [40, 50, 60, 70, 80], PANEL_B_OFFSETS,
    )

    legend_y = 835
    legend_items = [
        (Style("Full model", "#1F4E79", "star"), "Full model"),
        (Style("Physics baseline", "#6C757D", "square"), "Physics baseline"),
        (Style("Direct decoder", "#E69F00", "circle"), "Direct decoder"),
    ]
    x = 112
    for style, label in legend_items:
        svg.append(marker(x, legend_y, 8, style))
        svg.append(text(x + 17, legend_y + 5, label, 13))
        x += 175
    svg.append(f'<line x1="{x:.1f}" y1="{legend_y:.1f}" x2="{x+38:.1f}" y2="{legend_y:.1f}" stroke="#4D4D4D" stroke-width="1.6" stroke-dasharray="7 5"/>')
    svg.append(text(x + 48, legend_y + 5, "Pareto frontier", 13))

    svg.append(text(1120, legend_y + 5, "Marker area = peak allocated VRAM:", 13, 500))
    x = 1405
    for value, label in ((0.1, "0.1 GB"), (2.0, "2 GB"), (6.0, "6 GB")):
        radius = marker_radius(value)
        svg.append(f'<circle cx="{x:.1f}" cy="{legend_y:.1f}" r="{radius:.1f}" fill="#A0A0A0" opacity="0.55" stroke="#FFFFFF"/>')
        svg.append(text(x + radius + 8, legend_y + 5, label, 13))
        x += 140

    svg.append(text(105, 915, "Deployment settings: Full/physics models use FP32; direct decoders use 4-bit NF4 with BF16 compute.", 13, fill="#555555"))
    svg.append(text(105, 944, "Direct decoders have one training seed. Full-model Panel-B score uses three seeds and the verified structural/text components.", 13, fill="#555555"))
    svg.extend(["</g>", "</svg>"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"saved_benchmark_figure={output}")


if __name__ == "__main__":
    main()
