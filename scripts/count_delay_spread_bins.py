from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset, apply_semantic_key_mode, semantic_key_mode_choices


DEFAULT_BIN_EDGES = (0.0, 25.0, 50.0, 100.0, 200.0, 400.0)
DEFAULT_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _finite(value: object) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _delay_spread_ns(sample) -> float:
    if hasattr(sample, "delay_spread_ns"):
        return _finite(getattr(sample, "delay_spread_ns"))
    if hasattr(sample, "delay_spread_s"):
        return _finite(getattr(sample, "delay_spread_s")) * 1e9
    if hasattr(sample, "delay_spread"):
        return _finite(getattr(sample, "delay_spread")) * 1e9
    return math.nan


def _parse_edges(text: str) -> tuple[float, ...]:
    edges = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(edges) < 2:
        raise ValueError("--bin-edges must contain at least two comma-separated values.")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("--bin-edges must be strictly increasing.")
    return edges


def _build_edges(values: torch.Tensor, args: argparse.Namespace) -> tuple[float, ...]:
    if args.bin_width is None:
        return _parse_edges(args.bin_edges)
    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive.")
    max_ns = float(args.max_ns) if args.max_ns is not None else float(values.max())
    if max_ns <= 0:
        max_ns = args.bin_width
    steps = int(math.ceil(max_ns / args.bin_width))
    return tuple(float(idx * args.bin_width) for idx in range(steps + 1))


def _format_edge(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "neg").replace(".", "p")


def _format_value(value: float) -> str:
    return f"{value:.4f}"


def _group_value(sample, group_by: str) -> str:
    if group_by == "none":
        return "all"
    key = getattr(sample, "semantic_key", None)
    if key is None:
        return "unknown"
    return str(getattr(key, group_by, "unknown"))


def _bin_rows(values: torch.Tensor, edges: tuple[float, ...]) -> list[tuple[str, torch.Tensor]]:
    rows: list[tuple[str, torch.Tensor]] = []
    for idx, (lower, upper) in enumerate(zip(edges, edges[1:])):
        mask = (values >= lower) & (values < upper)
        rows.append((f"{_format_edge(lower)}_{_format_edge(upper)}", mask))
    rows.append((f"{_format_edge(edges[-1])}_plus", values >= edges[-1]))
    return rows


def _print_summary(prefix: str, values: torch.Tensor, *, total_samples: int) -> None:
    print(f"{prefix}_total_samples={total_samples}")
    print(f"{prefix}_valid_delay_spread_count={values.numel()}")
    print(f"{prefix}_invalid_delay_spread_count={total_samples - values.numel()}")
    if values.numel() == 0:
        return
    quantiles = torch.quantile(
        values,
        torch.tensor(DEFAULT_QUANTILES, dtype=values.dtype),
    )
    print(f"{prefix}_min={_format_value(float(values.min()))}")
    print(f"{prefix}_max={_format_value(float(values.max()))}")
    print(f"{prefix}_mean={_format_value(float(values.mean()))}")
    print(f"{prefix}_std={_format_value(float(values.std(correction=0)))}")
    print(
        f"{prefix}_quantiles="
        + ",".join(
            f"p{int(q * 100):02d}:{_format_value(float(value))}"
            for q, value in zip(DEFAULT_QUANTILES, quantiles.tolist(), strict=True)
        )
    )


def _print_bins(prefix: str, values: torch.Tensor, edges: tuple[float, ...]) -> None:
    print(
        f"{prefix}_bin_order="
        + ",".join(
            f"{_format_edge(lower)}_{_format_edge(upper)}:{_format_value(lower)}-{_format_value(upper)}"
            for lower, upper in zip(edges, edges[1:])
        )
        + f",{_format_edge(edges[-1])}_plus:{_format_value(edges[-1])}-inf"
    )
    total = max(values.numel(), 1)
    cumulative = 0
    for label, mask in _bin_rows(values, edges):
        count = int(mask.sum().item())
        cumulative += count
        if count > 0:
            bin_values = values[mask]
            range_text = f"{_format_value(float(bin_values.min()))},{_format_value(float(bin_values.max()))}"
            mean_text = _format_value(float(bin_values.mean()))
        else:
            range_text = "nan,nan"
            mean_text = "nan"
        print(
            f"{prefix}_bin_{label}=count:{count} "
            f"fraction:{count / total:.6f} "
            f"cumulative:{cumulative} "
            f"cumulative_fraction:{cumulative / total:.6f} "
            f"range:{range_text} "
            f"mean:{mean_text}"
        )


def _print_semantic_histograms(samples) -> None:
    histograms: dict[str, Counter[str]] = {
        "ds_bin": Counter(),
        "k_factor_bin": Counter(),
        "los_status": Counter(),
        "path_richness": Counter(),
    }
    for sample in samples:
        key = getattr(sample, "semantic_key", None)
        if key is None:
            continue
        for field, counter in histograms.items():
            counter[str(getattr(key, field, "unknown"))] += 1
    for field, counter in histograms.items():
        print(
            f"semantic_{field}_histogram="
            + ",".join(f"{label}:{counter[label]}" for label in sorted(counter))
        )


def run(
    data_path: str,
    *,
    semantic_key_mode: str,
    bin_edges: tuple[float, ...] | None,
    args: argparse.Namespace,
) -> None:
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    samples = apply_semantic_key_mode(dataset.samples, semantic_key_mode)
    values_by_group: dict[str, list[float]] = defaultdict(list)
    valid_samples = []
    for sample in samples:
        value = _delay_spread_ns(sample)
        if not math.isfinite(value):
            continue
        values_by_group[_group_value(sample, args.group_by)].append(value)
        valid_samples.append(sample)

    all_values = torch.tensor(
        [value for values in values_by_group.values() for value in values],
        dtype=torch.float32,
    )
    edges = bin_edges if bin_edges is not None else _build_edges(all_values, args)

    print(f"data_path={data_path}")
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"group_by={args.group_by}")
    _print_summary("delay_spread_ns", all_values, total_samples=len(samples))
    _print_bins("delay_spread_ns", all_values, edges)
    _print_semantic_histograms(samples)

    if args.group_by == "none":
        return

    for group, raw_values in sorted(values_by_group.items()):
        group_values = torch.tensor(raw_values, dtype=torch.float32)
        _print_summary(
            f"group_{args.group_by}_{group}_delay_spread_ns",
            group_values,
            total_samples=len(raw_values),
        )
        _print_bins(f"group_{args.group_by}_{group}_delay_spread_ns", group_values, edges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count samples by delay_spread_ns bins in a preprocessed CSI .pt dataset."
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--semantic-key-mode",
        default="full",
        choices=semantic_key_mode_choices(),
        help="Apply the same semantic-key remapping used by training/evaluation.",
    )
    parser.add_argument(
        "--bin-edges",
        default=",".join(str(edge) for edge in DEFAULT_BIN_EDGES),
        help="Comma-separated ns bin edges. The script also prints a final >=last_edge bin.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        help="Use fixed-width ns bins instead of --bin-edges.",
    )
    parser.add_argument(
        "--max-ns",
        type=float,
        help="Upper edge for --bin-width. Defaults to the max delay_spread_ns.",
    )
    parser.add_argument(
        "--group-by",
        default="none",
        choices=("none", "ds_bin", "k_factor_bin", "los_status", "path_richness"),
        help="Also print per-group delay-spread bin counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bin_edges = None if args.bin_width is not None else _parse_edges(args.bin_edges)
    run(
        args.data_path,
        semantic_key_mode=args.semantic_key_mode,
        bin_edges=bin_edges,
        args=args,
    )


if __name__ == "__main__":
    main()
