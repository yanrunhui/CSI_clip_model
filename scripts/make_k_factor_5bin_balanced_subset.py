from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset

K_FACTOR_5BIN_RANGES = (
    ("weak", float("-inf"), 3.0),
    ("strong_low", 3.0, 15.0),
    ("strong_mid", 15.0, 30.0),
    ("strong_high", 30.0, 45.0),
    ("strong_very_high", 45.0, 70.0),
)


def finite_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def k_factor_bin(value: float) -> str | None:
    for bin_idx, (label, lower, upper) in enumerate(K_FACTOR_5BIN_RANGES):
        lower_hit = value >= lower
        upper_hit = value <= upper if bin_idx == len(K_FACTOR_5BIN_RANGES) - 1 else value < upper
        if lower_hit and upper_hit:
            return label
    return None


def build_balanced_subset(
    input_path: str,
    output_path: str,
    max_per_bin: int,
    seed: int,
) -> None:
    if max_per_bin <= 0:
        raise ValueError("--max-per-bin must be a positive integer.")

    rng = random.Random(seed)
    dataset = PreprocessedCSIDataset.from_pt(input_path)
    samples = dataset.samples

    by_bin = defaultdict(list)
    invalid_count = 0
    out_of_range_count = 0
    for sample in samples:
        value = finite_float(getattr(sample, "k_factor_db", None))
        if value is None:
            invalid_count += 1
            continue
        label = k_factor_bin(value)
        if label is None:
            out_of_range_count += 1
            continue
        by_bin[label].append(sample)

    balanced = []
    selected_counts = {}
    for label, _, _ in K_FACTOR_5BIN_RANGES:
        candidates = by_bin[label][:]
        rng.shuffle(candidates)
        selected = candidates[:max_per_bin]
        balanced.extend(selected)
        selected_counts[label] = len(selected)
    rng.shuffle(balanced)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(balanced, output)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"seed={seed}")
    print(f"max_per_bin={max_per_bin}")
    print(f"input_samples={len(samples)}")
    print(f"invalid_k_factor_samples={invalid_count}")
    print(f"out_of_range_k_factor_samples={out_of_range_count}")
    print(f"output_samples={len(balanced)}")
    print("bin_ranges=" + ";".join(_format_range(label, lower, upper) for label, lower, upper in K_FACTOR_5BIN_RANGES))
    for label, _, _ in K_FACTOR_5BIN_RANGES:
        print(
            f"bin_{label}=available:{len(by_bin[label])} "
            f"selected:{selected_counts[label]}"
        )
    print("selected_histogram=" + ",".join(f"{label}:{selected_counts[label]}" for label, _, _ in K_FACTOR_5BIN_RANGES))
    print("semantic_k_factor_bin_histogram=" + _semantic_k_factor_histogram(balanced))


def _format_range(label: str, lower: float, upper: float) -> str:
    lower_text = "-inf" if math.isinf(lower) and lower < 0 else _format_number(lower)
    upper_text = "inf" if math.isinf(upper) else _format_number(upper)
    return f"{label}=[{lower_text},{upper_text}]"


def _format_number(value: float) -> str:
    return f"{value:g}"


def _semantic_k_factor_histogram(samples) -> str:
    counts = Counter(getattr(sample.semantic_key, "k_factor_bin", "unknown") for sample in samples)
    return ",".join(f"{label}:{count}" for label, count in sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_400k.pt",
        help="Source preprocessed .pt sample list.",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_k_factor_5bin_balanced_2000.pt",
        help="Output balanced .pt sample list.",
    )
    parser.add_argument(
        "--max-per-bin",
        type=int,
        default=2000,
        help="Maximum samples to keep for each K-factor bin.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_balanced_subset(
        input_path=args.input,
        output_path=args.output,
        max_per_bin=args.max_per_bin,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
