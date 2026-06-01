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


def split_balanced_subset(
    input_path: str,
    train_output_path: str,
    eval_output_path: str,
    eval_fraction: float,
    seed: int,
) -> None:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("--eval-fraction must be between 0 and 1.")

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

    train_samples = []
    eval_samples = []
    split_counts = {}
    for label, _, _ in K_FACTOR_5BIN_RANGES:
        candidates = by_bin[label][:]
        rng.shuffle(candidates)
        if len(candidates) <= 1:
            eval_count = 0
        else:
            eval_count = int(round(len(candidates) * eval_fraction))
            eval_count = min(max(eval_count, 1), len(candidates) - 1)
        eval_bin_samples = candidates[:eval_count]
        train_bin_samples = candidates[eval_count:]
        eval_samples.extend(eval_bin_samples)
        train_samples.extend(train_bin_samples)
        split_counts[label] = (len(train_bin_samples), len(eval_bin_samples))

    rng.shuffle(train_samples)
    rng.shuffle(eval_samples)

    train_output = Path(train_output_path)
    eval_output = Path(eval_output_path)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_samples, train_output)
    torch.save(eval_samples, eval_output)

    print(f"input={input_path}")
    print(f"train_output={train_output_path}")
    print(f"eval_output={eval_output_path}")
    print(f"seed={seed}")
    print(f"eval_fraction={eval_fraction}")
    print(f"input_samples={len(samples)}")
    print(f"invalid_k_factor_samples={invalid_count}")
    print(f"out_of_range_k_factor_samples={out_of_range_count}")
    print(f"train_samples={len(train_samples)}")
    print(f"eval_samples={len(eval_samples)}")
    for label, _, _ in K_FACTOR_5BIN_RANGES:
        train_count, eval_count = split_counts[label]
        print(
            f"bin_{label}=available:{len(by_bin[label])} "
            f"train:{train_count} eval:{eval_count}"
        )
    print("train_histogram=" + _k_histogram(train_samples))
    print("eval_histogram=" + _k_histogram(eval_samples))
    print("train_semantic_k_factor_bin_histogram=" + _semantic_k_factor_histogram(train_samples))
    print("eval_semantic_k_factor_bin_histogram=" + _semantic_k_factor_histogram(eval_samples))


def _k_histogram(samples) -> str:
    counts = Counter()
    for sample in samples:
        value = finite_float(getattr(sample, "k_factor_db", None))
        if value is None:
            continue
        label = k_factor_bin(value)
        if label is not None:
            counts[label] += 1
    return ",".join(f"{label}:{counts[label]}" for label, _, _ in K_FACTOR_5BIN_RANGES)


def _semantic_k_factor_histogram(samples) -> str:
    counts = Counter(getattr(sample.semantic_key, "k_factor_bin", "unknown") for sample in samples)
    return ",".join(f"{label}:{count}" for label, count in sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_k_factor_5bin_balanced_2000.pt",
        help="Input balanced .pt sample list.",
    )
    parser.add_argument(
        "--train-output",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_k_factor_5bin_balanced_2000_train.pt",
    )
    parser.add_argument(
        "--eval-output",
        default="/root/autodl-tmp/CSI_model/artifacts/d2los_k_factor_5bin_balanced_2000_eval.pt",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_balanced_subset(
        input_path=args.input,
        train_output_path=args.train_output,
        eval_output_path=args.eval_output,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
