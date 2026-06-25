from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset


def _first_path_delay_ns(sample) -> float:
    value = getattr(sample, "first_path_delay_s", math.nan)
    try:
        value = float(value) * 1e9
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _format_counts(samples) -> str:
    counts = Counter(sample.semantic_key.los_status for sample in samples)
    return ",".join(f"{label}:{counts[label]}" for label in sorted(counts))


def filter_dataset(input_path: Path, output_path: Path, max_first_delay_ns: float) -> None:
    dataset = PreprocessedCSIDataset.from_pt(str(input_path))
    samples = dataset.samples
    filtered = [
        sample
        for sample in samples
        if math.isfinite(_first_path_delay_ns(sample))
        and _first_path_delay_ns(sample) < max_first_delay_ns
    ]
    if not filtered:
        raise ValueError(
            f"No samples remain after filtering first_path_delay_ns < {max_first_delay_ns}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, output_path)
    dropped = len(samples) - len(filtered)
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"max_first_path_delay_ns={max_first_delay_ns:g}")
    print(f"samples={len(samples)} -> {len(filtered)} dropped={dropped}")
    print(f"los_status_before={_format_counts(samples)}")
    print(f"los_status_after={_format_counts(filtered)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-first-path-delay-ns", type=float, default=1280.0)
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more preprocessed .pt files to filter.",
    )
    parser.add_argument(
        "--suffix",
        default="_lt1280",
        help="Suffix inserted before .pt for output files.",
    )
    args = parser.parse_args()
    if args.max_first_path_delay_ns <= 0.0:
        raise ValueError("--max-first-path-delay-ns must be positive.")

    for raw_path in args.paths:
        input_path = Path(raw_path)
        output_path = input_path.with_name(
            f"{input_path.stem}{args.suffix}{input_path.suffix}"
        )
        filter_dataset(input_path, output_path, args.max_first_path_delay_ns)


if __name__ == "__main__":
    main()
