from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset  # noqa: E402


NF_NAME_PATTERN = re.compile(r"^nf(\d+)$")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected nfN=PATH, got {value!r}.")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if NF_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"Numerology name must use nfN, got {name!r}.")
    if not path.exists():
        raise FileNotFoundError(path)
    return name, path


def expected_nf(name: str) -> int:
    match = NF_NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(name)
    return int(match.group(1))


def finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def source_nf(sample) -> int:
    stored = int(getattr(sample, "source_n_freq", 0) or 0)
    if stored > 0:
        return stored
    bandwidth = finite_float(getattr(sample, "bandwidth_hz", math.nan))
    spacing = finite_float(getattr(sample, "subcarrier_spacing_hz", math.nan))
    if bandwidth is not None and spacing is not None and spacing > 0.0:
        return max(int(round(bandwidth / spacing)), 1)
    return int(sample.tokens.shape[-1])


def compact_histogram(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def scan(name: str, path: Path) -> dict:
    samples = PreprocessedCSIDataset.from_pt(path).samples
    if not samples:
        raise ValueError(f"No samples loaded from {path}.")
    group_ids = []
    statuses = Counter()
    source_nfs = Counter()
    token_nfs = Counter()
    spacings = Counter()
    bandwidths = Counter()
    geometries = Counter()
    reflection_path_counts = Counter()
    for sample in samples:
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError(f"{name} contains a sample without group_id.")
        group_ids.append(group_id)
        statuses[str(sample.semantic_key.los_status)] += 1
        source_nfs[source_nf(sample)] += 1
        token_nfs[int(sample.tokens.shape[-1])] += 1
        spacing = finite_float(getattr(sample, "subcarrier_spacing_hz", math.nan))
        bandwidth = finite_float(getattr(sample, "bandwidth_hz", math.nan))
        spacings["nan" if spacing is None else f"{spacing:.3f}"] += 1
        bandwidths["nan" if bandwidth is None else f"{bandwidth:.3f}"] += 1
        geometries[
            (
                str(getattr(sample, "array_type", "")),
                int(getattr(sample, "array_rows", 0)),
                int(getattr(sample, "array_cols", 0)),
            )
        ] += 1
        reflection_path_counts[int(getattr(sample, "reflection_path_count", 0))] += 1
    if len(group_ids) != len(set(group_ids)):
        raise ValueError(f"{name} contains duplicate group_id values.")

    result = {
        "name": name,
        "path": str(path),
        "sample_count": len(samples),
        "group_ids": set(group_ids),
        "los_count": statuses["los"],
        "nlos_count": statuses["nlos"],
        "source_nf_histogram": compact_histogram(source_nfs),
        "token_nf_histogram": compact_histogram(token_nfs),
        "subcarrier_spacing_hz_histogram": compact_histogram(spacings),
        "bandwidth_hz_histogram": compact_histogram(bandwidths),
        "geometry_histogram": {
            f"{array_type}:{rows}x{cols}": int(count)
            for (array_type, rows, cols), count in sorted(geometries.items())
        },
        "reflection_path_count_histogram": compact_histogram(reflection_path_counts),
    }
    print(f"dataset={name}")
    print(f"sample_count={result['sample_count']}")
    print(f"los_count={result['los_count']}")
    print(f"nlos_count={result['nlos_count']}")
    for key in (
        "source_nf_histogram",
        "token_nf_histogram",
        "subcarrier_spacing_hz_histogram",
        "bandwidth_hz_histogram",
        "geometry_histogram",
        "reflection_path_count_histogram",
    ):
        print(f"{key}={json.dumps(result[key], sort_keys=True)}")
    del samples
    gc.collect()
    return result


def serializable(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "group_ids"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--held-out-nf", type=int, required=True)
    parser.add_argument("--require-aligned-training", action="store_true")
    parser.add_argument("--require-aligned-tests", action="store_true")
    parser.add_argument("--require-reflection-path-labels", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    train_inputs = [parse_named_path(value) for value in args.train]
    test_inputs = [parse_named_path(value) for value in args.test]
    train_names = [name for name, _ in train_inputs]
    test_names = [name for name, _ in test_inputs]
    if len(train_names) != len(set(train_names)):
        raise ValueError("Training numerology names must be unique.")
    if len(test_names) != len(set(test_names)):
        raise ValueError("Test numerology names must be unique.")
    if args.held_out_nf in {expected_nf(name) for name in train_names}:
        raise ValueError(f"Held-out nf{args.held_out_nf} appears in training inputs.")
    if args.held_out_nf not in {expected_nf(name) for name in test_names}:
        raise ValueError(f"Test inputs do not contain held-out nf{args.held_out_nf}.")

    train_summaries = [scan(name, path) for name, path in train_inputs]
    test_summaries = [scan(name, path) for name, path in test_inputs]
    for summary in (*train_summaries, *test_summaries):
        expected = expected_nf(summary["name"])
        actual = {int(value) for value in summary["source_nf_histogram"]}
        if actual != {expected}:
            raise ValueError(
                f"{summary['name']} expected source nf {expected}, got {sorted(actual)}."
            )
        if summary["los_count"] == 0 or summary["nlos_count"] == 0:
            raise ValueError(f"{summary['name']} is missing LoS or NLoS samples.")
        if args.require_reflection_path_labels:
            reflection_values = {
                int(value) for value in summary["reflection_path_count_histogram"]
            }
            if not reflection_values or reflection_values == {0}:
                raise ValueError(
                    f"{summary['name']} has no informative reflection-path labels."
                )

    aligned_training_count = None
    if args.require_aligned_training:
        reference_ids = train_summaries[0]["group_ids"]
        for summary in train_summaries[1:]:
            if summary["group_ids"] != reference_ids:
                raise ValueError(
                    "Training group IDs are not aligned: "
                    f"{train_summaries[0]['name']} vs {summary['name']}."
                )
        aligned_training_count = len(reference_ids)
        print(f"aligned_training_group_count={aligned_training_count}")

    aligned_test_count = None
    if args.require_aligned_tests:
        reference_ids = test_summaries[0]["group_ids"]
        for summary in test_summaries[1:]:
            if summary["group_ids"] != reference_ids:
                raise ValueError(
                    "Test group IDs are not aligned: "
                    f"{test_summaries[0]['name']} vs {summary['name']}."
                )
        aligned_test_count = len(reference_ids)
        print(f"aligned_test_group_count={aligned_test_count}")

    train_group_ids = set().union(
        *(summary["group_ids"] for summary in train_summaries)
    )
    overlap_counts = {}
    for summary in test_summaries:
        overlap_count = len(train_group_ids & summary["group_ids"])
        overlap_counts[summary["name"]] = overlap_count
        print(f"{summary['name']}_group_id_overlap_count={overlap_count}")
        if overlap_count:
            raise ValueError(
                f"Test {summary['name']} overlaps training by {overlap_count} group IDs."
            )

    payload = {
        "training_numerologies": [expected_nf(name) for name in train_names],
        "held_out_numerology": args.held_out_nf,
        "train": [serializable(summary) for summary in train_summaries],
        "test": [serializable(summary) for summary in test_summaries],
        "aligned_training_group_count": aligned_training_count,
        "aligned_test_group_count": aligned_test_count,
        "test_group_id_overlap_counts": overlap_counts,
        "status": "passed",
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved_validation_json={args.output_json}")
    print("cross_numerology_split_validation=passed")


if __name__ == "__main__":
    main()
