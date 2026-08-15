from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset  # noqa: E402


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}.")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path)
    if not name:
        raise ValueError("Configuration name must not be empty.")
    if not path.exists():
        raise FileNotFoundError(path)
    return name, path


def geometry_signature(sample) -> tuple[str, int, int]:
    return (
        str(getattr(sample, "array_type", "")),
        int(getattr(sample, "array_rows", 0)),
        int(getattr(sample, "array_cols", 0)),
    )


def scan(name: str, path: Path) -> dict:
    samples = PreprocessedCSIDataset.from_pt(path).samples
    group_ids = []
    statuses = Counter()
    signatures = Counter()
    config_keys = Counter()
    coordinate_valid_count = 0
    coordinate_count_histogram = Counter()
    for sample in samples:
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError(f"{name} contains a sample without group_id.")
        group_ids.append(group_id)
        statuses[str(sample.semantic_key.los_status)] += 1
        signature = geometry_signature(sample)
        signatures[signature] += 1
        config_keys[str(getattr(sample, "config_key", ""))] += 1
        expected_count = signature[1] * signature[2]
        coordinates = getattr(sample, "antenna_coordinates_wavelengths", None)
        coordinate_count = (
            int(coordinates.shape[0])
            if isinstance(coordinates, torch.Tensor) and coordinates.ndim == 2
            else 0
        )
        coordinate_count_histogram[coordinate_count] += 1
        if (
            isinstance(coordinates, torch.Tensor)
            and coordinates.ndim == 2
            and coordinates.shape == (expected_count, 3)
            and bool(torch.isfinite(coordinates).all())
        ):
            coordinate_valid_count += 1
    if len(group_ids) != len(set(group_ids)):
        raise ValueError(f"{name} contains duplicate group_id values.")
    result = {
        "name": name,
        "path": str(path),
        "sample_count": len(samples),
        "group_ids": set(group_ids),
        "los_count": statuses["los"],
        "nlos_count": statuses["nlos"],
        "geometry_signatures": {
            f"{array_type}:{rows}x{cols}": count
            for (array_type, rows, cols), count in sorted(signatures.items())
        },
        "geometry_signature_set": set(signatures),
        "config_keys": dict(sorted(config_keys.items())),
        "coordinate_valid_count": coordinate_valid_count,
        "coordinate_valid_rate": (
            coordinate_valid_count / len(samples) if samples else math.nan
        ),
        "coordinate_count_histogram": {
            str(count): frequency
            for count, frequency in sorted(coordinate_count_histogram.items())
        },
    }
    print(f"dataset={name}")
    print(f"sample_count={result['sample_count']}")
    print(f"los_count={result['los_count']}")
    print(f"nlos_count={result['nlos_count']}")
    print(
        "geometry_signatures="
        + json.dumps(result["geometry_signatures"], sort_keys=True)
    )
    print(f"coordinate_valid_rate={result['coordinate_valid_rate']:.6g}")
    del samples
    gc.collect()
    return result


def serializable(summary: dict) -> dict:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"group_ids", "geometry_signature_set"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument(
        "--seen-test",
        action="append",
        default=[],
        help="Test configuration name allowed to share a training geometry.",
    )
    parser.add_argument("--require-aligned-tests", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    train_inputs = [parse_named_path(value) for value in args.train]
    test_inputs = [parse_named_path(value) for value in args.test]
    all_names = [name for name, _ in train_inputs + test_inputs]
    if len(all_names) != len(set(all_names)):
        raise ValueError("Train/test configuration names must be unique.")
    seen_test_names = set(args.seen_test)
    unknown_seen = seen_test_names - {name for name, _ in test_inputs}
    if unknown_seen:
        raise ValueError(f"Unknown --seen-test names: {sorted(unknown_seen)}")

    train_summaries = [scan(name, path) for name, path in train_inputs]
    test_summaries = [scan(name, path) for name, path in test_inputs]
    for summary in train_summaries:
        if summary["coordinate_valid_rate"] != 1.0:
            raise ValueError(
                f"Train {summary['name']} has invalid antenna coordinates."
            )
    train_group_ids = set().union(
        *(summary["group_ids"] for summary in train_summaries)
    )
    train_geometries = set().union(
        *(summary["geometry_signature_set"] for summary in train_summaries)
    )
    checks = []
    for summary in test_summaries:
        overlap_count = len(train_group_ids & summary["group_ids"])
        geometry_overlap = bool(train_geometries & summary["geometry_signature_set"])
        is_seen_reference = summary["name"] in seen_test_names
        if overlap_count:
            raise ValueError(
                f"Test {summary['name']} overlaps training by {overlap_count} group IDs."
            )
        if geometry_overlap and not is_seen_reference:
            raise ValueError(f"Unseen test {summary['name']} uses a training geometry.")
        if summary["coordinate_valid_rate"] != 1.0:
            raise ValueError(f"Test {summary['name']} has invalid antenna coordinates.")
        checks.append(
            {
                "test": summary["name"],
                "group_id_overlap_count": overlap_count,
                "geometry_seen_in_training": geometry_overlap,
                "evaluation_type": (
                    "seen_reference" if is_seen_reference else "unseen_array"
                ),
            }
        )
        print(f"{summary['name']}_group_id_overlap_count={overlap_count}")
        print(f"{summary['name']}_geometry_seen_in_training={geometry_overlap}")

    if args.require_aligned_tests:
        reference_ids = test_summaries[0]["group_ids"]
        for summary in test_summaries[1:]:
            if summary["group_ids"] != reference_ids:
                raise ValueError(
                    f"Test group IDs are not aligned: {test_summaries[0]['name']} "
                    f"vs {summary['name']}."
                )
        print(f"aligned_test_group_count={len(reference_ids)}")

    payload = {
        "train": [serializable(summary) for summary in train_summaries],
        "test": [serializable(summary) for summary in test_summaries],
        "checks": checks,
        "require_aligned_tests": args.require_aligned_tests,
        "status": "passed",
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved_validation_json={args.output_json}")
    print("cross_array_split_validation=passed")


if __name__ == "__main__":
    main()
