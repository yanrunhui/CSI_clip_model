from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset, physics_raw_values  # noqa: E402


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(
            "--input entries must use NAME=PATH, "
            f"got {value!r}."
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("Paired configuration name must not be empty.")
    return name, Path(path)


def samples_by_group(path: Path) -> dict[str, object]:
    samples = PreprocessedCSIDataset.from_pt(path).samples
    result = {}
    duplicates = []
    for sample in samples:
        group_id = str(getattr(sample, "group_id", ""))
        if not group_id:
            raise ValueError(f"Sample without group_id in {path}.")
        if group_id in result:
            duplicates.append(group_id)
        result[group_id] = sample
    if duplicates:
        raise ValueError(
            f"Duplicate group_id values in {path}: "
            + ",".join(sorted(set(duplicates))[:10])
        )
    return result


def finite_close(left: float, right: float, atol: float) -> bool:
    left_finite = math.isfinite(left)
    right_finite = math.isfinite(right)
    if left_finite != right_finite:
        return False
    return not left_finite or abs(left - right) <= atol


def labels_match(reference, candidate) -> bool:
    if reference.semantic_key.los_status != candidate.semantic_key.los_status:
        return False
    reference_physics = physics_raw_values(reference)
    candidate_physics = physics_raw_values(candidate)
    finite = torch.isfinite(reference_physics) & torch.isfinite(candidate_physics)
    finite_mismatch = torch.isfinite(reference_physics) ^ torch.isfinite(
        candidate_physics
    )
    if bool(finite_mismatch.any()):
        return False
    if bool(
        (
            (reference_physics[finite] - candidate_physics[finite]).abs()
            > 1e-3
        ).any()
    ):
        return False
    return finite_close(
        float(getattr(reference, "los_delay_s", math.nan)),
        float(getattr(candidate, "los_delay_s", math.nan)),
        atol=1e-12,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Configuration split as NAME=PATH. Repeat at least twice.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-label-mismatch",
        action="store_true",
        help="Keep shared group IDs even when physical labels differ.",
    )
    args = parser.parse_args()

    named_paths = [parse_named_path(value) for value in args.input]
    if len(named_paths) < 2:
        raise ValueError("Provide at least two --input configurations.")
    if len({name for name, _ in named_paths}) != len(named_paths):
        raise ValueError("Configuration names must be unique.")

    grouped = {
        name: samples_by_group(path)
        for name, path in named_paths
    }
    shared_ids = set.intersection(
        *(set(samples) for samples in grouped.values())
    )
    ordered_ids = sorted(shared_ids)
    random.Random(args.seed).shuffle(ordered_ids)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        ordered_ids = ordered_ids[: args.max_samples]

    reference_name = named_paths[0][0]
    mismatch_ids = []
    for group_id in ordered_ids:
        reference = grouped[reference_name][group_id]
        if any(
            not labels_match(reference, grouped[name][group_id])
            for name, _ in named_paths[1:]
        ):
            mismatch_ids.append(group_id)
    if mismatch_ids and not args.allow_label_mismatch:
        raise ValueError(
            f"{len(mismatch_ids)} shared group IDs have different labels. "
            "Inspect preprocessing or use --allow-label-mismatch only for diagnosis. "
            f"Examples: {','.join(mismatch_ids[:10])}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, source_path in named_paths:
        output_path = args.output_dir / f"{name}_paired.pt"
        torch.save(
            [grouped[name][group_id] for group_id in ordered_ids],
            output_path,
        )
        outputs[name] = str(output_path)
        print(
            f"saved_paired_configuration={name} "
            f"samples={len(ordered_ids)} path={output_path}"
        )

    manifest = {
        "seed": args.seed,
        "sample_count": len(ordered_ids),
        "shared_group_count_before_limit": len(shared_ids),
        "label_mismatch_count": len(mismatch_ids),
        "inputs": {name: str(path) for name, path in named_paths},
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "paired_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved_paired_manifest={manifest_path}")


if __name__ == "__main__":
    main()
