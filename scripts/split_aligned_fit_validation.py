from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--input must use NAME=PATH, got {value!r}.")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path)
    if not name:
        raise ValueError("Input configuration name must not be empty.")
    if not path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {path}")
    return name, path


def load_samples(path: Path) -> list[object]:
    samples = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {path}.")
    return samples


def sample_group_id(sample: object, path: Path) -> str:
    group_id = str(getattr(sample, "group_id", "")).strip()
    if not group_id:
        raise ValueError(f"Sample without group_id in {path}.")
    return group_id


def sample_los_status(sample: object, path: Path) -> str:
    semantic_key = getattr(sample, "semantic_key", None)
    status = str(getattr(semantic_key, "los_status", "")).lower()
    if status not in {"los", "nlos"}:
        raise ValueError(f"Unsupported los_status={status!r} in {path}.")
    return status


def scan(path: Path) -> dict[str, str]:
    samples = load_samples(path)
    statuses = {}
    for sample in samples:
        group_id = sample_group_id(sample, path)
        if group_id in statuses:
            raise ValueError(f"Duplicate group_id={group_id!r} in {path}.")
        statuses[group_id] = sample_los_status(sample, path)
    del samples
    gc.collect()
    return statuses


def shuffled(values: list[str], seed: int) -> list[str]:
    result = sorted(values)
    random.Random(seed).shuffle(result)
    return result


def digest(group_ids: list[str]) -> str:
    hasher = hashlib.sha256()
    for group_id in group_ids:
        hasher.update(group_id.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Aligned dataset as NAME=PATH. Repeat for every configuration.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-per-status", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=34_567)
    args = parser.parse_args()

    if args.validation_per_status <= 0:
        raise ValueError("--validation-per-status must be positive.")
    named_paths = [parse_named_path(value) for value in args.input]
    if len({name for name, _ in named_paths}) != len(named_paths):
        raise ValueError("Input configuration names must be unique.")

    reference_name, reference_path = named_paths[0]
    reference_statuses = scan(reference_path)
    reference_ids = set(reference_statuses)
    print(
        f"reference={reference_name} samples={len(reference_ids)} "
        f"status_counts={dict(Counter(reference_statuses.values()))}"
    )
    for name, path in named_paths[1:]:
        statuses = scan(path)
        if set(statuses) != reference_ids:
            missing = len(reference_ids - set(statuses))
            extra = len(set(statuses) - reference_ids)
            raise ValueError(
                f"{name} group IDs differ from reference: missing={missing} extra={extra}."
            )
        mismatched = [
            group_id
            for group_id in reference_ids
            if statuses[group_id] != reference_statuses[group_id]
        ]
        if mismatched:
            raise ValueError(f"{name} has {len(mismatched)} LoS/NLoS mismatches.")

    by_status = {
        status: [
            group_id
            for group_id, sample_status in reference_statuses.items()
            if sample_status == status
        ]
        for status in ("los", "nlos")
    }
    for status, group_ids in by_status.items():
        if len(group_ids) <= args.validation_per_status:
            raise ValueError(
                f"Not enough {status} samples for validation_per_status="
                f"{args.validation_per_status}."
            )

    validation_ids = shuffled(
        shuffled(by_status["los"], args.seed)[: args.validation_per_status]
        + shuffled(by_status["nlos"], args.seed + 1)[: args.validation_per_status],
        args.seed + 2,
    )
    validation_set = set(validation_ids)
    fit_ids = shuffled(
        [group_id for group_id in reference_ids if group_id not in validation_set],
        args.seed + 3,
    )
    if set(fit_ids) & validation_set:
        raise AssertionError("Fit and validation IDs overlap.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fit_group_ids.txt").write_text(
        "\n".join(fit_ids) + "\n", encoding="utf-8"
    )
    (args.output_dir / "validation_group_ids.txt").write_text(
        "\n".join(validation_ids) + "\n", encoding="utf-8"
    )

    outputs = {}
    selected_order = {group_id: index for index, group_id in enumerate(fit_ids)}
    validation_order = {
        group_id: index for index, group_id in enumerate(validation_ids)
    }
    for name, path in named_paths:
        samples = load_samples(path)
        fit_samples: list[object | None] = [None] * len(fit_ids)
        validation_samples: list[object | None] = [None] * len(validation_ids)
        for sample in samples:
            group_id = sample_group_id(sample, path)
            if group_id in selected_order:
                fit_samples[selected_order[group_id]] = sample
            elif group_id in validation_order:
                validation_samples[validation_order[group_id]] = sample
        if any(sample is None for sample in fit_samples + validation_samples):
            raise ValueError(f"{name} is missing selected fit/validation samples.")
        fit_path = args.output_dir / f"{name}_fit.pt"
        validation_path = args.output_dir / f"{name}_val.pt"
        torch.save(fit_samples, fit_path)
        torch.save(validation_samples, validation_path)
        outputs[name] = {"fit": str(fit_path), "validation": str(validation_path)}
        print(
            f"saved_configuration={name} fit={len(fit_samples)} "
            f"validation={len(validation_samples)}"
        )
        del samples, fit_samples, validation_samples
        gc.collect()

    manifest = {
        "seed": args.seed,
        "fit_count": len(fit_ids),
        "validation_count": len(validation_ids),
        "fit_los_count": sum(reference_statuses[group_id] == "los" for group_id in fit_ids),
        "fit_nlos_count": sum(reference_statuses[group_id] == "nlos" for group_id in fit_ids),
        "validation_los_count": sum(
            reference_statuses[group_id] == "los" for group_id in validation_ids
        ),
        "validation_nlos_count": sum(
            reference_statuses[group_id] == "nlos" for group_id in validation_ids
        ),
        "fit_group_id_sha256": digest(fit_ids),
        "validation_group_id_sha256": digest(validation_ids),
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "fit_validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved_manifest={manifest_path}")


if __name__ == "__main__":
    main()
