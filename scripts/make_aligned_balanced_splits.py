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


def load_samples(path: Path):
    samples = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {path}.")
    return samples


def sample_group_id(sample, path: Path) -> str:
    group_id = str(getattr(sample, "group_id", "")).strip()
    if not group_id:
        raise ValueError(f"Found a sample without group_id in {path}.")
    return group_id


def sample_los_status(sample, path: Path) -> str:
    semantic_key = getattr(sample, "semantic_key", None)
    status = str(getattr(semantic_key, "los_status", "")).lower()
    if status not in {"los", "nlos"}:
        raise ValueError(
            f"Unsupported los_status={status!r} for group "
            f"{sample_group_id(sample, path)!r} in {path}."
        )
    return status


def scan_group_status(path: Path) -> dict[str, str]:
    samples = load_samples(path)
    statuses: dict[str, str] = {}
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


def id_digest(group_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for group_id in group_ids:
        digest.update(group_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_ids(path: Path, group_ids: list[str]) -> None:
    path.write_text("\n".join(group_ids) + "\n", encoding="utf-8")


def save_configuration_splits(
    *,
    name: str,
    input_path: Path,
    output_dir: Path,
    all_ids: list[str],
    train_ids: list[str],
    test_ids: list[str],
    reference_statuses: dict[str, str],
) -> dict[str, str]:
    samples = load_samples(input_path)
    selected_ids = set(all_ids)
    selected = {}
    for sample in samples:
        group_id = sample_group_id(sample, input_path)
        if group_id not in selected_ids:
            continue
        status = sample_los_status(sample, input_path)
        if status != reference_statuses[group_id]:
            raise ValueError(
                f"LoS/NLoS mismatch for group_id={group_id!r}: "
                f"reference={reference_statuses[group_id]} {name}={status}."
            )
        selected[group_id] = sample

    missing = selected_ids - set(selected)
    if missing:
        raise ValueError(
            f"{name} is missing {len(missing)} selected group IDs. "
            f"Examples: {','.join(sorted(missing)[:10])}"
        )

    outputs = {
        "all": output_dir / f"{name}.pt",
        "train": output_dir / f"{name}_train.pt",
        "test": output_dir / f"{name}_test.pt",
    }
    torch.save([selected[group_id] for group_id in all_ids], outputs["all"])
    torch.save([selected[group_id] for group_id in train_ids], outputs["train"])
    torch.save([selected[group_id] for group_id in test_ids], outputs["test"])
    print(
        f"saved_configuration={name} all={len(all_ids)} "
        f"train={len(train_ids)} test={len(test_ids)}"
    )

    del selected
    del samples
    gc.collect()
    return {key: str(path) for key, path in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Preprocessed candidate dataset as NAME=PATH. Repeat for every configuration.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-status", type=int, default=50_000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=23_421)
    args = parser.parse_args()

    if args.samples_per_status <= 0:
        raise ValueError("--samples-per-status must be positive.")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")

    named_paths = [parse_named_path(value) for value in args.input]
    names = [name for name, _ in named_paths]
    if len(set(names)) != len(names):
        raise ValueError("Input configuration names must be unique.")

    status_maps = {}
    shared_ids: set[str] | None = None
    for name, path in named_paths:
        statuses = scan_group_status(path)
        status_maps[name] = statuses
        shared_ids = (
            set(statuses)
            if shared_ids is None
            else shared_ids.intersection(statuses)
        )
        counts = Counter(statuses.values())
        print(
            f"scanned_configuration={name} samples={len(statuses)} "
            f"los={counts['los']} nlos={counts['nlos']}"
        )

    assert shared_ids is not None
    reference_name = names[0]
    reference_statuses = status_maps[reference_name]
    mismatches = []
    for group_id in shared_ids:
        reference_status = reference_statuses[group_id]
        if any(
            status_maps[name][group_id] != reference_status
            for name in names[1:]
        ):
            mismatches.append(group_id)
    if mismatches:
        raise ValueError(
            f"{len(mismatches)} shared group IDs disagree on LoS/NLoS status. "
            f"Examples: {','.join(sorted(mismatches)[:10])}"
        )

    candidates = {
        status: [
            group_id
            for group_id in shared_ids
            if reference_statuses[group_id] == status
        ]
        for status in ("los", "nlos")
    }
    for status, group_ids in candidates.items():
        if len(group_ids) < args.samples_per_status:
            raise ValueError(
                f"Only {len(group_ids)} common {status} samples are available; "
                f"{args.samples_per_status} requested."
            )

    chosen_los = shuffled(candidates["los"], args.seed)[: args.samples_per_status]
    chosen_nlos = shuffled(candidates["nlos"], args.seed + 1)[: args.samples_per_status]
    test_per_status = int(round(args.samples_per_status * args.test_fraction))
    test_per_status = min(max(test_per_status, 1), args.samples_per_status - 1)

    test_ids = shuffled(
        chosen_los[:test_per_status] + chosen_nlos[:test_per_status],
        args.seed + 2,
    )
    train_ids = shuffled(
        chosen_los[test_per_status:] + chosen_nlos[test_per_status:],
        args.seed + 3,
    )
    all_ids = shuffled(chosen_los + chosen_nlos, args.seed + 4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_ids(args.output_dir / "selected_group_ids.txt", all_ids)
    write_ids(args.output_dir / "train_group_ids.txt", train_ids)
    write_ids(args.output_dir / "test_group_ids.txt", test_ids)

    outputs = {}
    for name, path in named_paths:
        outputs[name] = save_configuration_splits(
            name=name,
            input_path=path,
            output_dir=args.output_dir,
            all_ids=all_ids,
            train_ids=train_ids,
            test_ids=test_ids,
            reference_statuses=reference_statuses,
        )

    manifest = {
        "seed": args.seed,
        "samples_per_status": args.samples_per_status,
        "test_fraction": args.test_fraction,
        "common_group_count": len(shared_ids),
        "common_los_count": len(candidates["los"]),
        "common_nlos_count": len(candidates["nlos"]),
        "selected_count": len(all_ids),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "selected_los_count": len(chosen_los),
        "selected_nlos_count": len(chosen_nlos),
        "train_los_count": args.samples_per_status - test_per_status,
        "train_nlos_count": args.samples_per_status - test_per_status,
        "test_los_count": test_per_status,
        "test_nlos_count": test_per_status,
        "selected_group_ids_sha256": id_digest(all_ids),
        "train_group_ids_sha256": id_digest(train_ids),
        "test_group_ids_sha256": id_digest(test_ids),
        "inputs": {name: str(path) for name, path in named_paths},
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "aligned_balanced_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved_manifest={manifest_path}")


if __name__ == "__main__":
    main()
