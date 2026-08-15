from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
        raise FileNotFoundError(path)
    return name, path


def load_samples(path: Path) -> list[object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = payload.samples if hasattr(payload, "samples") else payload
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {path}.")
    return samples


def group_id(sample: object, path: Path) -> str:
    value = str(getattr(sample, "group_id", "")).strip()
    if not value:
        raise ValueError(f"Sample without group_id in {path}.")
    return value


def los_status(sample: object, path: Path) -> str:
    semantic_key = getattr(sample, "semantic_key", None)
    value = str(getattr(semantic_key, "los_status", "")).lower()
    if value not in {"los", "nlos"}:
        raise ValueError(f"Unsupported los_status={value!r} in {path}.")
    return value


def delay_spread_ns(sample: object) -> float:
    if hasattr(sample, "delay_spread_ns"):
        value = getattr(sample, "delay_spread_ns")
        scale = 1.0
    elif hasattr(sample, "delay_spread_s"):
        value = getattr(sample, "delay_spread_s")
        scale = 1e9
    elif hasattr(sample, "delay_spread"):
        value = getattr(sample, "delay_spread")
        scale = 1e9
    else:
        return math.nan
    try:
        value = float(value) * scale
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def delay_ns(sample: object, field: str) -> float:
    try:
        value = float(getattr(sample, field)) * 1e9
    except (AttributeError, TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def scan(
    path: Path,
    *,
    include_delays: bool,
) -> tuple[dict[str, str], dict[str, float], dict[str, float], dict[str, float]]:
    samples = load_samples(path)
    statuses = {}
    delay_spreads = {}
    first_path_delays = {}
    los_delays = {}
    for sample in samples:
        sample_id = group_id(sample, path)
        if sample_id in statuses:
            raise ValueError(f"Duplicate group_id={sample_id!r} in {path}.")
        statuses[sample_id] = los_status(sample, path)
        if include_delays:
            delay_spreads[sample_id] = delay_spread_ns(sample)
            first_path_delays[sample_id] = delay_ns(sample, "first_path_delay_s")
            los_delays[sample_id] = delay_ns(sample, "los_delay_s")
    del samples
    gc.collect()
    return statuses, delay_spreads, first_path_delays, los_delays


def read_excluded_ids(paths: list[Path]) -> set[str]:
    excluded = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        excluded.update(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return excluded


def shuffled(values: list[str], seed: int) -> list[str]:
    result = sorted(values)
    random.Random(seed).shuffle(result)
    return result


def digest(values: list[str]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def save_selected(
    *,
    name: str,
    path: Path,
    selected_ids: list[str],
    reference_statuses: dict[str, str],
    output_dir: Path,
) -> Path:
    selected_order = {sample_id: index for index, sample_id in enumerate(selected_ids)}
    selected: list[object | None] = [None] * len(selected_ids)
    for sample in load_samples(path):
        sample_id = group_id(sample, path)
        index = selected_order.get(sample_id)
        if index is None:
            continue
        if los_status(sample, path) != reference_statuses[sample_id]:
            raise ValueError(f"LoS/NLoS mismatch for group_id={sample_id!r}.")
        selected[index] = sample
    if any(sample is None for sample in selected):
        raise ValueError(f"{name} is missing one or more selected group IDs.")
    output_path = output_dir / f"{name}_final_test.pt"
    torch.save(selected, output_path)
    print(f"saved_configuration={name} samples={len(selected)} path={output_path}")
    del selected
    gc.collect()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Candidate configuration as NAME=PATH. Repeat for every nf.",
    )
    parser.add_argument(
        "--exclude-ids",
        action="append",
        type=Path,
        default=[],
        help="Text file containing previously used group IDs. Repeat as needed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-status", type=int, default=10_000)
    parser.add_argument("--max-delay-ns", type=float, default=1920.0)
    parser.add_argument("--max-delay-spread-ns", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=45_678)
    args = parser.parse_args()

    if args.samples_per_status <= 0:
        raise ValueError("--samples-per-status must be positive.")
    if args.max_delay_ns <= 0.0:
        raise ValueError("--max-delay-ns must be positive.")
    if args.max_delay_spread_ns <= 0.0:
        raise ValueError("--max-delay-spread-ns must be positive.")
    named_paths = [parse_named_path(value) for value in args.input]
    names = [name for name, _ in named_paths]
    if len(set(names)) != len(names):
        raise ValueError("Input configuration names must be unique.")

    status_maps = {}
    reference_delay_spreads = None
    reference_first_path_delays = None
    reference_los_delays = None
    shared_ids: set[str] | None = None
    for index, (name, path) in enumerate(named_paths):
        statuses, spreads, first_path_delays, los_delays = scan(
            path,
            include_delays=index == 0,
        )
        status_maps[name] = statuses
        if index == 0:
            reference_delay_spreads = spreads
            reference_first_path_delays = first_path_delays
            reference_los_delays = los_delays
        shared_ids = set(statuses) if shared_ids is None else shared_ids & set(statuses)
        counts = Counter(statuses.values())
        print(
            f"scanned_configuration={name} samples={len(statuses)} "
            f"los={counts['los']} nlos={counts['nlos']}"
        )

    assert shared_ids is not None
    assert reference_delay_spreads is not None
    assert reference_first_path_delays is not None
    assert reference_los_delays is not None
    reference_name = names[0]
    reference_statuses = status_maps[reference_name]
    mismatches = [
        sample_id
        for sample_id in shared_ids
        if any(
            status_maps[name][sample_id] != reference_statuses[sample_id]
            for name in names[1:]
        )
    ]
    if mismatches:
        raise ValueError(
            f"Found {len(mismatches)} cross-configuration status mismatches."
        )

    excluded_ids = read_excluded_ids(args.exclude_ids)
    available_ids = shared_ids - excluded_ids
    if available_ids & excluded_ids:
        raise AssertionError("Excluded group IDs remain available.")
    eligible_ids = {
        sample_id
        for sample_id in available_ids
        if math.isfinite(reference_delay_spreads[sample_id])
        and reference_delay_spreads[sample_id] < args.max_delay_spread_ns
        and 0.0 <= reference_first_path_delays[sample_id] < args.max_delay_ns
        and (
            reference_statuses[sample_id] == "nlos"
            or 0.0 <= reference_los_delays[sample_id] < args.max_delay_ns
        )
    }
    by_status = {
        status: [
            sample_id
            for sample_id in eligible_ids
            if reference_statuses[sample_id] == status
        ]
        for status in ("los", "nlos")
    }
    for status, candidates in by_status.items():
        if len(candidates) < args.samples_per_status:
            raise ValueError(
                f"Only {len(candidates)} unused eligible {status} samples remain; "
                f"{args.samples_per_status} requested."
            )

    selected_los = shuffled(by_status["los"], args.seed)[: args.samples_per_status]
    selected_nlos = shuffled(by_status["nlos"], args.seed + 1)[
        : args.samples_per_status
    ]
    selected_ids = shuffled(selected_los + selected_nlos, args.seed + 2)
    if set(selected_ids) & excluded_ids:
        raise AssertionError("Final holdout overlaps excluded group IDs.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.output_dir / "final_test_group_ids.txt"
    ids_path.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    outputs = {}
    for name, path in named_paths:
        outputs[name] = str(
            save_selected(
                name=name,
                path=path,
                selected_ids=selected_ids,
                reference_statuses=reference_statuses,
                output_dir=args.output_dir,
            )
        )

    manifest = {
        "seed": args.seed,
        "samples_per_status": args.samples_per_status,
        "max_delay_ns": args.max_delay_ns,
        "max_delay_spread_ns": args.max_delay_spread_ns,
        "common_group_count": len(shared_ids),
        "excluded_group_count": len(excluded_ids),
        "shared_excluded_group_count": len(shared_ids & excluded_ids),
        "unused_common_group_count": len(available_ids),
        "eligible_unused_los_count": len(by_status["los"]),
        "eligible_unused_nlos_count": len(by_status["nlos"]),
        "final_test_count": len(selected_ids),
        "final_test_los_count": len(selected_los),
        "final_test_nlos_count": len(selected_nlos),
        "excluded_overlap_count": len(set(selected_ids) & excluded_ids),
        "final_test_group_ids_sha256": digest(selected_ids),
        "exclude_id_files": [str(path) for path in args.exclude_ids],
        "inputs": {name: str(path) for name, path in named_paths},
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "final_holdout_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"final_test_count={len(selected_ids)}")
    print(f"final_test_los_count={len(selected_los)}")
    print(f"final_test_nlos_count={len(selected_nlos)}")
    print("excluded_overlap_count=0")
    print(f"saved_manifest={manifest_path}")


if __name__ == "__main__":
    main()
