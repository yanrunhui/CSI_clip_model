from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


GROUP_ID_RE = re.compile(
    r"^(?P<dataset>.+)-map_(?P<map>\d+)-source_(?P<source>\d+)-rx_(?P<rx>\d+)$"
)
PROP_HEADER_BYTES = 32
PROP_RX_RECORD_BYTES = 32
PROP_MAGIC = {b"PORP", b"PROP"}
PROP_RX_DTYPE = np.dtype(
    [
        ("point_id", "<u4"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("rss_dbm", "<f4"),
        ("path_loss_db", "<f4"),
        ("path_count", "<u4"),
        ("path_offset", "<u4"),
    ]
)


def parse_group_id(group_id: str) -> tuple[int, int, int]:
    match = GROUP_ID_RE.fullmatch(group_id)
    if match is None:
        raise ValueError(
            f"Unsupported group_id={group_id!r}; expected a D2Los ID ending in "
            "-map_N-source_N-rx_N."
        )
    return (
        int(match.group("map")),
        int(match.group("source")),
        int(match.group("rx")),
    )


def los_status(sample) -> str:
    status = str(getattr(getattr(sample, "semantic_key", None), "los_status", "")).lower()
    if status not in {"los", "nlos"}:
        raise ValueError(f"Unsupported LoS status {status!r} in {sample.group_id!r}.")
    return status


def split_random_stratified(
    statuses: Sequence[str], test_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    by_status: dict[str, list[int]] = defaultdict(list)
    for index, status in enumerate(statuses):
        by_status[status].append(index)

    rng = random.Random(seed)
    test: list[int] = []
    train: list[int] = []
    for status in sorted(by_status):
        indices = by_status[status][:]
        rng.shuffle(indices)
        test_count = int(round(len(indices) * test_fraction))
        test_count = min(max(test_count, 1), len(indices) - 1)
        test.extend(indices[:test_count])
        train.extend(indices[test_count:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def subsample_stratified(
    indices: Sequence[int],
    statuses: Sequence[str],
    target_count: int | None,
    seed: int,
) -> list[int]:
    """Select an exact-size subset while approximately preserving LoS/NLoS ratio."""
    selected = list(indices)
    if target_count is None or target_count == len(selected):
        return selected
    if target_count <= 0:
        raise ValueError("Requested subset count must be positive.")
    if target_count > len(selected):
        raise ValueError(
            f"Requested {target_count} samples from a pool containing only {len(selected)}."
        )

    by_status: dict[str, list[int]] = defaultdict(list)
    for index in selected:
        by_status[statuses[index]].append(index)
    exact = {
        status: target_count * len(status_indices) / len(selected)
        for status, status_indices in by_status.items()
    }
    allocations = {
        status: min(int(math.floor(exact[status])), len(status_indices))
        for status, status_indices in by_status.items()
    }
    remaining = target_count - sum(allocations.values())
    allocation_order = sorted(
        by_status,
        key=lambda status: (
            exact[status] - allocations[status],
            len(by_status[status]) - allocations[status],
            status,
        ),
        reverse=True,
    )
    while remaining:
        progressed = False
        for status in allocation_order:
            if allocations[status] >= len(by_status[status]):
                continue
            allocations[status] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("Unable to allocate the requested stratified subset.")

    rng = random.Random(seed)
    result: list[int] = []
    for status in sorted(by_status):
        status_indices = by_status[status][:]
        rng.shuffle(status_indices)
        result.extend(status_indices[: allocations[status]])
    rng.shuffle(result)
    return result


def _choose_groups_near_target(
    group_counts: dict[tuple, int], test_fraction: float, seed: int
) -> set[tuple]:
    """Choose indivisible groups close to the requested sample count."""
    keys = sorted(group_counts)
    random.Random(seed).shuffle(keys)
    target = sum(group_counts.values()) * test_fraction
    chosen: set[tuple] = set()
    current = 0
    for key in keys:
        candidate = current + group_counts[key]
        if abs(candidate - target) < abs(current - target):
            chosen.add(key)
            current = candidate
    if not chosen:
        chosen.add(min(keys, key=lambda key: abs(group_counts[key] - target)))
    if len(chosen) == len(keys):
        chosen.remove(min(chosen, key=lambda key: group_counts[key]))
    return chosen


def _choose_groups_balanced_across_strata(
    group_indices: dict[tuple, list[int]],
    strata: Sequence[str],
    test_fraction: float,
    seed: int,
) -> set[tuple]:
    """Keep groups intact while matching total and per-stratum test fractions."""
    keys = sorted(group_indices)
    labels = sorted(set(strata))
    totals = Counter(strata)
    targets = {"__all__": len(strata) * test_fraction}
    targets.update({label: totals[label] * test_fraction for label in labels})
    vectors = {}
    for key, indices in group_indices.items():
        counts = Counter(strata[index] for index in indices)
        vectors[key] = {
            "__all__": len(indices),
            **{label: counts[label] for label in labels},
        }

    def score(chosen: set[tuple]) -> float:
        errors = []
        for label, target in targets.items():
            observed = sum(vectors[key][label] for key in chosen)
            errors.append((observed - target) / max(target, 1.0))
        return sum(error * error for error in errors)

    rng = random.Random(seed)
    ideal_group_count = len(keys) * test_fraction
    candidate_group_counts = sorted(
        {
            min(max(int(round(ideal_group_count)) + offset, 1), len(keys) - 1)
            for offset in (-2, -1, 0, 1, 2)
        }
    )
    best_chosen: set[tuple] | None = None
    best_score = math.inf
    for group_count in candidate_group_counts:
        for _ in range(12):
            chosen = set(rng.sample(keys, group_count))
            current_score = score(chosen)
            while True:
                best_swap = None
                best_swap_score = current_score
                rejected = [key for key in keys if key not in chosen]
                for old_key in sorted(chosen):
                    for new_key in rejected:
                        candidate = (chosen - {old_key}) | {new_key}
                        candidate_score = score(candidate)
                        if candidate_score + 1e-15 < best_swap_score:
                            best_swap = (old_key, new_key)
                            best_swap_score = candidate_score
                if best_swap is None:
                    break
                chosen.remove(best_swap[0])
                chosen.add(best_swap[1])
                current_score = best_swap_score
            if current_score < best_score:
                best_chosen = chosen
                best_score = current_score
    assert best_chosen is not None
    return best_chosen


def split_by_groups(
    group_keys: Sequence[tuple],
    test_fraction: float,
    seed: int,
    per_map: bool,
    strata: Sequence[str] | None = None,
) -> tuple[list[int], list[int]]:
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for index, key in enumerate(group_keys):
        grouped[key].append(index)

    test_groups: set[tuple] = set()
    if per_map:
        groups_by_map: dict[int, dict[tuple, int]] = defaultdict(dict)
        for key, indices in grouped.items():
            groups_by_map[int(key[0])][key] = len(indices)
        single_block_groups: list[tuple] = []
        for map_id, counts in sorted(groups_by_map.items()):
            if len(counts) < 2:
                single_block_groups.extend(counts)
                continue
            test_groups.update(
                _choose_groups_near_target(counts, test_fraction, seed + 10_007 * map_id)
            )
        if single_block_groups:
            # A one-block map cannot appear on both sides without spatial leakage.
            # Assign each such block wholly to the side that best preserves the
            # requested global test fraction.
            rng = random.Random(seed + 97_531)
            rng.shuffle(single_block_groups)
            target_test_count = len(group_keys) * test_fraction
            current_test_count = sum(len(grouped[key]) for key in test_groups)
            for key in single_block_groups:
                candidate_count = current_test_count + len(grouped[key])
                if abs(candidate_count - target_test_count) < abs(
                    current_test_count - target_test_count
                ):
                    test_groups.add(key)
                    current_test_count = candidate_count
            print(
                f"single_block_maps={len(single_block_groups)} "
                f"single_blocks_assigned_to_test={sum(key in test_groups for key in single_block_groups)} "
                "policy=whole_block_global_balance",
                flush=True,
            )
        if not test_groups:
            smallest = min(grouped, key=lambda key: len(grouped[key]))
            test_groups.add(smallest)
        if len(test_groups) == len(grouped):
            largest = max(test_groups, key=lambda key: len(grouped[key]))
            test_groups.remove(largest)
    else:
        if strata is None:
            counts = {key: len(indices) for key, indices in grouped.items()}
            test_groups = _choose_groups_near_target(counts, test_fraction, seed)
        else:
            if len(strata) != len(group_keys):
                raise ValueError("strata and group_keys must have the same length.")
            test_groups = _choose_groups_balanced_across_strata(
                grouped, strata, test_fraction, seed
            )

    test = [index for index, key in enumerate(group_keys) if key in test_groups]
    train = [index for index, key in enumerate(group_keys) if key not in test_groups]
    random.Random(seed + 1).shuffle(train)
    random.Random(seed + 2).shuffle(test)
    return train, test


def _propbin_path(d2los_root: Path, map_id: int, source_id: int) -> Path:
    candidates = sorted(
        (d2los_root / f"map_{map_id}").glob(
            f"special_points_propbin_*/source_{source_id}.propbin*"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one propbin for map_{map_id}/source_{source_id}, "
            f"found {len(candidates)} under {d2los_root}."
        )
    return candidates[0]


def _read_selected_rx_coordinates(path: Path, rx_indices: Iterable[int]) -> dict[int, tuple[float, float, float]]:
    wanted = sorted(set(int(index) for index in rx_indices))
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        header = handle.read(PROP_HEADER_BYTES)
        if len(header) != PROP_HEADER_BYTES or header[:4] not in PROP_MAGIC:
            raise ValueError(f"Invalid RayVerse propbin header: {path}")
        rx_count = int(np.frombuffer(header, dtype="<u4", count=1, offset=20)[0])
        if wanted and wanted[-1] >= rx_count:
            raise IndexError(
                f"rx_{wanted[-1]} is outside {path} (rx_count={rx_count})."
            )
        raw = handle.read(rx_count * PROP_RX_RECORD_BYTES)
    if len(raw) != rx_count * PROP_RX_RECORD_BYTES:
        raise EOFError(f"Truncated receiver table in {path}.")
    records = np.frombuffer(raw, dtype=PROP_RX_DTYPE, count=rx_count)
    return {
        index: (
            float(records[index]["x"]),
            float(records[index]["y"]),
            float(records[index]["z"]),
        )
        for index in wanted
    }


def load_coordinates(
    d2los_root: Path, parsed_ids: Sequence[tuple[int, int, int]]
) -> list[tuple[float, float, float]]:
    requests: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for sample_index, (map_id, source_id, rx_index) in enumerate(parsed_ids):
        requests[(map_id, source_id)].append((sample_index, rx_index))

    coordinates: list[tuple[float, float, float] | None] = [None] * len(parsed_ids)
    for file_number, ((map_id, source_id), indexed_rx) in enumerate(
        sorted(requests.items()), start=1
    ):
        path = _propbin_path(d2los_root, map_id, source_id)
        lookup = _read_selected_rx_coordinates(path, (rx for _, rx in indexed_rx))
        for sample_index, rx_index in indexed_rx:
            coordinates[sample_index] = lookup[rx_index]
        if file_number % 500 == 0 or file_number == len(requests):
            print(f"coordinate_files_scanned={file_number}/{len(requests)}", flush=True)
    if any(coordinate is None for coordinate in coordinates):
        raise RuntimeError("Internal error: some receiver coordinates were not resolved.")
    return [coordinate for coordinate in coordinates if coordinate is not None]


def spatial_block_keys(
    parsed_ids: Sequence[tuple[int, int, int]],
    coordinates: Sequence[tuple[float, float, float]],
    block_size_m: float,
) -> list[tuple[int, int, int]]:
    return [
        (map_id, math.floor(x / block_size_m), math.floor(y / block_size_m))
        for (map_id, _, _), (x, y, _) in zip(parsed_ids, coordinates)
    ]


def _sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _overlap_summary(
    train: Sequence[int],
    test: Sequence[int],
    parsed_ids: Sequence[tuple[int, int, int]],
    coordinates: Sequence[tuple[float, float, float]],
    block_keys: Sequence[tuple[int, int, int]],
) -> dict[str, int | float]:
    train_maps = {parsed_ids[index][0] for index in train}
    test_maps = {parsed_ids[index][0] for index in test}
    train_sources = {parsed_ids[index][:2] for index in train}
    test_sources = {parsed_ids[index][:2] for index in test}
    train_blocks = {block_keys[index] for index in train}
    test_blocks = {block_keys[index] for index in test}
    train_positions = {
        (parsed_ids[index][0], *coordinates[index]) for index in train
    }
    test_positions = {
        (parsed_ids[index][0], *coordinates[index]) for index in test
    }
    return {
        "map_overlap_count": len(train_maps & test_maps),
        "test_map_overlap_rate": len(train_maps & test_maps) / len(test_maps),
        "map_source_overlap_count": len(train_sources & test_sources),
        "test_map_source_overlap_rate": len(train_sources & test_sources) / len(test_sources),
        "spatial_block_overlap_count": len(train_blocks & test_blocks),
        "exact_position_overlap_count": len(train_positions & test_positions),
    }


def summarize_split(
    name: str,
    train: Sequence[int],
    test: Sequence[int],
    group_ids: Sequence[str],
    statuses: Sequence[str],
    parsed_ids: Sequence[tuple[int, int, int]],
    coordinates: Sequence[tuple[float, float, float]],
    block_keys: Sequence[tuple[int, int, int]],
    require_complete_partition: bool = True,
) -> dict:
    train_set = set(train)
    test_set = set(test)
    if train_set & test_set:
        raise ValueError(f"{name} train/test selections overlap.")
    if not train_set | test_set <= set(range(len(group_ids))):
        raise ValueError(f"{name} contains an out-of-range sample index.")
    if require_complete_partition and train_set | test_set != set(range(len(group_ids))):
        raise ValueError(f"{name} does not form an exact, disjoint partition.")
    train_statuses = Counter(statuses[index] for index in train)
    test_statuses = Counter(statuses[index] for index in test)
    overlap = _overlap_summary(
        train, test, parsed_ids, coordinates, block_keys
    )
    summary = {
        "train_count": len(train),
        "test_count": len(test),
        "test_fraction": len(test) / max(len(train) + len(test), 1),
        "unused_candidate_count": len(group_ids) - len(train) - len(test),
        "train_los_count": train_statuses["los"],
        "train_nlos_count": train_statuses["nlos"],
        "test_los_count": test_statuses["los"],
        "test_nlos_count": test_statuses["nlos"],
        "train_group_ids_sha256": _sha256_lines(group_ids[index] for index in train),
        "test_group_ids_sha256": _sha256_lines(group_ids[index] for index in test),
        **overlap,
    }
    print(
        f"split={name} train={len(train)} test={len(test)} "
        f"map_overlap={overlap['map_overlap_count']} "
        f"source_overlap={overlap['map_source_overlap_count']} "
        f"block_overlap={overlap['spatial_block_overlap_count']} "
        f"position_overlap={overlap['exact_position_overlap_count']}",
        flush=True,
    )
    return summary


def write_id_file(path: Path, indices: Sequence[int], group_ids: Sequence[str]) -> None:
    path.write_text(
        "".join(f"{group_ids[index]}\n" for index in indices), encoding="utf-8"
    )


def save_split_samples(
    output_dir: Path,
    name: str,
    train: Sequence[int],
    test: Sequence[int],
    samples: Sequence,
) -> dict[str, str]:
    train_path = output_dir / f"{name}_train.pt"
    test_path = output_dir / f"{name}_test.pt"
    print(f"saving={train_path}", flush=True)
    torch.save([samples[index] for index in train], train_path)
    print(f"saving={test_path}", flush=True)
    torch.save([samples[index] for index in test], test_path)
    return {"train": str(train_path), "test": str(test_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create comparable sample-random, spatial-block-disjoint, and "
            "scenario/map-disjoint D2Los splits."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--d2los-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=23_421)
    parser.add_argument("--block-size-m", type=float, default=32.0)
    parser.add_argument(
        "--candidate-samples",
        type=int,
        help=(
            "LoS/NLoS-stratified candidate pool selected before coordinate loading. "
            "Use more than train-samples + test-samples to leave room for whole-block assignment."
        ),
    )
    parser.add_argument("--train-samples", type=int, help="Exact saved train size per split.")
    parser.add_argument("--test-samples", type=int, help="Exact saved test size per split.")
    parser.add_argument(
        "--split-types",
        nargs="+",
        choices=("random", "spatial_block", "scenario_disjoint"),
        default=("random", "spatial_block", "scenario_disjoint"),
        help="Only calculate and save the requested split types.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write group-ID memberships and diagnostics without duplicating the large .pt payload.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.d2los_root.exists():
        raise FileNotFoundError(args.d2los_root)
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between zero and one.")
    if args.block_size_m <= 0.0:
        raise ValueError("--block-size-m must be positive.")
    if args.candidate_samples is not None and args.candidate_samples <= 0:
        raise ValueError("--candidate-samples must be positive.")
    if args.train_samples is not None and args.train_samples <= 0:
        raise ValueError("--train-samples must be positive.")
    if args.test_samples is not None and args.test_samples <= 0:
        raise ValueError("--test-samples must be positive.")
    requested_total = (args.train_samples or 0) + (args.test_samples or 0)
    if args.candidate_samples is not None and requested_total > args.candidate_samples:
        raise ValueError(
            "--candidate-samples must be at least --train-samples + --test-samples."
        )

    print(f"loading_input={args.input} mmap=true", flush=True)
    all_samples = torch.load(args.input, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(all_samples, list) or len(all_samples) < 2:
        raise ValueError(f"Expected a sample list with at least two entries in {args.input}.")
    original_input_count = len(all_samples)
    all_statuses = [los_status(sample) for sample in all_samples]
    candidate_indices = subsample_stratified(
        range(original_input_count),
        all_statuses,
        args.candidate_samples,
        args.seed - 1,
    )
    samples = [all_samples[index] for index in candidate_indices]
    del all_samples
    group_ids = [str(getattr(sample, "group_id", "")).strip() for sample in samples]
    if any(not group_id for group_id in group_ids):
        raise ValueError("The input contains a sample without group_id.")
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("The input contains duplicate group_id values.")
    statuses = [los_status(sample) for sample in samples]
    parsed_ids = [parse_group_id(group_id) for group_id in group_ids]
    print(
        f"input_samples={len(samples)} maps={len({item[0] for item in parsed_ids})} "
        f"map_sources={len({item[:2] for item in parsed_ids})} "
        f"los={statuses.count('los')} nlos={statuses.count('nlos')}",
        flush=True,
    )

    coordinates = load_coordinates(args.d2los_root, parsed_ids)
    block_keys = spatial_block_keys(parsed_ids, coordinates, args.block_size_m)
    splits = {}
    if "random" in args.split_types:
        splits["random"] = split_random_stratified(
            statuses, args.test_fraction, args.seed
        )
    if "spatial_block" in args.split_types:
        splits["spatial_block"] = split_by_groups(
            block_keys, args.test_fraction, args.seed + 100, per_map=True
        )
    if "scenario_disjoint" in args.split_types:
        scenario_keys = [(map_id,) for map_id, _, _ in parsed_ids]
        splits["scenario_disjoint"] = split_by_groups(
            scenario_keys,
            args.test_fraction,
            args.seed + 200,
            per_map=False,
            strata=statuses,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    outputs = {}
    for name, (train, test) in splits.items():
        train = subsample_stratified(
            train,
            statuses,
            args.train_samples,
            args.seed + 1_000 + sum(map(ord, name)),
        )
        test = subsample_stratified(
            test,
            statuses,
            args.test_samples,
            args.seed + 2_000 + sum(map(ord, name)),
        )
        write_id_file(args.output_dir / f"{name}_train_group_ids.txt", train, group_ids)
        write_id_file(args.output_dir / f"{name}_test_group_ids.txt", test, group_ids)
        summaries[name] = summarize_split(
            name,
            train,
            test,
            group_ids,
            statuses,
            parsed_ids,
            coordinates,
            block_keys,
            require_complete_partition=(
                args.train_samples is None and args.test_samples is None
            ),
        )
        if not args.manifest_only:
            outputs[name] = save_split_samples(
                args.output_dir, name, train, test, samples
            )

    manifest = {
        "input": str(args.input),
        "d2los_root": str(args.d2los_root),
        "input_count": original_input_count,
        "candidate_count": len(samples),
        "candidate_samples_requested": args.candidate_samples,
        "train_samples_requested": args.train_samples,
        "test_samples_requested": args.test_samples,
        "split_types": list(args.split_types),
        "seed": args.seed,
        "requested_test_fraction": args.test_fraction,
        "spatial_block_size_m": args.block_size_m,
        "definitions": {
            "random": "LoS/NLoS-stratified sample-level random split.",
            "spatial_block": (
                "Whole (map, floor(x/block_size), floor(y/block_size)) receiver blocks "
                "are assigned to one side only. Multi-block maps contribute both sides; "
                "a map represented by one block is assigned wholly to one side."
            ),
            "scenario_disjoint": (
                "Whole maps are assigned to one side only; map selection jointly "
                "approximates the requested sample count and global LoS/NLoS ratio."
            ),
        },
        "summaries": summaries,
        "outputs": outputs,
        "manifest_only": args.manifest_only,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved_manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
