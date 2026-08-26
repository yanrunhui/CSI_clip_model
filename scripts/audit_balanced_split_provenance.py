from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SPLITS = ("random", "spatial_block", "scenario_disjoint")
SUBSETS = ("train", "test")
DEFAULT_SEED_OFFSETS = {
    ("random", "train"): 0,
    ("random", "test"): 1,
    ("spatial_block", "train"): 10,
    ("spatial_block", "test"): 11,
    ("scenario_disjoint", "train"): 20,
    ("scenario_disjoint", "test"): 21,
}
GROUP_ID_RE = re.compile(
    r"^(?P<dataset>.+)-map_(?P<map>\d+)-source_(?P<source>\d+)-rx_(?P<rx>\d+)$"
)


def parse_group_id(group_id: str) -> tuple[int, int, int]:
    match = GROUP_ID_RE.fullmatch(group_id)
    if match is None:
        raise ValueError(
            f"Unsupported group_id={group_id!r}; expected an ID ending in "
            "-map_N-source_N-rx_N."
        )
    return (
        int(match.group("map")),
        int(match.group("source")),
        int(match.group("rx")),
    )


def sample_status(sample) -> str:
    status = str(
        getattr(getattr(sample, "semantic_key", None), "los_status", "")
    ).lower()
    if status not in {"los", "nlos"}:
        raise ValueError(
            f"Unsupported los_status={status!r} in "
            f"sample {getattr(sample, 'group_id', '')!r}."
        )
    return status


def sha256_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, required: bool = True) -> dict:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def read_id_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not value for value in values):
        raise ValueError(f"Blank group ID in {path}.")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate group IDs in {path}.")
    for value in values:
        parse_group_id(value)
    return values


def load_source_metadata(path: Path) -> dict[str, dict[str, int | str]]:
    print(f"loading_source={path} mmap=true", flush=True)
    samples = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {path}.")
    metadata: dict[str, dict[str, int | str]] = {}
    for index, sample in enumerate(samples):
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError(f"Source sample {index} has no group_id.")
        if group_id in metadata:
            raise ValueError(f"Duplicate source group_id={group_id!r}.")
        map_id, source_id, rx_index = parse_group_id(group_id)
        metadata[group_id] = {
            "source_index": index,
            "los_status": sample_status(sample),
            "map_id": map_id,
            "source_id": source_id,
            "rx_index": rx_index,
        }
    print(f"source_samples={len(metadata)}", flush=True)
    del samples
    gc.collect()
    return metadata


def load_subset_records(path: Path) -> list[dict[str, int | str]]:
    print(f"loading_subset={path} mmap=true", flush=True)
    samples = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {path}.")
    records = []
    seen = set()
    for index, sample in enumerate(samples):
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError(f"Subset sample {index} in {path} has no group_id.")
        if group_id in seen:
            raise ValueError(f"Duplicate subset group_id={group_id!r} in {path}.")
        seen.add(group_id)
        map_id, source_id, rx_index = parse_group_id(group_id)
        records.append(
            {
                "subset_rank": index,
                "group_id": group_id,
                "los_status": sample_status(sample),
                "map_id": map_id,
                "source_id": source_id,
                "rx_index": rx_index,
            }
        )
    del samples
    gc.collect()
    return records


def replay_balanced_subset_ids(
    parent_ids: Sequence[str],
    source_metadata: dict[str, dict[str, int | str]],
    samples_per_status: int,
    seed: int,
) -> list[str]:
    pools = {"los": [], "nlos": []}
    for parent_index, group_id in enumerate(parent_ids):
        if group_id not in source_metadata:
            raise ValueError(f"Parent group_id={group_id!r} is absent from source data.")
        status = str(source_metadata[group_id]["los_status"])
        pools[status].append(parent_index)
    for status in ("los", "nlos"):
        if len(pools[status]) < samples_per_status:
            raise ValueError(
                f"Parent split has only {len(pools[status])} {status} samples; "
                f"{samples_per_status} requested."
            )
    rng = random.Random(seed)
    selected_indices = (
        rng.sample(pools["los"], samples_per_status)
        + rng.sample(pools["nlos"], samples_per_status)
    )
    rng.shuffle(selected_indices)
    return [parent_ids[index] for index in selected_indices]


def manifest_seed(
    subset_manifest: dict,
    split: str,
    subset: str,
    base_seed: int,
) -> tuple[int, str]:
    record = subset_manifest.get("outputs", {}).get(f"{split}_{subset}", {})
    if "seed" in record:
        return int(record["seed"]), "balanced_subset_manifest"
    return base_seed + DEFAULT_SEED_OFFSETS[(split, subset)], "derived_default"


def expected_per_status(subset: str, train_per_status: int, test_per_status: int) -> int:
    return train_per_status if subset == "train" else test_per_status


def check_parent_manifest_summary(
    parent_manifest: dict,
    split: str,
    subset: str,
    parent_ids: Sequence[str],
    statuses: Counter,
) -> dict:
    summary = parent_manifest.get("summaries", {}).get(split, {})
    checks = {
        "manifest_summary_present": bool(summary),
        "count_matches_manifest": None,
        "los_count_matches_manifest": None,
        "nlos_count_matches_manifest": None,
        "id_sha256_matches_manifest": None,
    }
    if not summary:
        return checks
    checks["count_matches_manifest"] = int(summary[f"{subset}_count"]) == len(parent_ids)
    checks["los_count_matches_manifest"] = (
        int(summary[f"{subset}_los_count"]) == statuses["los"]
    )
    checks["nlos_count_matches_manifest"] = (
        int(summary[f"{subset}_nlos_count"]) == statuses["nlos"]
    )
    digest_key = f"{subset}_group_ids_sha256"
    checks["id_sha256_matches_manifest"] = (
        str(summary[digest_key]) == sha256_lines(parent_ids)
        if digest_key in summary
        else None
    )
    for name, passed in checks.items():
        if passed is False:
            raise ValueError(
                f"Parent manifest check failed for {split}_{subset}: {name}."
            )
    return checks


def write_membership_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "split",
        "subset",
        "subset_rank",
        "parent_rank",
        "source_index",
        "group_id",
        "map_id",
        "source_id",
        "rx_index",
        "los_status",
        "in_expected_parent_split",
        "in_opposite_parent_split",
        "source_status_matches_subset",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_recorded_path(path_value: object) -> Path | None:
    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(str(path_value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
        rows.append(row)
    if not rows:
        raise ValueError(f"No JSON records found in {path}.")
    return rows


def audit_experiment_results(
    *,
    result_root: Path,
    result_seed: int,
    checkpoint_name: str,
    subset_dir: Path,
    subset_ids_by_key: dict[tuple[str, str], list[str]],
) -> dict:
    """Tie each trained checkpoint and saved evaluation payload to audited subsets."""
    experiment: dict[str, dict] = {}
    for split in SPLITS:
        run_dir = result_root / split / f"seed_{result_seed}"
        checkpoint_path = run_dir / checkpoint_name
        train_log_path = run_dir / "train_log.jsonl"
        signal_path = run_dir / "signal_descriptions.pt"

        print(f"loading_checkpoint_metadata={checkpoint_path} mmap=true", flush=True)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Expected a checkpoint dictionary in {checkpoint_path}.")
        checkpoint_args = checkpoint.get("args", {})
        if not isinstance(checkpoint_args, dict):
            raise ValueError(f"Checkpoint args are missing in {checkpoint_path}.")

        expected_train_path = (subset_dir / f"{split}_train.pt").resolve()
        recorded_train_path = resolve_recorded_path(checkpoint_args.get("data_path"))
        train_path_match = recorded_train_path == expected_train_path
        if not train_path_match:
            raise ValueError(
                f"{split} checkpoint records data_path={recorded_train_path}, "
                f"expected {expected_train_path}."
            )
        additional_paths = checkpoint_args.get("additional_data_paths", [])
        if additional_paths not in (None, [], ()):
            raise ValueError(
                f"{split} checkpoint used additional_data_paths={additional_paths!r}."
            )
        checkpoint_seed = int(checkpoint_args.get("seed", -1))
        if checkpoint_seed != result_seed:
            raise ValueError(
                f"{split} checkpoint seed={checkpoint_seed}, expected {result_seed}."
            )

        epochs = int(checkpoint.get("epoch", checkpoint_args.get("epochs", 0)))
        batch_size = int(checkpoint_args.get("batch_size", 0))
        if epochs <= 0 or batch_size <= 0:
            raise ValueError(f"Invalid epoch/batch metadata in {checkpoint_path}.")
        expected_train_count = len(subset_ids_by_key[(split, "train")])
        # scripts/pretrain.py builds the training DataLoader with drop_last=True.
        # Therefore a complete loader epoch uses floor(N / batch_size) batches.
        training_drop_last = True
        expected_steps = expected_train_count // batch_size
        samples_consumed_per_epoch = expected_steps * batch_size
        samples_dropped_per_epoch = expected_train_count - samples_consumed_per_epoch
        all_log_rows = read_jsonl(train_log_path)
        if len(all_log_rows) < epochs:
            raise ValueError(
                f"{train_log_path} has {len(all_log_rows)} rows, fewer than "
                f"checkpoint epoch {epochs}."
            )
        epoch_rows = all_log_rows[-epochs:]
        logged_epochs = [int(row.get("epoch", -1)) for row in epoch_rows]
        if logged_epochs != list(range(1, epochs + 1)):
            raise ValueError(
                f"The final {epochs} rows of {train_log_path} are not epochs 1..{epochs}."
            )
        logged_steps = [int(row.get("steps", -1)) for row in epoch_rows]
        all_steps_match = all(step == expected_steps for step in logged_steps)
        if not all_steps_match:
            raise ValueError(
                f"{split} logged steps do not all equal expected full-epoch "
                f"steps={expected_steps}; observed={sorted(set(logged_steps))}."
            )

        print(f"loading_evaluation_membership={signal_path} mmap=true", flush=True)
        signal_payload = torch.load(
            signal_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(signal_payload, dict):
            raise ValueError(f"Expected a dictionary in {signal_path}.")
        comparisons = signal_payload.get("comparisons")
        if not isinstance(comparisons, list):
            raise ValueError(f"Missing comparisons list in {signal_path}.")
        evaluated_ids = [str(row.get("group_id", "")) for row in comparisons]
        if any(not group_id for group_id in evaluated_ids):
            raise ValueError(f"Missing evaluation group_id in {signal_path}.")
        expected_test_ids = subset_ids_by_key[(split, "test")]
        evaluation_exact_order_match = evaluated_ids == expected_test_ids
        if not evaluation_exact_order_match:
            raise ValueError(
                f"{split} evaluation IDs do not exactly match {split}_test.pt."
            )

        filtering_args = {
            name: checkpoint_args.get(name)
            for name in (
                "min_class_size",
                "filter_attribute_values",
                "limit_samples",
                "limit_samples_by_attribute",
                "limit_samples_per_attribute_value",
                "max_delay_spread_ns",
            )
        }
        experiment[split] = {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": epochs,
            "checkpoint_seed": checkpoint_seed,
            "recorded_training_data_path": str(recorded_train_path),
            "expected_training_data_path": str(expected_train_path),
            "training_data_path_exact_match": train_path_match,
            "additional_training_data_paths": list(additional_paths or []),
            "checkpoint_filtering_args": filtering_args,
            "batch_size": batch_size,
            "training_dataloader_drop_last": training_drop_last,
            "expected_train_sample_count": expected_train_count,
            "expected_full_epoch_steps": expected_steps,
            "samples_consumed_per_epoch": samples_consumed_per_epoch,
            "samples_dropped_per_epoch_after_shuffle": samples_dropped_per_epoch,
            "training_loader_source": str(ROOT / "scripts/pretrain.py"),
            "training_loader_source_sha256": sha256_file(ROOT / "scripts/pretrain.py"),
            "train_log": str(train_log_path),
            "train_log_total_rows": len(all_log_rows),
            "audited_epoch_rows": len(epoch_rows),
            "logged_step_values": sorted(set(logged_steps)),
            "all_logged_steps_match_full_epoch_calculation": all_steps_match,
            "evaluation_payload": str(signal_path),
            "evaluated_sample_count": len(evaluated_ids),
            "evaluation_group_ids_sha256": sha256_lines(evaluated_ids),
            "evaluation_exact_test_id_order_match": evaluation_exact_order_match,
            "max_steps_per_epoch_provenance_note": (
                "Historical checkpoints do not record max_steps_per_epoch. The log "
                "shows the mathematically expected floor(N/batch_size) steps for the "
                "drop_last=True DataLoader in every epoch, but cannot independently "
                "prove that no equal-valued step cap was set."
            ),
        }
        print(
            f"verified_experiment={split} train_path=true "
            f"epochs={epochs} steps={expected_steps} "
            f"evaluated_ids={len(evaluated_ids)} exact_test_order=true",
            flush=True,
        )
        del checkpoint, signal_payload
        gc.collect()
    return experiment


def write_provenance_report(path: Path, payload: dict) -> None:
    scenario_train = payload["files"]["scenario_disjoint_train"]
    scenario_test = payload["files"]["scenario_disjoint_test"]
    scenario_checks = payload["split_checks"]["scenario_disjoint"]
    lines = [
        "# Balanced split provenance report",
        "",
        "## Original source",
        "",
        f"- File: `{payload['source_data']}`",
        f"- Samples: {payload['source_sample_count']:,}",
        f"- LoS/NLoS: {payload['source_los_count']:,} / {payload['source_nlos_count']:,}",
        "",
        "## Scenario-disjoint parent partition",
        "",
        f"- Train samples: {scenario_train['parent_sample_count']:,}",
        f"- Test samples: {scenario_test['parent_sample_count']:,}",
        f"- Train maps: {scenario_checks['parent_train_map_count']}",
        f"- Test maps: {scenario_checks['parent_test_map_count']}",
        f"- Shared maps: {scenario_checks['parent_map_overlap_count']}",
        f"- Train map IDs: {scenario_train['parent_maps']}",
        f"- Test map IDs: {scenario_test['parent_maps']}",
        "",
        "## Materialized balanced scenario subset",
        "",
        (
            f"- Train: {scenario_train['sample_count']:,} samples = "
            f"{scenario_train['los_count']:,} LoS + "
            f"{scenario_train['nlos_count']:,} NLoS"
        ),
        (
            f"- Test: {scenario_test['sample_count']:,} samples = "
            f"{scenario_test['los_count']:,} LoS + "
            f"{scenario_test['nlos_count']:,} NLoS"
        ),
        f"- Train sampling seed: {scenario_train['seed']}",
        f"- Test sampling seed: {scenario_test['seed']}",
        f"- Shared exact group IDs: {scenario_checks['subset_exact_group_id_overlap_count']}",
        f"- Shared maps: {scenario_checks['subset_map_overlap_count']}",
        f"- Shared map-qualified spatial blocks: {scenario_checks['spatial_block_overlap_count']}",
        f"- Shared map-qualified exact UE positions: {scenario_checks['exact_ue_position_overlap_count']}",
        (
            "- Every train/test ID belongs to its corresponding parent partition: "
            f"{scenario_train['all_ids_in_expected_parent_split'] and scenario_test['all_ids_in_expected_parent_split']}"
        ),
        (
            "- Exact seeded replay (including output order): "
            f"{scenario_train['exact_seeded_replay_order_match'] and scenario_test['exact_seeded_replay_order_match']}"
        ),
        f"- Train ID SHA-256: `{scenario_train['subset_group_ids_sha256']}`",
        f"- Test ID SHA-256: `{scenario_test['subset_group_ids_sha256']}`",
        "",
        "## Reproducibility artifacts",
        "",
        f"- Machine-readable audit: `{payload['output_dir']}/balanced_subset_provenance.json`",
        f"- Per-sample membership table: `{payload['membership_csv']}`",
        f"- Exact ordered ID files: `{payload['output_dir']}/*_group_ids.txt`",
    ]
    if payload.get("experiment_evidence"):
        lines.extend(["", "## Experiment-use evidence", ""])
        for split in SPLITS:
            evidence = payload["experiment_evidence"][split]
            lines.extend(
                [
                    f"### {split}",
                    "",
                    (
                        "- Checkpoint training path matches audited train file: "
                        f"{evidence['training_data_path_exact_match']}"
                    ),
                    (
                        f"- Training log: {evidence['checkpoint_epoch']} epochs, "
                        f"{evidence['expected_full_epoch_steps']} steps/epoch; "
                        f"drop_last={evidence['training_dataloader_drop_last']}; "
                        f"{evidence['samples_dropped_per_epoch_after_shuffle']} "
                        "shuffled-tail samples omitted per epoch"
                    ),
                    (
                        "- Saved evaluation IDs exactly match audited test file: "
                        f"{evidence['evaluation_exact_test_id_order_match']} "
                        f"({evidence['evaluated_sample_count']:,} samples)"
                    ),
                    "",
                ]
            )
    lines.extend(["Audit status: **passed**", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_provenance(
    *,
    source_data: Path,
    parent_dir: Path,
    subset_dir: Path,
    output_dir: Path,
    base_seed: int,
    train_per_status: int,
    test_per_status: int,
    hash_pt_files: bool,
    result_root: Path | None = None,
    result_seed: int = 0,
    checkpoint_name: str = "checkpoint_epoch_100.pt",
    d2los_root: Path | None = None,
    block_size_m: float = 32.0,
) -> dict:
    if train_per_status <= 0 or test_per_status <= 0:
        raise ValueError("Per-status sample targets must be positive.")
    parent_manifest_path = parent_dir / "split_manifest.json"
    subset_manifest_path = subset_dir / "balanced_subset_manifest.json"
    parent_manifest = read_json(parent_manifest_path, required=False)
    subset_manifest = read_json(subset_manifest_path, required=False)
    source_metadata = load_source_metadata(source_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    parent_ids_by_key: dict[tuple[str, str], list[str]] = {}
    subset_ids_by_key: dict[tuple[str, str], list[str]] = {}
    all_rows: list[dict] = []
    files: dict[str, dict] = {}

    for split in SPLITS:
        for subset in SUBSETS:
            key = (split, subset)
            parent_id_path = parent_dir / f"{split}_{subset}_group_ids.txt"
            subset_pt_path = subset_dir / f"{split}_{subset}.pt"
            parent_ids = read_id_file(parent_id_path)
            parent_id_set = set(parent_ids)
            parent_ids_by_key[key] = parent_ids
            opposite_parent_path = parent_dir / (
                f"{split}_{'test' if subset == 'train' else 'train'}_group_ids.txt"
            )
            opposite_parent_ids = set(read_id_file(opposite_parent_path))
            parent_statuses = Counter(
                str(source_metadata[group_id]["los_status"])
                for group_id in parent_ids
            )
            parent_manifest_checks = check_parent_manifest_summary(
                parent_manifest,
                split,
                subset,
                parent_ids,
                parent_statuses,
            )

            records = load_subset_records(subset_pt_path)
            subset_ids = [str(record["group_id"]) for record in records]
            subset_ids_by_key[key] = subset_ids
            subset_statuses = Counter(str(record["los_status"]) for record in records)
            target = expected_per_status(
                subset, train_per_status, test_per_status
            )
            expected_total = target * 2
            if len(records) != expected_total:
                raise ValueError(
                    f"{split}_{subset} has {len(records)} samples; "
                    f"expected {expected_total}."
                )
            if subset_statuses != Counter({"los": target, "nlos": target}):
                raise ValueError(
                    f"{split}_{subset} status counts are {dict(subset_statuses)}; "
                    f"expected los={target}, nlos={target}."
                )
            unexpected = set(subset_ids) - parent_id_set
            opposite = set(subset_ids) & opposite_parent_ids
            if unexpected:
                raise ValueError(
                    f"{split}_{subset} contains {len(unexpected)} IDs absent from "
                    "its expected parent split."
                )
            if opposite:
                raise ValueError(
                    f"{split}_{subset} contains {len(opposite)} IDs from the "
                    "opposite parent split."
                )

            seed, seed_source = manifest_seed(
                subset_manifest, split, subset, base_seed
            )
            expected_seed = base_seed + DEFAULT_SEED_OFFSETS[key]
            if seed != expected_seed:
                raise ValueError(
                    f"Unexpected seed for {split}_{subset}: got {seed}, "
                    f"expected {expected_seed}."
                )
            replayed_ids = replay_balanced_subset_ids(
                parent_ids,
                source_metadata,
                samples_per_status=target,
                seed=seed,
            )
            replay_exact_order_match = replayed_ids == subset_ids
            if not replay_exact_order_match:
                raise ValueError(
                    f"Exact seeded replay failed for {split}_{subset}."
                )

            parent_rank = {group_id: index for index, group_id in enumerate(parent_ids)}
            for record in records:
                group_id = str(record["group_id"])
                source_record = source_metadata.get(group_id)
                if source_record is None:
                    raise ValueError(
                        f"Subset group_id={group_id!r} is absent from source data."
                    )
                status_matches = (
                    str(source_record["los_status"])
                    == str(record["los_status"])
                )
                if not status_matches:
                    raise ValueError(
                        f"Source/subset status mismatch for group_id={group_id!r}."
                    )
                all_rows.append(
                    {
                        "split": split,
                        "subset": subset,
                        "subset_rank": record["subset_rank"],
                        "parent_rank": parent_rank[group_id],
                        "source_index": source_record["source_index"],
                        "group_id": group_id,
                        "map_id": record["map_id"],
                        "source_id": record["source_id"],
                        "rx_index": record["rx_index"],
                        "los_status": record["los_status"],
                        "in_expected_parent_split": True,
                        "in_opposite_parent_split": False,
                        "source_status_matches_subset": True,
                    }
                )

            exact_id_path = output_dir / f"{split}_{subset}_group_ids.txt"
            exact_id_path.write_text(
                "".join(f"{group_id}\n" for group_id in subset_ids),
                encoding="utf-8",
            )
            file_record = {
                "subset_pt": str(subset_pt_path),
                "parent_id_file": str(parent_id_path),
                "exact_subset_id_file": str(exact_id_path),
                "seed": seed,
                "seed_source": seed_source,
                "sample_count": len(records),
                "los_count": subset_statuses["los"],
                "nlos_count": subset_statuses["nlos"],
                "map_count": len({int(record["map_id"]) for record in records}),
                "maps": sorted({int(record["map_id"]) for record in records}),
                "parent_sample_count": len(parent_ids),
                "parent_los_count": parent_statuses["los"],
                "parent_nlos_count": parent_statuses["nlos"],
                "parent_map_count": len(
                    {parse_group_id(group_id)[0] for group_id in parent_ids}
                ),
                "parent_maps": sorted(
                    {parse_group_id(group_id)[0] for group_id in parent_ids}
                ),
                "parent_group_ids_sha256": sha256_lines(parent_ids),
                "subset_group_ids_sha256": sha256_lines(subset_ids),
                "all_ids_in_expected_parent_split": True,
                "ids_in_opposite_parent_split": 0,
                "all_ids_in_original_source": True,
                "all_source_statuses_match": True,
                "exact_seeded_replay_order_match": replay_exact_order_match,
                "parent_manifest_checks": parent_manifest_checks,
            }
            if hash_pt_files:
                print(f"hashing_subset={subset_pt_path}", flush=True)
                file_record["subset_pt_sha256"] = sha256_file(subset_pt_path)
            files[f"{split}_{subset}"] = file_record
            print(
                f"verified={split}_{subset} samples={len(records)} "
                f"los={subset_statuses['los']} nlos={subset_statuses['nlos']} "
                f"seed={seed} exact_replay=true",
                flush=True,
            )

    coordinate_by_id = None
    block_by_id = None
    if d2los_root is not None:
        from scripts.make_spatial_scenario_splits import (
            load_coordinates,
            spatial_block_keys,
        )

        source_ids = list(source_metadata)
        parsed_source_ids = [parse_group_id(group_id) for group_id in source_ids]
        print(
            f"rescanning_coordinates={d2los_root} block_size_m={block_size_m}",
            flush=True,
        )
        coordinates = load_coordinates(d2los_root, parsed_source_ids)
        blocks = spatial_block_keys(
            parsed_source_ids,
            coordinates,
            block_size_m,
        )
        coordinate_by_id = {
            group_id: (parsed_source_ids[index][0], *coordinates[index])
            for index, group_id in enumerate(source_ids)
        }
        block_by_id = {
            group_id: blocks[index] for index, group_id in enumerate(source_ids)
        }

    split_checks = {}
    for split in SPLITS:
        parent_train_ids = set(parent_ids_by_key[(split, "train")])
        parent_test_ids = set(parent_ids_by_key[(split, "test")])
        subset_train_ids = set(subset_ids_by_key[(split, "train")])
        subset_test_ids = set(subset_ids_by_key[(split, "test")])
        parent_train_maps = {
            parse_group_id(group_id)[0] for group_id in parent_train_ids
        }
        parent_test_maps = {
            parse_group_id(group_id)[0] for group_id in parent_test_ids
        }
        subset_train_maps = {
            parse_group_id(group_id)[0] for group_id in subset_train_ids
        }
        subset_test_maps = {
            parse_group_id(group_id)[0] for group_id in subset_test_ids
        }
        checks = {
            "parent_exact_group_id_overlap_count": len(
                parent_train_ids & parent_test_ids
            ),
            "subset_exact_group_id_overlap_count": len(
                subset_train_ids & subset_test_ids
            ),
            "parent_train_map_count": len(parent_train_maps),
            "parent_test_map_count": len(parent_test_maps),
            "parent_map_overlap_count": len(
                parent_train_maps & parent_test_maps
            ),
            "subset_train_map_count": len(subset_train_maps),
            "subset_test_map_count": len(subset_test_maps),
            "subset_map_overlap_count": len(
                subset_train_maps & subset_test_maps
            ),
            "subset_train_maps_are_parent_train_maps": subset_train_maps
            <= parent_train_maps,
            "subset_test_maps_are_parent_test_maps": subset_test_maps
            <= parent_test_maps,
        }
        if coordinate_by_id is not None and block_by_id is not None:
            parent_train_blocks = {block_by_id[group_id] for group_id in parent_train_ids}
            parent_test_blocks = {block_by_id[group_id] for group_id in parent_test_ids}
            subset_train_blocks = {block_by_id[group_id] for group_id in subset_train_ids}
            subset_test_blocks = {block_by_id[group_id] for group_id in subset_test_ids}
            parent_train_positions = {
                coordinate_by_id[group_id] for group_id in parent_train_ids
            }
            parent_test_positions = {
                coordinate_by_id[group_id] for group_id in parent_test_ids
            }
            subset_train_positions = {
                coordinate_by_id[group_id] for group_id in subset_train_ids
            }
            subset_test_positions = {
                coordinate_by_id[group_id] for group_id in subset_test_ids
            }
            checks.update(
                {
                    "parent_spatial_block_overlap_count": len(
                        parent_train_blocks & parent_test_blocks
                    ),
                    "subset_spatial_block_overlap_count": len(
                        subset_train_blocks & subset_test_blocks
                    ),
                    "parent_exact_ue_position_overlap_count": len(
                        parent_train_positions & parent_test_positions
                    ),
                    "subset_exact_ue_position_overlap_count": len(
                        subset_train_positions & subset_test_positions
                    ),
                    "spatial_coordinate_evidence": "rescanned_from_d2los_propbin",
                }
            )
        else:
            parent_summary = parent_manifest.get("summaries", {}).get(split, {})
            checks.update(
                {
                    "parent_spatial_block_overlap_count": parent_summary.get(
                        "spatial_block_overlap_count",
                        0 if len(parent_train_maps & parent_test_maps) == 0 else None,
                    ),
                    "parent_exact_ue_position_overlap_count": parent_summary.get(
                        "exact_position_overlap_count",
                        0 if len(parent_train_maps & parent_test_maps) == 0 else None,
                    ),
                    "subset_spatial_block_overlap_count": (
                        0 if len(subset_train_maps & subset_test_maps) == 0 else None
                    ),
                    "subset_exact_ue_position_overlap_count": (
                        0 if len(subset_train_maps & subset_test_maps) == 0 else None
                    ),
                    "spatial_coordinate_evidence": (
                        "parent_counts_from_split_manifest; subset zero is implied "
                        "only when map overlap is zero"
                    ),
                }
            )
        if checks["parent_exact_group_id_overlap_count"] != 0:
            raise ValueError(f"Parent {split} train/test IDs overlap.")
        if checks["subset_exact_group_id_overlap_count"] != 0:
            raise ValueError(f"Subset {split} train/test IDs overlap.")
        if split == "scenario_disjoint":
            if checks["parent_map_overlap_count"] != 0:
                raise ValueError("Parent scenario train/test maps overlap.")
            if checks["subset_map_overlap_count"] != 0:
                raise ValueError("Subset scenario train/test maps overlap.")
            for name in (
                "parent_spatial_block_overlap_count",
                "subset_spatial_block_overlap_count",
                "parent_exact_ue_position_overlap_count",
                "subset_exact_ue_position_overlap_count",
            ):
                if checks[name] != 0:
                    raise ValueError(f"Scenario-disjoint check failed: {name}={checks[name]}.")
            checks["spatial_block_overlap_count"] = checks[
                "subset_spatial_block_overlap_count"
            ]
            checks["exact_ue_position_overlap_count"] = checks[
                "subset_exact_ue_position_overlap_count"
            ]
        split_checks[split] = checks

    membership_csv = output_dir / "balanced_subset_membership.csv"
    write_membership_csv(membership_csv, all_rows)
    source_counts = Counter(
        str(record["los_status"]) for record in source_metadata.values()
    )
    experiment_evidence = (
        audit_experiment_results(
            result_root=result_root,
            result_seed=result_seed,
            checkpoint_name=checkpoint_name,
            subset_dir=subset_dir,
            subset_ids_by_key=subset_ids_by_key,
        )
        if result_root is not None
        else None
    )
    payload = {
        "status": "passed",
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sampling_algorithm": (
            "random.Random(seed); sample LoS parent indices; sample NLoS parent "
            "indices; concatenate; shuffle; preserve resulting order"
        ),
        "source_data": str(source_data),
        "source_sample_count": len(source_metadata),
        "source_los_count": source_counts["los"],
        "source_nlos_count": source_counts["nlos"],
        "source_group_ids_sha256_in_source_order": sha256_lines(
            source_metadata.keys()
        ),
        "parent_dir": str(parent_dir),
        "parent_manifest": (
            str(parent_manifest_path) if parent_manifest else None
        ),
        "subset_dir": str(subset_dir),
        "subset_manifest": (
            str(subset_manifest_path) if subset_manifest else None
        ),
        "output_dir": str(output_dir),
        "base_seed": base_seed,
        "train_per_status": train_per_status,
        "test_per_status": test_per_status,
        "d2los_root": str(d2los_root) if d2los_root is not None else None,
        "spatial_block_size_m": block_size_m,
        "membership_csv": str(membership_csv),
        "membership_row_count": len(all_rows),
        "files": files,
        "split_checks": split_checks,
        "experiment_evidence": experiment_evidence,
    }
    output_json = output_dir / "balanced_subset_provenance.json"
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "balanced_subset_provenance_report.md"
    write_provenance_report(report_path, payload)
    print(f"saved_membership_csv={membership_csv}", flush=True)
    print(f"saved_provenance_json={output_json}", flush=True)
    print(f"saved_provenance_report={report_path}", flush=True)
    print("BALANCED_SPLIT_PROVENANCE_AUDIT=passed", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct exact 40k/10k balanced split provenance from the "
            "original 400k source, parent group-ID lists, and materialized subsets."
        )
    )
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-seed", type=int, default=23_421)
    parser.add_argument("--train-per-status", type=int, default=20_000)
    parser.add_argument("--test-per-status", type=int, default=5_000)
    parser.add_argument(
        "--result-root",
        type=Path,
        help=(
            "Optional root containing SPLIT/seed_N checkpoints, train_log.jsonl, "
            "and signal_descriptions.pt. When set, prove which train/test files the "
            "completed experiments actually used."
        ),
    )
    parser.add_argument("--result-seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_epoch_100.pt",
    )
    parser.add_argument(
        "--d2los-root",
        type=Path,
        help=(
            "Optional D2Los_Data root. When set, independently rescan receiver "
            "coordinates and verify spatial-block/exact-position overlap."
        ),
    )
    parser.add_argument("--block-size-m", type=float, default=32.0)
    parser.add_argument(
        "--hash-pt-files",
        action="store_true",
        help="Also SHA-256 hash all six subset .pt files (reads about 10 GB).",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.subset_dir / "provenance"
    build_provenance(
        source_data=args.source_data,
        parent_dir=args.parent_dir,
        subset_dir=args.subset_dir,
        output_dir=output_dir,
        base_seed=args.base_seed,
        train_per_status=args.train_per_status,
        test_per_status=args.test_per_status,
        hash_pt_files=args.hash_pt_files,
        result_root=args.result_root,
        result_seed=args.result_seed,
        checkpoint_name=args.checkpoint_name,
        d2los_root=args.d2los_root,
        block_size_m=args.block_size_m,
    )


if __name__ == "__main__":
    main()
