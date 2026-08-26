from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.audit_balanced_split_provenance import (
    DEFAULT_SEED_OFFSETS,
    SPLITS,
    SUBSETS,
    build_provenance,
    replay_balanced_subset_ids,
)


def make_sample(group_id: str, status: str):
    return SimpleNamespace(
        group_id=group_id,
        semantic_key=SimpleNamespace(los_status=status),
    )


def write_ids(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def test_exact_replay_and_end_to_end_provenance(tmp_path: Path) -> None:
    source_path = tmp_path / "source.pt"
    parent_dir = tmp_path / "parent"
    subset_dir = tmp_path / "subset"
    output_dir = tmp_path / "provenance"
    result_root = tmp_path / "results"
    parent_dir.mkdir()
    subset_dir.mkdir()

    samples = []
    for map_id in range(8):
        for source_id in range(2):
            for rx_index in range(8):
                status = "los" if rx_index % 2 == 0 else "nlos"
                samples.append(
                    make_sample(
                        f"D2Los_Data-map_{map_id}-source_{source_id}-rx_{rx_index}",
                        status,
                    )
                )
    torch.save(samples, source_path)
    source_metadata = {
        sample.group_id: {"los_status": sample.semantic_key.los_status}
        for sample in samples
    }

    parent_manifest = {"summaries": {}}
    subset_manifest = {"outputs": {}, "status": "passed"}
    for split in SPLITS:
        if split == "scenario_disjoint":
            train_ids = [
                sample.group_id
                for sample in samples
                if int(sample.group_id.split("-map_")[1].split("-")[0]) < 6
            ]
            test_ids = [
                sample.group_id
                for sample in samples
                if int(sample.group_id.split("-map_")[1].split("-")[0]) >= 6
            ]
        else:
            train_ids = [
                sample.group_id
                for sample in samples
                if "-source_0-" in sample.group_id
            ]
            test_ids = [
                sample.group_id
                for sample in samples
                if "-source_1-" in sample.group_id
            ]
        parent_manifest["summaries"][split] = {}
        for subset, parent_ids in (("train", train_ids), ("test", test_ids)):
            write_ids(parent_dir / f"{split}_{subset}_group_ids.txt", parent_ids)
            target = 2 if subset == "train" else 1
            seed = 23_421 + DEFAULT_SEED_OFFSETS[(split, subset)]
            chosen_ids = replay_balanced_subset_ids(
                parent_ids,
                source_metadata,
                samples_per_status=target,
                seed=seed,
            )
            lookup = {sample.group_id: sample for sample in samples}
            torch.save(
                [lookup[group_id] for group_id in chosen_ids],
                subset_dir / f"{split}_{subset}.pt",
            )
            subset_manifest["outputs"][f"{split}_{subset}"] = {
                "seed": seed,
                "total": target * 2,
                "los": target,
                "nlos": target,
            }

    for split in SPLITS:
        run_dir = result_root / split / "seed_0"
        run_dir.mkdir(parents=True)
        torch.save(
            {
                "epoch": 2,
                "args": {
                    "data_path": str((subset_dir / f"{split}_train.pt").resolve()),
                    "additional_data_paths": [],
                    "epochs": 2,
                    "seed": 0,
                    "batch_size": 2,
                    "min_class_size": 1,
                },
            },
            run_dir / "checkpoint_epoch_2.pt",
        )
        (run_dir / "train_log.jsonl").write_text(
            '\n'.join(
                json.dumps({"epoch": epoch, "steps": 2})
                for epoch in (1, 2)
            )
            + "\n",
            encoding="utf-8",
        )
        test_samples = torch.load(
            subset_dir / f"{split}_test.pt", weights_only=False
        )
        torch.save(
            {
                "comparisons": [
                    {"group_id": sample.group_id} for sample in test_samples
                ]
            },
            run_dir / "signal_descriptions.pt",
        )

    (parent_dir / "split_manifest.json").write_text(
        json.dumps(parent_manifest), encoding="utf-8"
    )
    (subset_dir / "balanced_subset_manifest.json").write_text(
        json.dumps(subset_manifest), encoding="utf-8"
    )

    payload = build_provenance(
        source_data=source_path,
        parent_dir=parent_dir,
        subset_dir=subset_dir,
        output_dir=output_dir,
        base_seed=23_421,
        train_per_status=2,
        test_per_status=1,
        hash_pt_files=True,
        result_root=result_root,
        result_seed=0,
        checkpoint_name="checkpoint_epoch_2.pt",
    )
    assert payload["status"] == "passed"
    assert payload["membership_row_count"] == 18
    assert payload["split_checks"]["scenario_disjoint"]["parent_map_overlap_count"] == 0
    assert payload["split_checks"]["scenario_disjoint"]["subset_map_overlap_count"] == 0
    for split in SPLITS:
        for subset in SUBSETS:
            record = payload["files"][f"{split}_{subset}"]
            assert record["exact_seeded_replay_order_match"] is True
            assert record["all_ids_in_expected_parent_split"] is True
            assert "subset_pt_sha256" in record
        evidence = payload["experiment_evidence"][split]
        assert evidence["training_data_path_exact_match"] is True
        assert evidence["evaluation_exact_test_id_order_match"] is True
        assert evidence["logged_step_values"] == [2]
        assert evidence["training_dataloader_drop_last"] is True
        assert evidence["samples_dropped_per_epoch_after_shuffle"] == 0
