from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


GROUP_ID_RE = re.compile(r"^.+-map_(\d+)-source_(\d+)-rx_(\d+)$")


def load_ids(path: Path) -> set[tuple[int, int, int]]:
    samples = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    parsed = set()
    for sample in samples:
        group_id = str(getattr(sample, "group_id", ""))
        match = GROUP_ID_RE.fullmatch(group_id)
        if match is None:
            raise ValueError(f"Unsupported group_id={group_id!r} in {path}.")
        parsed.add(tuple(int(value) for value in match.groups()))
    if len(parsed) != len(samples):
        raise ValueError(f"Duplicate group IDs in {path}.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact-link, map, and map/source overlap in an existing split."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    train = load_ids(args.train)
    test = load_ids(args.test)
    train_maps = {item[0] for item in train}
    test_maps = {item[0] for item in test}
    train_sources = {item[:2] for item in train}
    test_sources = {item[:2] for item in test}
    payload = {
        "train_count": len(train),
        "test_count": len(test),
        "exact_group_id_overlap_count": len(train & test),
        "train_map_count": len(train_maps),
        "test_map_count": len(test_maps),
        "map_overlap_count": len(train_maps & test_maps),
        "test_map_overlap_rate": len(train_maps & test_maps) / len(test_maps),
        "train_map_source_count": len(train_sources),
        "test_map_source_count": len(test_sources),
        "map_source_overlap_count": len(train_sources & test_sources),
        "test_map_source_overlap_rate": len(train_sources & test_sources) / len(test_sources),
    }
    for key, value in payload.items():
        print(f"{key}={value}")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
