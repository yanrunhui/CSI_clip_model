from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    required = {"timestamp", "local_mat"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    timestamps = [row["timestamp"] for row in rows]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError("Manifest contains duplicate timestamps")
    return sorted(rows, key=lambda row: row["timestamp"])


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or count > len(rows):
        raise ValueError(f"Cannot select {count} rows from {len(rows)} candidates")
    indexes = [((2 * i + 1) * len(rows)) // (2 * count) for i in range(count)]
    if len(set(indexes)) != count:
        raise RuntimeError("Even selection produced duplicate indexes")
    return [rows[index] for index in indexes]


def write_subset(path: Path, rows: list[dict[str, str]], subset_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subset_name", "subset_rank", *rows[0].keys()]
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows):
            writer.writerow({"subset_name": subset_name, "subset_rank": rank, **row})
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic nested BUPT debug/pilot/final subset manifests."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--debug-count", type=int, default=10)
    parser.add_argument("--pilot-count", type=int, default=60)
    parser.add_argument("--final-count", type=int, default=120)
    parser.add_argument(
        "--include-bursts",
        action="store_true",
        help="Use every medium-pool row. By default pointwise subsets use uniform rows only.",
    )
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="Allow manifest rows whose local_mat file has not been copied yet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if not args.allow_missing_files:
        missing = [row["local_mat"] for row in rows if not Path(row["local_mat"]).is_file()]
        if missing:
            examples = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} MAT files listed in the manifest do not exist. Examples:\n{examples}"
            )

    candidates = rows
    if not args.include_bursts and "selection_type" in rows[0]:
        candidates = [row for row in rows if row["selection_type"] == "uniform"]
    if args.final_count > len(candidates):
        raise ValueError(
            f"final-count={args.final_count} exceeds {len(candidates)} eligible rows"
        )
    if not 0 < args.debug_count <= args.pilot_count <= args.final_count:
        raise ValueError("Require 0 < debug-count <= pilot-count <= final-count")

    final_rows = evenly_spaced(candidates, args.final_count)
    pilot_rows = evenly_spaced(final_rows, args.pilot_count)
    debug_rows = evenly_spaced(pilot_rows, args.debug_count)
    output_dir = args.output_dir or args.manifest.parent / "subsets"
    write_subset(output_dir / f"debug_{args.debug_count}.csv", debug_rows, "debug")
    write_subset(output_dir / f"pilot_{args.pilot_count}.csv", pilot_rows, "pilot")
    write_subset(output_dir / f"final_{args.final_count}.csv", final_rows, "final")

    if "selection_type" in rows[0]:
        burst_rows = [row for row in rows if row["selection_type"] == "continuous_burst"]
        if burst_rows:
            write_subset(output_dir / "temporal_bursts.csv", burst_rows, "temporal_bursts")

    print(f"Candidate rows: {len(candidates)}")
    print(f"Nested subsets: {len(debug_rows)} <= {len(pilot_rows)} <= {len(final_rows)}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
