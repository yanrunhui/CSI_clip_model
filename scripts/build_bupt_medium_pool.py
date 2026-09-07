from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
import re


TIMESTAMP_RE = re.compile(r"_(\d{17})\.[^.]+$")
MODALITY_DIRS = {
    "lidar": Path("LiDAR"),
    "panorama": Path("Panorama"),
    "img_front": Path("Multi-view images/imgf"),
    "img_back": Path("Multi-view images/imgb"),
    "img_left": Path("Multi-view images/imgl"),
    "img_right": Path("Multi-view images/imgr"),
}


@dataclass(frozen=True)
class ArchiveMember:
    timestamp: str
    zip_path: Path
    member_name: str
    size_bytes: int
    compressed_size_bytes: int


def timestamp_from_name(name: str) -> str | None:
    match = TIMESTAMP_RE.search(name)
    return match.group(1) if match else None


def discover_members(zip_paths: list[Path], suffix: str | None = None) -> list[ArchiveMember]:
    members: list[ArchiveMember] = []
    for zip_path in sorted(zip_paths):
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if suffix is not None and not info.filename.lower().endswith(suffix.lower()):
                    continue
                timestamp = timestamp_from_name(info.filename)
                if timestamp is None:
                    continue
                members.append(
                    ArchiveMember(
                        timestamp=timestamp,
                        zip_path=zip_path,
                        member_name=info.filename,
                        size_bytes=info.file_size,
                        compressed_size_bytes=info.compress_size,
                    )
                )
    return members


def modality_timestamps(environment_root: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for name, relative_dir in MODALITY_DIRS.items():
        directory = environment_root / relative_dir
        members = discover_members(list(directory.glob("*.zip")))
        result[name] = {member.timestamp for member in members}
    return result


def evenly_spaced(items: list[ArchiveMember], count: int) -> list[ArchiveMember]:
    if count == 0:
        return []
    if count < 0 or count > len(items):
        raise ValueError(f"Cannot select {count} evenly spaced items from {len(items)} candidates")
    indexes = [((2 * i + 1) * len(items)) // (2 * count) for i in range(count)]
    if len(set(indexes)) != count:
        raise RuntimeError("Internal error: evenly spaced selection produced duplicate indexes")
    return [items[index] for index in indexes]


def select_medium_pool(
    candidates: list[ArchiveMember],
    count: int,
    burst_count: int,
    burst_length: int,
) -> tuple[list[ArchiveMember], dict[str, tuple[str, str]]]:
    burst_total = burst_count * burst_length
    if count <= 0:
        raise ValueError("count must be positive")
    if burst_count < 0 or burst_length < 0:
        raise ValueError("burst_count and burst_length must be non-negative")
    if burst_total > count:
        raise ValueError("burst_count * burst_length cannot exceed count")
    if count > len(candidates):
        raise ValueError(f"Requested {count} files but only {len(candidates)} are eligible")
    if burst_count and len(candidates) // burst_count < burst_length:
        raise ValueError("There are not enough candidates for non-overlapping burst windows")

    annotations: dict[str, tuple[str, str]] = {}
    burst_timestamps: set[str] = set()
    for burst_index in range(burst_count):
        segment_start = burst_index * len(candidates) // burst_count
        segment_end = (burst_index + 1) * len(candidates) // burst_count
        center = (segment_start + segment_end) // 2
        start = max(segment_start, center - burst_length // 2)
        start = min(start, segment_end - burst_length)
        burst_id = f"burst_{burst_index + 1:02d}"
        for member in candidates[start : start + burst_length]:
            burst_timestamps.add(member.timestamp)
            annotations[member.timestamp] = ("continuous_burst", burst_id)

    remaining = [member for member in candidates if member.timestamp not in burst_timestamps]
    uniform = evenly_spaced(remaining, count - len(burst_timestamps))
    for member in uniform:
        annotations[member.timestamp] = ("uniform", "")

    selected_by_timestamp = {
        member.timestamp: member
        for member in candidates
        if member.timestamp in annotations
    }
    selected = sorted(selected_by_timestamp.values(), key=lambda item: item.timestamp)
    if len(selected) != count:
        raise RuntimeError(f"Expected {count} selected files, got {len(selected)}")
    return selected, annotations


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_member(member: ArchiveMember, destination: Path, overwrite: bool) -> str:
    if destination.exists():
        if destination.stat().st_size == member.size_bytes and not overwrite:
            return "existing"
        if not overwrite:
            raise FileExistsError(
                f"Destination exists with an unexpected size: {destination}. "
                "Use --overwrite to replace it."
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(member.zip_path) as archive:
        with archive.open(member.member_name) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if temporary.stat().st_size != member.size_bytes:
        raise IOError(
            f"Extracted size mismatch for {member.member_name}: "
            f"expected {member.size_bytes}, got {temporary.stat().st_size}"
        )
    os.replace(temporary, destination)
    return "extracted"


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def files_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(file for file in path.rglob("*") if file.is_file())


def context_files(source_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for source in files_under(source_root / "TX"):
        pairs.append((source, source.relative_to(source_root)))
    rx_gnss = source_root / "RX/3700MHz/Environment/GNSS"
    for source in files_under(rx_gnss):
        pairs.append((source, source.relative_to(source_root)))
    return pairs


def copy_context_data(
    pairs: list[tuple[Path, Path]],
    output_root: Path,
    overwrite: bool,
) -> None:
    for source, relative in pairs:
        destination = output_root / relative
        if destination.exists() and not overwrite:
            if destination.stat().st_size == source.stat().st_size:
                print(f"context existing: {destination}")
                continue
            raise FileExistsError(
                f"Context destination exists with an unexpected size: {destination}. "
                "Use --overwrite to replace it."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"context copied: {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reproducible BUPT medium pool by extracting complete CIR MAT "
            "members from the original ZIP archives."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--burst-count", type=int, default=5)
    parser.add_argument(
        "--burst-length",
        type=int,
        default=20,
        help="Complete 0.1-second MAT files per continuous burst (20 = about 2 seconds).",
    )
    parser.add_argument(
        "--require-all-modalities",
        action="store_true",
        help="Select only timestamps shared by CIR, LiDAR, panorama, and all four cameras.",
    )
    parser.add_argument(
        "--copy-small-metadata",
        action="store_true",
        help=(
            "Deprecated compatibility flag. TX and RX GNSS are now always copied "
            "into their original directory structure."
        ),
    )
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reserve-gb", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    cir_root = source_root / "RX/3700MHz/Channel/CIR"
    environment_root = source_root / "RX/3700MHz/Environment"
    if not cir_root.is_dir():
        raise SystemExit(f"CIR directory not found: {cir_root}")

    all_members = discover_members(list(cir_root.glob("*.zip")), suffix=".mat")
    timestamps = [member.timestamp for member in all_members]
    if len(timestamps) != len(set(timestamps)):
        raise SystemExit("Duplicate CIR timestamps were found; refusing ambiguous selection")
    all_members.sort(key=lambda member: member.timestamp)

    modalities = modality_timestamps(environment_root)
    eligible = all_members
    if args.require_all_modalities:
        common = set(timestamps)
        for values in modalities.values():
            common &= values
        eligible = [member for member in all_members if member.timestamp in common]

    selected, annotations = select_medium_pool(
        eligible,
        count=args.count,
        burst_count=args.burst_count,
        burst_length=args.burst_length,
    )
    total_bytes = sum(member.size_bytes for member in selected)
    context = context_files(source_root)
    context_bytes = sum(source.stat().st_size for source, _ in context)
    required_bytes = total_bytes + context_bytes
    print(f"Discovered CIR MAT files: {len(all_members)}")
    print(f"Eligible files: {len(eligible)}")
    print(f"Selected files: {len(selected)}")
    print(f"Selected uncompressed size: {total_bytes / 1024**3:.2f} GiB")
    print(
        f"TX plus RX GNSS context: {len(context)} files, "
        f"{context_bytes / 1024**2:.2f} MiB"
    )
    print(f"Estimated pool size: {required_bytes / 1024**3:.2f} GiB")
    print(f"Time range: {selected[0].timestamp} .. {selected[-1].timestamp}")
    counts: dict[str, int] = {}
    for timestamp in annotations:
        selection_type = annotations[timestamp][0]
        counts[selection_type] = counts.get(selection_type, 0) + 1
    print(f"Selection composition: {counts}")
    if args.dry_run:
        print("Dry run only; no files were written.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_root).free
    reserve_bytes = int(args.reserve_gb * 1024**3)
    if free_bytes < required_bytes + reserve_bytes:
        raise SystemExit(
            f"Insufficient free space: need {required_bytes / 1024**3:.2f} GiB plus "
            f"{args.reserve_gb:.2f} GiB reserve, but only {free_bytes / 1024**3:.2f} GiB is free"
        )

    full_index = {member.timestamp: index for index, member in enumerate(all_members)}
    rows: list[dict[str, object]] = []
    for selected_index, member in enumerate(selected):
        destination = (
            output_root
            / "RX/3700MHz/Channel/CIR"
            / Path(member.member_name).name
        )
        action = extract_member(member, destination, overwrite=args.overwrite)
        digest = sha256_file(destination) if args.sha256 else ""
        selection_type, burst_id = annotations[member.timestamp]
        row: dict[str, object] = {
            "sample_id": f"bupt_{selected_index:04d}",
            "timestamp": member.timestamp,
            "selection_type": selection_type,
            "burst_id": burst_id,
            "full_source_index": full_index[member.timestamp],
            "source_zip": str(member.zip_path),
            "source_member": member.member_name,
            "local_mat": str(destination),
            "size_bytes": member.size_bytes,
            "compressed_size_bytes": member.compressed_size_bytes,
            "sha256": digest,
            "copy_status": action,
        }
        for modality_name, modality_values in modalities.items():
            row[f"has_{modality_name}"] = int(member.timestamp in modality_values)
        rows.append(row)
        print(f"[{selected_index + 1:03d}/{len(selected):03d}] {action}: {destination.name}")

    write_manifest(output_root / "manifest.csv", rows)
    copy_context_data(context, output_root, overwrite=args.overwrite)
    print(f"Medium pool ready: {output_root}")
    print(f"Manifest: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, zipfile.BadZipFile, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
