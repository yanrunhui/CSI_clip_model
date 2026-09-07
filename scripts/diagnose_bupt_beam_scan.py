from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_prominences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search BUPT CIR snapshot sequences for a stable repeated scan period. "
            "The test detects periodic structure; it cannot identify beam angles or "
            "prove that the period is caused by phased-array steering."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--subset-manifest", type=Path)
    source.add_argument("--cir-zip-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-mats", type=int, default=10,
        help="Number of MATs to analyze; 0 means every MAT in the source.",
    )
    parser.add_argument(
        "--sample-mode", choices=("uniform", "first"), default="uniform",
        help="How to select max-mats from ZIP archives.",
    )
    parser.add_argument(
        "--temp-dir", type=Path,
        help="Temporary extraction directory; only one MAT is retained at a time.",
    )
    parser.add_argument("--min-period", type=int, default=2)
    parser.add_argument("--max-period", type=int, default=512)
    parser.add_argument("--feature-bins", type=int, default=64)
    parser.add_argument("--chunk-columns", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def read_manifest(path: Path, max_mats: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "local_mat" not in rows[0] or "timestamp" not in rows[0]:
        raise ValueError("Subset manifest must contain timestamp and local_mat columns")
    if max_mats < 0:
        raise ValueError("max-mats cannot be negative")
    return rows if max_mats == 0 else rows[:max_mats]


TIMESTAMP_PATTERN = re.compile(r"_(\d{17})\.mat$", re.IGNORECASE)


def read_zip_sources(directory: Path, max_mats: int, sample_mode: str) -> list[dict[str, str]]:
    if max_mats < 0:
        raise ValueError("max-mats cannot be negative")
    zip_paths = sorted(directory.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No ZIP archives found in {directory}")
    rows: list[dict[str, str]] = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                match = TIMESTAMP_PATTERN.search(Path(member).name)
                if member.lower().endswith(".mat") and match:
                    rows.append({
                        "timestamp": match.group(1),
                        "zip_path": str(zip_path),
                        "zip_member": member,
                    })
    rows.sort(key=lambda item: (item["timestamp"], item["zip_path"], item["zip_member"]))
    if not rows:
        raise ValueError(f"No timestamped MAT members found under {directory}")
    if max_mats == 0 or max_mats >= len(rows):
        return rows
    if sample_mode == "first":
        return rows[:max_mats]
    indices = [((2 * index + 1) * len(rows)) // (2 * max_mats) for index in range(max_mats)]
    return [rows[index] for index in indices]


@contextmanager
def materialized_mat(row: dict[str, str], temp_dir: Path | None):
    if "local_mat" in row:
        path = Path(row["local_mat"])
        if not path.is_file():
            raise FileNotFoundError(path)
        yield path, str(path)
        return

    zip_path = Path(row["zip_path"])
    member = row["zip_member"]
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="bupt_cir_", suffix=".mat", dir=temp_dir)
    os.close(descriptor)
    temporary_path = Path(name)
    try:
        with zipfile.ZipFile(zip_path) as archive, archive.open(member) as source:
            with temporary_path.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        yield temporary_path, f"{zip_path}::{member}"
    finally:
        temporary_path.unlink(missing_ok=True)


def as_complex(array: np.ndarray) -> np.ndarray:
    if array.dtype.fields and {"real", "imag"} <= set(array.dtype.fields):
        return array["real"] + 1j * array["imag"]
    if np.iscomplexobj(array):
        return array
    raise TypeError(f"Unsupported CIR dtype: {array.dtype}")


def scalar(group: h5py.Group | None, name: str, default: float) -> float:
    if group is None or name not in group:
        return default
    value = np.asarray(group[name]).reshape(-1)
    return float(value[0]) if value.size else default


def extract_sequences(
    path: Path,
    feature_bins: int,
    chunk_columns: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        dataset = handle["/CIR/data"]
        if dataset.ndim != 2:
            raise ValueError(f"Expected 2-D /CIR/data in {path}, got {dataset.shape}")
        n_taps, n_snapshots = dataset.shape
        if not 2 <= feature_bins <= n_taps:
            raise ValueError(f"feature-bins must be in [2, {n_taps}]")
        edges = np.linspace(0, n_taps, feature_bins + 1, dtype=np.int64)
        total_power = np.empty(n_snapshots, dtype=np.float64)
        profile_features = np.empty((n_snapshots, feature_bins), dtype=np.float64)
        for start in range(0, n_snapshots, chunk_columns):
            stop = min(start + chunk_columns, n_snapshots)
            block = as_complex(np.asarray(dataset[:, start:stop]))
            power = np.square(np.abs(block), dtype=np.float64)
            total_power[start:stop] = power.sum(axis=0)
            binned = np.add.reduceat(power, edges[:-1], axis=0)[:feature_bins]
            profile_features[start:stop] = binned.T
        info = handle.get("/CIR/info")
        time_gap_seconds = scalar(info, "CIRTimeGap", 2.047e-5)

    epsilon = max(float(np.median(profile_features)) * 1e-12, np.finfo(np.float64).tiny)
    log_profiles = np.log(profile_features + epsilon)
    log_profiles -= log_profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(log_profiles, axis=1, keepdims=True)
    normalized_profiles = log_profiles / np.maximum(norms, 1e-12)

    log_total = 10.0 * np.log10(np.maximum(total_power, np.finfo(np.float64).tiny))
    return normalized_profiles, log_total, time_gap_seconds


def lag_similarity(features: np.ndarray, max_period: int) -> np.ndarray:
    return np.asarray(
        [
            float(np.einsum("ij,ij->", features[:-lag], features[lag:]))
            / (len(features) - lag)
            for lag in range(1, max_period + 1)
        ],
        dtype=np.float64,
    )


def lag_autocorrelation(values: np.ndarray, max_period: int) -> np.ndarray:
    trend_window = min(len(values) // 2 * 2 - 1, max(5, 2 * max_period + 1))
    if trend_window % 2 == 0:
        trend_window -= 1
    residual = values - uniform_filter1d(values, size=trend_window, mode="nearest")
    residual -= residual.mean()
    scale = residual.std()
    if scale <= 0.0:
        return np.zeros(max_period, dtype=np.float64)
    residual /= scale
    return np.asarray(
        [float(np.mean(residual[:-lag] * residual[lag:])) for lag in range(1, max_period + 1)],
        dtype=np.float64,
    )


def prominences(values: np.ndarray, min_period: int) -> np.ndarray:
    result = np.zeros_like(values)
    start = max(min_period - 1, 0)
    peaks, _ = find_peaks(values[start:])
    peaks = peaks + start
    if peaks.size:
        result[peaks] = peak_prominences(values, peaks)[0]
    return result


def robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, 1e-9)


def ranked_periods(
    shape_similarity: np.ndarray,
    power_correlation: np.ndarray,
    min_period: int,
    top_k: int,
) -> list[dict[str, float | int]]:
    shape_prominence = prominences(shape_similarity, min_period=min_period)
    power_prominence = prominences(power_correlation, min_period=min_period)
    shape_scale = robust_scale(shape_prominence[shape_prominence > 0.0]) if np.any(shape_prominence > 0.0) else 1.0
    power_scale = robust_scale(power_prominence[power_prominence > 0.0]) if np.any(power_prominence > 0.0) else 1.0
    combined = shape_prominence / shape_scale + power_prominence / power_scale
    combined[: max(min_period - 1, 0)] = -np.inf

    order = np.argsort(combined)[::-1]
    rows: list[dict[str, float | int]] = []
    for index in order:
        if len(rows) >= top_k or not math.isfinite(float(combined[index])):
            break
        lag = int(index + 1)
        multiple_support = 0.0
        for multiplier in (2, 3, 4):
            multiple_index = multiplier * lag - 1
            if multiple_index < len(combined):
                multiple_support += max(float(combined[multiple_index]), 0.0) / multiplier
        rows.append(
            {
                "rank": len(rows) + 1,
                "period_snapshots": lag,
                "combined_prominence_score": float(combined[index]),
                "multiple_support_score": multiple_support,
                "shape_similarity": float(shape_similarity[index]),
                "shape_prominence": float(shape_prominence[index]),
                "power_autocorrelation": float(power_correlation[index]),
                "power_prominence": float(power_prominence[index]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.subset_manifest is not None:
        rows = read_manifest(args.subset_manifest, max_mats=args.max_mats)
    else:
        rows = read_zip_sources(args.cir_zip_dir, args.max_mats, args.sample_mode)
    if args.min_period < 2 or args.max_period < args.min_period:
        raise ValueError("Require 2 <= min-period <= max-period")
    if args.top_k <= 0 or args.chunk_columns <= 0:
        raise ValueError("top-k and chunk-columns must be positive")

    shape_curves: list[np.ndarray] = []
    power_curves: list[np.ndarray] = []
    per_mat_rows: list[dict[str, object]] = []
    time_gaps: list[float] = []
    snapshot_counts: list[int] = []
    local_top_periods: list[set[int]] = []

    for index, row in enumerate(rows):
        with materialized_mat(row, args.temp_dir) as (path, source_label):
            features, total_power_db, time_gap_seconds = extract_sequences(
                path,
                feature_bins=args.feature_bins,
                chunk_columns=args.chunk_columns,
            )
        max_period = min(args.max_period, len(features) // 3)
        shape = lag_similarity(features, max_period=max_period)
        power = lag_autocorrelation(total_power_db, max_period=max_period)
        shape_curves.append(shape)
        power_curves.append(power)
        time_gaps.append(time_gap_seconds)
        snapshot_counts.append(len(features))
        ranking = ranked_periods(shape, power, args.min_period, min(args.top_k, 10))
        local_top_periods.append({int(item["period_snapshots"]) for item in ranking[:5]})
        for candidate in ranking:
            per_mat_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "source_mat": source_label,
                    "snapshot_count": len(features),
                    "cir_time_gap_seconds": time_gap_seconds,
                    "period_microseconds": float(candidate["period_snapshots"])
                    * time_gap_seconds
                    * 1e6,
                    "cycles_per_mat": len(features) / float(candidate["period_snapshots"]),
                    **candidate,
                }
            )
        print(f"[{index + 1:04d}/{len(rows):04d}] analyzed {Path(source_label).name}")

    common_length = min(len(curve) for curve in shape_curves)
    aggregate_shape = np.median(np.stack([curve[:common_length] for curve in shape_curves]), axis=0)
    aggregate_power = np.median(np.stack([curve[:common_length] for curve in power_curves]), axis=0)
    aggregate_ranking = ranked_periods(
        aggregate_shape,
        aggregate_power,
        min_period=args.min_period,
        top_k=args.top_k,
    )
    median_gap = float(np.median(time_gaps))
    median_snapshots = float(np.median(snapshot_counts))
    for candidate in aggregate_ranking:
        period = int(candidate["period_snapshots"])
        candidate["period_microseconds"] = period * median_gap * 1e6
        candidate["cycles_per_mat"] = median_snapshots / period
        candidate["mat_top5_support"] = sum(
            any(abs(period - local_period) <= 1 for local_period in periods)
            for periods in local_top_periods
        )
        candidate["mat_count"] = len(rows)

    lag_rows = [
        {
            "lag_snapshots": lag,
            "period_microseconds": lag * median_gap * 1e6,
            "median_shape_similarity": float(aggregate_shape[lag - 1]),
            "median_power_autocorrelation": float(aggregate_power[lag - 1]),
        }
        for lag in range(1, common_length + 1)
    ]

    best = aggregate_ranking[0]
    support_fraction = float(best["mat_top5_support"]) / len(rows)
    conclusion = (
        "stable_periodic_structure_detected"
        if support_fraction >= 0.6
        else "no_stable_cross_mat_period_detected"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "aggregate_period_candidates.csv", aggregate_ranking)
    write_csv(args.output_dir / "per_mat_period_candidates.csv", per_mat_rows)
    write_csv(args.output_dir / "lag_curves.csv", lag_rows)
    with (args.output_dir / "beam_scan_diagnosis.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "conclusion": conclusion,
                "warning": (
                    "Periodic CIR structure is evidence of repeated acquisition states, "
                    "not proof of phased-array beam steering. Beam codebook and acquisition "
                    "control metadata are still required."
                ),
                "mat_count": len(rows),
                "median_snapshot_count": median_snapshots,
                "median_cir_time_gap_seconds": median_gap,
                "best_candidate": best,
                "parameters": {
                    "source_mode": "manifest" if args.subset_manifest is not None else "zip",
                    "sample_mode": args.sample_mode,
                    "min_period": args.min_period,
                    "max_period": args.max_period,
                    "feature_bins": args.feature_bins,
                },
            },
            stream,
            indent=2,
        )

    print(f"Conclusion: {conclusion}")
    print(f"Best candidate period: {best['period_snapshots']} snapshots")
    print(f"Period duration: {best['period_microseconds']:.3f} us")
    print(
        f"Cross-MAT top-5 support: {best['mat_top5_support']}/{len(rows)} "
        f"({support_fraction:.1%})"
    )
    print(f"Output directory: {args.output_dir.resolve()}")
    print("A stable period alone does not identify beam angles; obtain the scan codebook to confirm.")


if __name__ == "__main__":
    main()
