from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import h5py
import numpy as np


INFO_FIELDS = (
    "frequencyMHz",
    "bandwidthMHz",
    "sampleRateMHz",
    "codeRateMHz",
    "pnOrder",
    "pnLength",
    "samplesPerChip",
    "CIRLength",
    "CIRTimeGap",
    "blockDuration",
    "localCIRNum",
    "Attenuator_dB",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Subset manifest is empty: {path}")
    required = {"timestamp", "local_mat"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Subset manifest is missing columns: {sorted(missing)}")
    return rows


def as_complex(array: np.ndarray) -> np.ndarray:
    if array.dtype.fields and {"real", "imag"} <= set(array.dtype.fields):
        return array["real"] + 1j * array["imag"]
    if np.iscomplexobj(array):
        return array
    raise TypeError(f"Unsupported CIR dataset dtype: {array.dtype}")


def scalar(group: h5py.Group, name: str) -> float:
    if name not in group:
        return float("nan")
    value = np.asarray(group[name]).reshape(-1)
    return float(value[0]) if value.size else float("nan")


def selected_snapshot_indexes(count: int, selected_count: int) -> np.ndarray:
    if selected_count <= 0:
        raise ValueError("snapshots-per-mat must be positive")
    if selected_count > count:
        raise ValueError(
            f"snapshots-per-mat={selected_count} exceeds the MAT snapshot count {count}"
        )
    return np.asarray(
        [((2 * i + 1) * count) // (2 * selected_count) for i in range(selected_count)],
        dtype=np.int64,
    )


def mean_pdp(dataset: h5py.Dataset, chunk_columns: int) -> np.ndarray:
    accumulator = np.zeros(dataset.shape[0], dtype=np.float64)
    for start in range(0, dataset.shape[1], chunk_columns):
        stop = min(start + chunk_columns, dataset.shape[1])
        block = as_complex(np.asarray(dataset[:, start:stop]))
        if not np.isfinite(block).all():
            raise ValueError(f"Non-finite CIR values in columns {start}:{stop}")
        accumulator += np.square(np.abs(block), dtype=np.float64).sum(axis=1)
    return accumulator / dataset.shape[1]


def frequency_response(
    cir: np.ndarray,
    target_nf: int,
    sample_rate_hz: float,
    output_bandwidth_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_nf <= 0 or target_nf > cir.shape[1]:
        raise ValueError(f"target-nf must be in [1, {cir.shape[1]}]")
    if not 0.0 < output_bandwidth_hz <= sample_rate_hz:
        raise ValueError(
            "output-bandwidth-hz must be positive and no greater than the CIR sample rate; "
            f"got output={output_bandwidth_hz}, sample_rate={sample_rate_hz}"
        )
    full = np.fft.fftshift(np.fft.fft(cir, axis=1), axes=1)
    source_frequencies = np.fft.fftshift(
        np.fft.fftfreq(cir.shape[1], d=1.0 / sample_rate_hz)
    )
    subcarrier_spacing_hz = output_bandwidth_hz / target_nf
    target_frequencies = (
        np.arange(target_nf, dtype=np.float64) - target_nf / 2.0
    ) * subcarrier_spacing_hz
    insertion = np.searchsorted(source_frequencies, target_frequencies)
    insertion = np.clip(insertion, 1, len(source_frequencies) - 1)
    left = insertion - 1
    right = insertion
    choose_right = (
        np.abs(source_frequencies[right] - target_frequencies)
        < np.abs(source_frequencies[left] - target_frequencies)
    )
    indexes = np.where(choose_right, right, left).astype(np.int64)
    if len(np.unique(indexes)) != target_nf:
        raise ValueError("Frequency selection produced duplicate source FFT bins")
    interpolation_weight = (
        (target_frequencies - source_frequencies[left])
        / (source_frequencies[right] - source_frequencies[left])
    )
    selected = (
        full[:, left] * (1.0 - interpolation_weight[None, :])
        + full[:, right] * interpolation_weight[None, :]
    )
    return (
        selected.astype(np.complex64),
        indexes,
        target_frequencies.astype(np.float64),
    )


def phase_align(cfr: np.ndarray) -> np.ndarray:
    reference = cfr[:, :1]
    rotation = np.exp(-1j * np.angle(reference))
    return (cfr * rotation).astype(np.complex64)


def upa_1x1_tokens(cfr: np.ndarray) -> np.ndarray:
    """Represent one physical link in a zero-padded 2x2 token (d_token=8)."""
    tokens = np.zeros((cfr.shape[0], 1, 8, cfr.shape[1]), dtype=np.float32)
    tokens[:, 0, 0, :] = cfr.real
    tokens[:, 0, 4, :] = cfr.imag
    return tokens


def process_mat(
    mat_path: Path,
    output_path: Path,
    snapshots_per_mat: int,
    target_nf: int,
    output_bandwidth_hz: float,
    chunk_columns: int,
    noise_tail_fraction: float,
    first_path_threshold_db: float,
) -> dict[str, object]:
    with h5py.File(mat_path, "r") as handle:
        if "/CIR/data" not in handle:
            raise KeyError(f"/CIR/data not found in {mat_path}")
        dataset = handle["/CIR/data"]
        if dataset.ndim != 2:
            raise ValueError(f"Expected 2-D /CIR/data, got shape {dataset.shape}")
        snapshot_indexes = selected_snapshot_indexes(dataset.shape[1], snapshots_per_mat)
        cir = as_complex(np.asarray(dataset[:, snapshot_indexes])).T.astype(np.complex64)
        pdp = mean_pdp(dataset, chunk_columns=chunk_columns)
        info_group = handle.get("/CIR/info")
        info = {
            name: scalar(info_group, name) if isinstance(info_group, h5py.Group) else float("nan")
            for name in INFO_FIELDS
        }

    tail_count = max(1, int(round(len(pdp) * noise_tail_fraction)))
    noise_floor = float(np.median(pdp[-tail_count:]))
    clean_pdp = np.maximum(pdp - noise_floor, 0.0)
    threshold = noise_floor * (10.0 ** (first_path_threshold_db / 10.0))
    detected = np.flatnonzero(pdp >= threshold)
    first_path_tap = int(detected[0]) if detected.size else -1
    sample_rate_mhz = info["sampleRateMHz"]
    sample_period_ns = 1000.0 / sample_rate_mhz if sample_rate_mhz > 0 else float("nan")
    first_path_delay_uncalibrated_ns = (
        first_path_tap * sample_period_ns if first_path_tap >= 0 else float("nan")
    )

    sample_rate_hz = info["sampleRateMHz"] * 1e6
    cfr_raw, frequency_indexes, selected_frequency_hz = frequency_response(
        cir,
        target_nf=target_nf,
        sample_rate_hz=sample_rate_hz,
        output_bandwidth_hz=output_bandwidth_hz,
    )
    cfr_aligned = phase_align(cfr_raw)
    tokens = upa_1x1_tokens(cfr_aligned)
    beam_positions = np.zeros((1, 2), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            cfr_raw=cfr_raw,
            cfr_phase_aligned=cfr_aligned,
            tokens=tokens,
            beam_positions=beam_positions,
            snapshot_indexes=snapshot_indexes,
            frequency_indexes=frequency_indexes,
            selected_frequency_hz=selected_frequency_hz,
            output_bandwidth_hz=np.asarray(output_bandwidth_hz, dtype=np.float64),
            output_subcarrier_spacing_hz=np.asarray(
                output_bandwidth_hz / target_nf, dtype=np.float64
            ),
            mean_pdp=pdp.astype(np.float32),
            noise_subtracted_pdp=clean_pdp.astype(np.float32),
            noise_floor=np.asarray(noise_floor, dtype=np.float64),
            first_path_tap=np.asarray(first_path_tap, dtype=np.int64),
            first_path_delay_uncalibrated_ns=np.asarray(
                first_path_delay_uncalibrated_ns, dtype=np.float64
            ),
            config_key=np.asarray("UPA-1x1"),
            source_mat=np.asarray(str(mat_path)),
            **{name: np.asarray(value, dtype=np.float64) for name, value in info.items()},
        )
    os.replace(temporary, output_path)
    return {
        "processed_file": str(output_path),
        "source_taps": int(cir.shape[1]),
        "source_snapshots": int(info.get("localCIRNum", 0)),
        "saved_snapshots": int(snapshots_per_mat),
        "target_nf": int(target_nf),
        "output_bandwidth_hz": float(output_bandwidth_hz),
        "output_subcarrier_spacing_hz": float(output_bandwidth_hz / target_nf),
        "noise_floor": noise_floor,
        "first_path_tap": first_path_tap,
        "first_path_delay_uncalibrated_ns": first_path_delay_uncalibrated_ns,
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No processed rows to write")
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert selected BUPT MATLAB 7.3 CIR files into compact 128-bin NPZ shards."
    )
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshots-per-mat", type=int, default=32)
    parser.add_argument("--target-nf", type=int, default=128)
    parser.add_argument(
        "--output-bandwidth-hz",
        type=float,
        default=100e6,
        help="Centered output bandwidth used by the trained model (default: 100 MHz).",
    )
    parser.add_argument("--chunk-columns", type=int, default=64)
    parser.add_argument("--noise-tail-fraction", type=float, default=0.2)
    parser.add_argument("--first-path-threshold-db", type=float, default=10.0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.noise_tail_fraction <= 1.0:
        raise ValueError("noise-tail-fraction must be in (0, 1]")
    rows = read_rows(args.subset_manifest)
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("max-files must be positive")
        rows = rows[: args.max_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        mat_path = Path(row["local_mat"])
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        output_path = args.output_dir / f"{row['timestamp']}.npz"
        if output_path.exists() and not args.overwrite:
            print(f"[{index + 1:03d}/{len(rows):03d}] existing: {output_path.name}")
            result: dict[str, object] = {"processed_file": str(output_path)}
        else:
            result = process_mat(
                mat_path=mat_path,
                output_path=output_path,
                snapshots_per_mat=args.snapshots_per_mat,
                target_nf=args.target_nf,
                output_bandwidth_hz=args.output_bandwidth_hz,
                chunk_columns=args.chunk_columns,
                noise_tail_fraction=args.noise_tail_fraction,
                first_path_threshold_db=args.first_path_threshold_db,
            )
            print(f"[{index + 1:03d}/{len(rows):03d}] processed: {output_path.name}")
        processed_rows.append({**row, **result})

    write_manifest(args.output_dir / "processed_manifest.csv", processed_rows)
    print(f"Processed files: {len(processed_rows)}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(
        "Note: tokens use a zero-padded UPA-1x1 representation (d_token=8). "
        "This is for single-link smoke testing, not a validated angle representation."
    )


if __name__ == "__main__":
    main()
