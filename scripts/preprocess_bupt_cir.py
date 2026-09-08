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
        np.arange(target_nf, dtype=np.float64) - (target_nf - 1) / 2.0
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


def shared_soft_path_mask(
    cir: np.ndarray,
    noise_tail_fraction: float,
    peak_drop_db: float,
    noise_margin_db: float,
) -> tuple[np.ndarray, dict[str, float | np.ndarray]]:
    """Build one delay-domain amplitude mask shared by all selected snapshots.

    The mask is zero at/below the tail-estimated noise floor, one at/above the
    more conservative of peak-minus-drop and noise-plus-margin, and follows a
    smoothstep curve between those two power levels.  Applying one mask to all
    snapshots avoids making a real weak path blink on and off independently.
    """
    if cir.ndim != 2:
        raise ValueError(f"Expected [snapshots, taps] CIR, got {cir.shape}")
    shared_pdp = np.mean(np.square(np.abs(cir), dtype=np.float64), axis=0)
    tail_count = max(1, int(round(shared_pdp.size * noise_tail_fraction)))
    noise_floor = float(np.median(shared_pdp[-tail_count:]))
    peak_power = float(np.max(shared_pdp))
    peak_relative_threshold = peak_power * (10.0 ** (-peak_drop_db / 10.0))
    noise_relative_threshold = noise_floor * (10.0 ** (noise_margin_db / 10.0))
    threshold = min(
        peak_power,
        max(peak_relative_threshold, noise_relative_threshold),
    )

    mask = np.zeros_like(shared_pdp, dtype=np.float64)
    if peak_power > 0.0 and threshold > noise_floor:
        tiny = np.finfo(np.float64).tiny
        power_db = 10.0 * np.log10(np.maximum(shared_pdp, tiny))
        lower_db = 10.0 * np.log10(max(noise_floor, tiny))
        upper_db = 10.0 * np.log10(max(threshold, tiny))
        position = np.clip((power_db - lower_db) / (upper_db - lower_db), 0.0, 1.0)
        mask = position * position * (3.0 - 2.0 * position)
        mask[shared_pdp <= noise_floor] = 0.0
        mask[shared_pdp >= threshold] = 1.0
    elif peak_power > 0.0:
        mask[shared_pdp >= threshold] = 1.0

    filtered_cir = cir * mask.astype(np.float32)[None, :]
    input_energy = float(np.sum(np.square(np.abs(cir), dtype=np.float64)))
    output_energy = float(np.sum(np.square(np.abs(filtered_cir), dtype=np.float64)))
    retained_energy_fraction = output_energy / input_energy if input_energy > 0.0 else 0.0
    diagnostics: dict[str, float | np.ndarray] = {
        "shared_snapshot_pdp": shared_pdp.astype(np.float32),
        "shared_soft_mask": mask.astype(np.float32),
        "shared_mask_noise_floor": noise_floor,
        "shared_mask_peak_power": peak_power,
        "shared_mask_peak_relative_threshold": peak_relative_threshold,
        "shared_mask_noise_relative_threshold": noise_relative_threshold,
        "shared_mask_threshold": threshold,
        "shared_mask_retained_energy_fraction": retained_energy_fraction,
        "shared_mask_nonzero_tap_fraction": float(np.mean(mask > 0.0)),
        "shared_mask_full_weight_tap_fraction": float(np.mean(mask >= 1.0)),
    }
    return filtered_cir.astype(np.complex64), diagnostics


def native_siso_tokens(cfr: np.ndarray) -> np.ndarray:
    """Represent one physical link as [snapshots, 1, real/imag, frequency]."""
    return np.stack([cfr.real, cfr.imag], axis=1)[:, None].astype(np.float32)


def legacy_upa_1x1_tokens(cfr: np.ndarray) -> np.ndarray:
    """Represent one physical link in the legacy zero-padded 2x2 token."""
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
    token_layout: str,
    cir_denoising: str,
    support_peak_drop_db: float,
    support_noise_margin_db: float,
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

    denoising_diagnostics: dict[str, float | np.ndarray] = {}
    if cir_denoising == "shared_soft_mask":
        cir_for_fft, denoising_diagnostics = shared_soft_path_mask(
            cir,
            noise_tail_fraction=noise_tail_fraction,
            peak_drop_db=support_peak_drop_db,
            noise_margin_db=support_noise_margin_db,
        )
    elif cir_denoising == "none":
        cir_for_fft = cir
    else:
        raise ValueError(f"Unsupported CIR denoising mode: {cir_denoising}")

    sample_rate_hz = info["sampleRateMHz"] * 1e6
    cfr_raw, frequency_indexes, selected_frequency_hz = frequency_response(
        cir_for_fft,
        target_nf=target_nf,
        sample_rate_hz=sample_rate_hz,
        output_bandwidth_hz=output_bandwidth_hz,
    )
    cfr_aligned = phase_align(cfr_raw)
    if token_layout == "native_siso":
        tokens = native_siso_tokens(cfr_aligned)
        config_key = "ULA-1"
    elif token_layout == "legacy_upa_2x2_padded":
        tokens = legacy_upa_1x1_tokens(cfr_aligned)
        config_key = "UPA-1x1"
    else:
        raise ValueError(f"Unsupported token layout: {token_layout}")
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
            config_key=np.asarray(config_key),
            token_layout=np.asarray(token_layout),
            cir_denoising=np.asarray(cir_denoising),
            support_peak_drop_db=np.asarray(support_peak_drop_db, dtype=np.float64),
            support_noise_margin_db=np.asarray(support_noise_margin_db, dtype=np.float64),
            source_mat=np.asarray(str(mat_path)),
            **{
                name: np.asarray(value)
                for name, value in denoising_diagnostics.items()
            },
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
        "token_layout": token_layout,
        "d_token": int(tokens.shape[2]),
        "cir_denoising": cir_denoising,
        "shared_mask_retained_energy_fraction": float(
            denoising_diagnostics.get("shared_mask_retained_energy_fraction", 1.0)
        ),
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
        description="Convert selected BUPT MATLAB 7.3 CIR files into compact NPZ shards."
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
    parser.add_argument(
        "--cir-denoising",
        choices=("none", "shared_soft_mask"),
        default="none",
        help=(
            "Optional delay-domain filtering before FFT. shared_soft_mask builds one "
            "soft support mask from the selected snapshots' mean PDP."
        ),
    )
    parser.add_argument(
        "--support-peak-drop-db",
        type=float,
        default=25.0,
        help="Shared-mask threshold component in dB below the mean-PDP peak.",
    )
    parser.add_argument(
        "--support-noise-margin-db",
        type=float,
        default=6.0,
        help="Shared-mask threshold component in dB above the tail noise floor.",
    )
    parser.add_argument(
        "--token-layout",
        choices=("native_siso", "legacy_upa_2x2_padded"),
        default="legacy_upa_2x2_padded",
        help=(
            "native_siso stores [snapshots,1,2,Nf] without virtual antennas; "
            "legacy_upa_2x2_padded preserves compatibility with old d_token=8 checkpoints."
        ),
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.noise_tail_fraction <= 1.0:
        raise ValueError("noise-tail-fraction must be in (0, 1]")
    if args.support_peak_drop_db <= 0.0:
        raise ValueError("support-peak-drop-db must be positive")
    if args.support_noise_margin_db < 0.0:
        raise ValueError("support-noise-margin-db must be non-negative")
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
                token_layout=args.token_layout,
                cir_denoising=args.cir_denoising,
                support_peak_drop_db=args.support_peak_drop_db,
                support_noise_margin_db=args.support_noise_margin_db,
            )
            print(f"[{index + 1:03d}/{len(rows):03d}] processed: {output_path.name}")
        processed_rows.append({**row, **result})

    write_manifest(args.output_dir / "processed_manifest.csv", processed_rows)
    print(f"Processed files: {len(processed_rows)}")
    print(f"Output directory: {args.output_dir.resolve()}")
    if args.token_layout == "native_siso":
        print("Token layout: native SISO [snapshots,1,2,Nf], without virtual-array padding.")
    else:
        print(
            "Token layout: legacy zero-padded UPA-1x1 (d_token=8). "
            "This is only for compatibility with old array checkpoints."
        )
    print(f"CIR denoising: {args.cir_denoising}")


if __name__ == "__main__":
    main()
