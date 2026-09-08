from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from diagnose_bupt_beam_scan import (
    extract_sequences,
    materialized_mat,
    read_manifest,
    read_zip_sources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test 7x8/8x7 scan layouts and associate strongest scan state with GNSS bearing."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--subset-manifest", type=Path)
    source.add_argument("--cir-zip-dir", type=Path)
    parser.add_argument("--rx-gnss", type=Path, required=True)
    parser.add_argument("--tx-gnss", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--period", type=int, default=56)
    parser.add_argument("--max-mats", type=int, default=60)
    parser.add_argument("--sample-mode", choices=("uniform", "first"), default="uniform")
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--feature-bins", type=int, default=64)
    parser.add_argument("--utc-offset-hours", type=float, default=8.0)
    return parser.parse_args()


def read_tx_position(path: Path) -> tuple[float, float, float]:
    try:
        frame = pd.read_excel(path)
    except Exception:
        frame = pd.read_csv(path)
    required = {"Lat_Deg", "Lon_Deg", "Alt_M"}
    if not required <= set(frame.columns):
        raise ValueError(f"TX GNSS lacks columns {sorted(required)}")
    return tuple(float(frame[name].median()) for name in ("Lat_Deg", "Lon_Deg", "Alt_M"))


def read_rx_gnss(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"UTC_Seconds", "Lat_Deg", "Lon_Deg", "Alt_M", "Heading"}
    if not required <= set(frame.columns):
        raise ValueError(f"RX GNSS lacks columns {sorted(required)}")
    return frame.sort_values("UTC_Seconds").reset_index(drop=True)


def timestamp_to_utc_seconds(timestamp: str, utc_offset_hours: float) -> float:
    hh = int(timestamp[8:10])
    mm = int(timestamp[10:12])
    ss = int(timestamp[12:14])
    fraction = int(timestamp[14:17]) / 1000.0
    return (hh * 3600 + mm * 60 + ss + fraction - utc_offset_hours * 3600) % 86400


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta = math.radians(lon2 - lon1)
    y = math.sin(delta) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta)
    return math.degrees(math.atan2(y, x)) % 360.0


def wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def additive_grid_r2(values: np.ndarray, rows: int, columns: int) -> float:
    grid = values.reshape(rows, columns)
    fitted = grid.mean(axis=1, keepdims=True) + grid.mean(axis=0, keepdims=True) - grid.mean()
    total = float(np.square(grid - grid.mean()).sum())
    residual = float(np.square(grid - fitted).sum())
    return 1.0 - residual / total if total > 0.0 else 0.0


def circular_order_score(states: np.ndarray, bearings_deg: np.ndarray, period: int) -> tuple[float, int]:
    state_phase = 2.0 * np.pi * states / period
    bearing_phase = np.deg2rad(bearings_deg)
    positive = abs(np.mean(np.exp(1j * (bearing_phase - state_phase))))
    negative = abs(np.mean(np.exp(1j * (bearing_phase + state_phase))))
    return (float(positive), 1) if positive >= negative else (float(negative), -1)


def main() -> None:
    args = parse_args()
    if args.period != 56:
        raise ValueError("This layout comparison currently requires period=56")
    rows = (
        read_manifest(args.subset_manifest, args.max_mats)
        if args.subset_manifest is not None
        else read_zip_sources(args.cir_zip_dir, args.max_mats, args.sample_mode)
    )
    rx = read_rx_gnss(args.rx_gnss)
    tx_lat, tx_lon, tx_alt = read_tx_position(args.tx_gnss)
    records: list[dict[str, object]] = []
    states_rows: list[dict[str, object]] = []

    for index, row in enumerate(rows):
        with materialized_mat(row, args.temp_dir) as (path, source_label):
            features, power_db, _ = extract_sequences(path, args.feature_bins, 128)
            with h5py.File(path, "r") as handle:
                global_index = np.asarray(handle["/CIR/time/globalBlockIndex"]).reshape(-1).astype(np.int64)
        states = np.mod(global_index, args.period)
        state_power = np.asarray([power_db[states == state].mean() for state in range(args.period)])
        counts = np.asarray([(states == state).sum() for state in range(args.period)])
        prototypes = np.stack([features[states == state].mean(axis=0) for state in range(args.period)])
        prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
        sim7 = float(np.einsum("ij,ij->", prototypes[:-7], prototypes[7:]) / (args.period - 7))
        sim8 = float(np.einsum("ij,ij->", prototypes[:-8], prototypes[8:]) / (args.period - 8))

        utc_seconds = timestamp_to_utc_seconds(row["timestamp"], args.utc_offset_hours)
        nearest = rx.iloc[int(np.abs(rx["UTC_Seconds"].to_numpy() - utc_seconds).argmin())]
        rx_to_tx = bearing_deg(float(nearest.Lat_Deg), float(nearest.Lon_Deg), tx_lat, tx_lon)
        tx_to_rx = bearing_deg(tx_lat, tx_lon, float(nearest.Lat_Deg), float(nearest.Lon_Deg))
        strongest = int(np.argmax(state_power))
        record = {
            "timestamp": row["timestamp"],
            "source_mat": source_label,
            "utc_seconds": utc_seconds,
            "gnss_time_error_s": abs(float(nearest.UTC_Seconds) - utc_seconds),
            "rx_lat_deg": float(nearest.Lat_Deg),
            "rx_lon_deg": float(nearest.Lon_Deg),
            "rx_heading_deg": float(nearest.Heading),
            "bearing_rx_to_tx_deg": rx_to_tx,
            "bearing_tx_to_rx_deg": tx_to_rx,
            "rx_relative_bearing_deg": wrap180(rx_to_tx - float(nearest.Heading)),
            "strongest_state": strongest,
            "state_power_range_db": float(state_power.max() - state_power.min()),
            "grid_7x8_additive_r2": additive_grid_r2(state_power, 7, 8),
            "grid_8x7_additive_r2": additive_grid_r2(state_power, 8, 7),
            "pdp_similarity_lag7": sim7,
            "pdp_similarity_lag8": sim8,
        }
        records.append(record)
        for state in range(args.period):
            states_rows.append({
                "timestamp": row["timestamp"],
                "state_index": state,
                "sample_count": int(counts[state]),
                "mean_power_db": float(state_power[state]),
                "is_strongest": state == strongest,
                "rx_relative_bearing_deg": record["rx_relative_bearing_deg"],
            })
        print(f"[{index + 1:03d}/{len(rows):03d}] {Path(source_label).name}", flush=True)

    result = pd.DataFrame(records)
    state_result = pd.DataFrame(states_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "scan_layout_gnss_per_mat.csv", index=False)
    state_result.to_csv(args.output_dir / "scan_layout_gnss_per_state.csv", index=False)

    score, direction = circular_order_score(
        result["strongest_state"].to_numpy(float),
        result["rx_relative_bearing_deg"].to_numpy(float),
        args.period,
    )
    bearing_unwrapped = np.unwrap(np.deg2rad(result["rx_relative_bearing_deg"].to_numpy(float)))
    bearing_span = float(np.rad2deg(bearing_unwrapped.max() - bearing_unwrapped.min()))
    summary = {
        "mat_count": len(result),
        "period": args.period,
        "median_gnss_time_error_s": float(result.gnss_time_error_s.median()),
        "relative_bearing_span_deg": bearing_span,
        "unique_strongest_states": int(result.strongest_state.nunique()),
        "median_state_power_range_db": float(result.state_power_range_db.median()),
        "median_grid_7x8_additive_r2": float(result.grid_7x8_additive_r2.median()),
        "median_grid_8x7_additive_r2": float(result.grid_8x7_additive_r2.median()),
        "median_pdp_similarity_lag7": float(result.pdp_similarity_lag7.median()),
        "median_pdp_similarity_lag8": float(result.pdp_similarity_lag8.median()),
        "strongest_state_bearing_circular_score": score,
        "state_order_direction": direction,
        "warning": (
            "These are blind structural tests. A high score supports an ordered beam scan, "
            "but physical angles still require antenna heading calibration or a beam codebook."
        ),
    }
    with (args.output_dir / "scan_layout_gnss_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
