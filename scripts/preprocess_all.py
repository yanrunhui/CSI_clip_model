from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.caption import CaptionGenerator
from data.dataset import PreprocessedSample
from data.preprocess import preprocess_sample
from data.semantic_key import SemanticKey

PROP_MAGIC = {b"PORP", b"PROP"}
PROP_HEADER_BYTES = 32
PROP_RX_RECORD_BYTES = 32
PROP_PATH_RECORD_BYTES = 24

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
PROP_PATH_DTYPE = np.dtype(
    [
        ("path_loss_db", "<f2"),
        ("distance_m", "<f2"),
        ("delay_ns", "<f2"),
        ("aod_az_rad", "<f2"),
        ("aod_el_rad", "<f2"),
        ("aoa_az_rad", "<f2"),
        ("aoa_el_rad", "<f2"),
        ("interaction_count", "<u2"),
        ("interaction_type", "<u4"),
        ("interaction_offset", "<u4"),
    ]
)


@dataclass
class LoadedScenarioDataset:
    channel: np.ndarray
    los: np.ndarray
    num_paths: np.ndarray
    delay: np.ndarray
    aoa_az: np.ndarray
    aoa_el: np.ndarray
    aod_az: np.ndarray
    aod_el: np.ndarray
    power: np.ndarray
    phase: np.ndarray
    inter: np.ndarray
    ch_params: dict


@dataclass
class RayVersePropbin:
    path: Path
    version: int
    carrier_frequency_ghz: float
    map_extent_m: float
    rx_records: np.ndarray
    path_records: np.ndarray
    interactions: np.ndarray


def _load_first_mat_array(path: Path) -> np.ndarray:
    data = loadmat(path)
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            return value
    raise ValueError(f"No ndarray payload found in {path}")


def _load_first_npz_array(path: Path) -> np.ndarray:
    with np.load(path) as data:
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray):
                return value
    raise ValueError(f"No ndarray payload found in {path}")


def _load_first_array(path: Path) -> np.ndarray:
    if path.suffix == ".mat":
        return _load_first_mat_array(path)
    if path.suffix == ".npz":
        return _load_first_npz_array(path)
    raise ValueError(f"Unsupported scenario array file: {path}")


def _load_stacked_pattern(
    scenario_dir: Path,
    prefix: str,
    max_rows: int | None = None,
) -> np.ndarray:
    files: list[Path] = []
    for suffix in ("mat", "npz"):
        files = sorted(scenario_dir.glob(f"{prefix}_t*.{suffix}"))
        if not files:
            files = sorted(scenario_dir.glob(f"{prefix}_*.{suffix}"))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"No files matching {prefix}_*.mat/.npz in {scenario_dir}")
    arrays = []
    rows_loaded = 0
    for path in files:
        array = _load_first_array(path)
        arrays.append(array)
        rows_loaded += int(array.shape[0])
        if max_rows is not None and rows_loaded >= max_rows:
            break

    rank = max(array.ndim for array in arrays)
    normalized = []
    for array in arrays:
        expanded = array
        while expanded.ndim < rank:
            expanded = np.expand_dims(expanded, axis=-1)
        normalized.append(expanded)

    max_shape = list(normalized[0].shape)
    for array in normalized[1:]:
        for dim in range(1, rank):
            max_shape[dim] = max(max_shape[dim], array.shape[dim])

    padded = []
    for array in normalized:
        pad_width = [(0, 0)]
        for dim in range(1, rank):
            pad_width.append((0, max_shape[dim] - array.shape[dim]))
        if any(width != (0, 0) for width in pad_width[1:]):
            pad_value = -1 if np.issubdtype(array.dtype, np.integer) else np.nan
            array = np.pad(array, pad_width, mode="constant", constant_values=pad_value)
        padded.append(array)

    stacked = np.concatenate(padded, axis=0)
    if max_rows is not None:
        stacked = stacked[:max_rows]
    return stacked


def _natural_index(path: Path, prefix: str) -> int:
    match = re.search(rf"{re.escape(prefix)}_(\d+)", path.stem)
    if match is None:
        match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match is not None else 0


def _pad_rows(rows: list[np.ndarray], pad_value: float = np.nan) -> np.ndarray:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    max_len = max(int(row.size) for row in rows)
    out = np.full((len(rows), max_len), pad_value, dtype=np.float32)
    for i, row in enumerate(rows):
        flat = np.asarray(row, dtype=np.float32).reshape(-1)
        out[i, : flat.size] = flat
    return out


def _interactions_as_reflection_code(interaction_count: np.ndarray) -> np.ndarray:
    counts = np.asarray(interaction_count, dtype=np.int32).reshape(-1)
    codes = np.zeros(counts.shape, dtype=np.float32)
    for idx, count in enumerate(counts):
        if count <= 0:
            continue
        # Existing semantic code treats decimal digit "1" as one reflection.
        codes[idx] = int("1" * min(int(count), 9))
    return codes


def _recursive_find(mapping, key: str):
    if isinstance(mapping, dict):
        if key in mapping:
            return mapping[key]
        for value in mapping.values():
            found = _recursive_find(value, key)
            if found is not None:
                return found
    elif isinstance(mapping, list):
        for value in mapping:
            found = _recursive_find(value, key)
            if found is not None:
                return found
    return None


def _deepmimo_v4_txrx_set(config: dict, is_tx: bool) -> dict | None:
    txrx_sets = config.get("txrx_sets", {})
    if not isinstance(txrx_sets, dict):
        return None

    for name in sorted(txrx_sets):
        txrx_set = txrx_sets[name]
        if isinstance(txrx_set, dict) and txrx_set.get("is_tx") is is_tx:
            return txrx_set

    target = "tx" if is_tx else "rx"
    for name in sorted(txrx_sets):
        txrx_set = txrx_sets[name]
        if not isinstance(txrx_set, dict):
            continue
        set_name = str(txrx_set.get("name", "")).lower()
        if set_name.startswith(target) or f"{target}_" in set_name:
            return txrx_set
    return None


def _factor_array_shape(num_antennas: int) -> tuple[int, int]:
    if num_antennas <= 1:
        return 1, 1
    root = int(np.sqrt(num_antennas))
    for rows in range(root, 0, -1):
        if num_antennas % rows == 0:
            return rows, num_antennas // rows
    return num_antennas, 1


def _array_shape_from_positions(
    ant_rel_positions: Any,
    num_antennas: int,
) -> tuple[int, int]:
    if num_antennas <= 1:
        return 1, 1

    if ant_rel_positions is not None:
        positions = np.asarray(ant_rel_positions, dtype=float)
        if positions.ndim == 2:
            if positions.shape[1] == num_antennas:
                axes = positions
            elif positions.shape[0] == num_antennas:
                axes = positions.T
            else:
                axes = positions

            counts = []
            for axis in axes:
                unique = np.unique(np.round(axis, decimals=9))
                if unique.size > 1:
                    counts.append(int(unique.size))

            if len(counts) == 1 and counts[0] == num_antennas:
                return counts[0], 1

            pairs = [
                (counts[i], counts[j])
                for i in range(len(counts))
                for j in range(i + 1, len(counts))
                if counts[i] * counts[j] == num_antennas
            ]
            if pairs:
                return min(pairs, key=lambda pair: abs(pair[0] - pair[1]))

            if counts and int(np.prod(counts)) == num_antennas:
                return counts[0], int(np.prod(counts[1:]))

    return _factor_array_shape(num_antennas)


def _raw_params(config: dict) -> dict:
    raw = _recursive_find(config, "raw_params")
    return raw if isinstance(raw, dict) else {}


def _first_scalar(value: Any, default: float | int) -> float | int:
    if value is None:
        return default
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return default
    return array[0].item()


def _array_shape_from_json(config: dict, antenna_key: str, fallback: tuple[int, int]) -> tuple[int, int]:
    antenna_cfg = _recursive_find(config, antenna_key) or {}
    shape = antenna_cfg.get("shape") if isinstance(antenna_cfg, dict) else None
    if shape is not None:
        shape = np.asarray(shape).astype(int).reshape(-1)
        if shape.size == 1:
            return int(shape[0]), 1
        return int(shape[0]), int(shape[1])

    is_tx = "bs" in antenna_key or "tx" in antenna_key
    txrx_set = _deepmimo_v4_txrx_set(config, is_tx=is_tx)
    if txrx_set is not None:
        num_antennas = int(txrx_set.get("num_ant", np.prod(fallback)))
        return _array_shape_from_positions(txrx_set.get("ant_rel_positions"), num_antennas)

    raw = _raw_params(config)
    raw_num_ant_key = "tx_array_num_ant" if is_tx else "rx_array_num_ant"
    raw_pos_key = "tx_array_ant_pos" if is_tx else "rx_array_ant_pos"
    if raw_num_ant_key in raw:
        num_antennas = int(_first_scalar(raw.get(raw_num_ant_key), np.prod(fallback)))
        return _array_shape_from_positions(raw.get(raw_pos_key), num_antennas)

    return fallback


def _spacing_from_json(config: dict, antenna_key: str, default: float = 0.5) -> float:
    antenna_cfg = _recursive_find(config, antenna_key) or {}
    if isinstance(antenna_cfg, dict) and "spacing" in antenna_cfg:
        return float(antenna_cfg["spacing"])
    return default


def _ofdm_from_json(config: dict) -> tuple[float, int, np.ndarray]:
    ofdm = _recursive_find(config, "ofdm") or {}
    raw = _raw_params(config)
    bandwidth = float(_first_scalar(ofdm.get("bandwidth"), _first_scalar(raw.get("bandwidth"), 1e6)))
    subcarriers = int(_first_scalar(ofdm.get("subcarriers"), _first_scalar(raw.get("subcarriers"), 64)))
    selected = _recursive_find(config, "selected_subcarriers")
    if selected is None:
        selected = np.arange(subcarriers)
    selected = np.asarray(selected).astype(int).reshape(-1)
    return bandwidth, subcarriers, selected


def _path_mask(power_dbw: np.ndarray, delay_s: np.ndarray) -> np.ndarray:
    return np.isfinite(power_dbw) & np.isfinite(delay_s)


INTERACTION_DIGITS = {
    "1": "reflection",
    "2": "diffraction",
    "3": "scattering",
}


def _interaction_counts(code: float) -> dict[str, int]:
    if not np.isfinite(code):
        return {"reflection": 0, "diffraction": 0, "scattering": 0}
    int_code = int(code)
    if int_code == 0:
        return {"reflection": 0, "diffraction": 0, "scattering": 0}
    digits = str(abs(int_code))
    return {
        name: sum(1 for digit in digits if INTERACTION_DIGITS.get(digit) == name)
        for name in INTERACTION_DIGITS.values()
    }


def _summed_interaction_counts(inter_code: np.ndarray, valid_mask: np.ndarray) -> dict[str, int]:
    totals = {"reflection": 0, "diffraction": 0, "scattering": 0}
    for code in inter_code[valid_mask]:
        counts = _interaction_counts(float(code))
        for name, count in counts.items():
            totals[name] += count
    return totals


def _infer_los_and_num_paths(inter: np.ndarray, power: np.ndarray, delay: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_users = power.shape[0]
    los = np.full(num_users, -1, dtype=np.int64)
    num_paths = np.zeros(num_users, dtype=np.int64)
    for i in range(num_users):
        mask = _path_mask(power[i], delay[i])
        num_paths[i] = int(mask.sum())
        if num_paths[i] == 0:
            continue
        los[i] = 1 if np.any(inter[i][mask] == 0) else 0
    return los, num_paths


def _steering_vector(rows: int, cols: int, spacing: float, az_deg: float, el_deg: float) -> np.ndarray:
    az = np.deg2rad(float(az_deg))
    el = np.deg2rad(float(el_deg))
    # Inference: use a simple direction-cosine UPA response with half-wavelength-like spacing.
    u = np.cos(el) * np.cos(az)
    v = np.cos(el) * np.sin(az)
    row_idx = np.arange(rows, dtype=np.float32)
    col_idx = np.arange(cols, dtype=np.float32)
    phase = 2.0 * np.pi * spacing * (
        row_idx[:, None] * u + col_idx[None, :] * v
    )
    return np.exp(1j * phase).reshape(rows * cols) / np.sqrt(rows * cols)


def _synthesize_channel(
    power_dbw: np.ndarray,
    phase_deg: np.ndarray,
    delay_s: np.ndarray,
    aoa_az_deg: np.ndarray,
    aoa_el_deg: np.ndarray,
    aod_az_deg: np.ndarray,
    aod_el_deg: np.ndarray,
    rx_shape: tuple[int, int],
    tx_shape: tuple[int, int],
    rx_spacing: float,
    tx_spacing: float,
    bandwidth_hz: float,
    total_subcarriers: int,
    selected_subcarriers: np.ndarray,
) -> np.ndarray:
    n_rx_ant = rx_shape[0] * rx_shape[1]
    n_tx_ant = tx_shape[0] * tx_shape[1]
    n_sc = int(selected_subcarriers.size)
    channel = np.zeros((n_rx_ant, n_tx_ant, n_sc), dtype=np.complex64)
    if n_sc == 0:
        return channel

    sc_spacing = bandwidth_hz / max(total_subcarriers, 1)
    centered_subcarriers = selected_subcarriers - selected_subcarriers.mean()
    freqs = centered_subcarriers * sc_spacing

    mask = _path_mask(power_dbw, delay_s)
    for p in np.where(mask)[0]:
        gain = np.sqrt(10.0 ** (power_dbw[p] / 10.0)) * np.exp(1j * np.deg2rad(phase_deg[p]))
        rx_sv = _steering_vector(rx_shape[0], rx_shape[1], rx_spacing, aoa_az_deg[p], aoa_el_deg[p])
        tx_sv = _steering_vector(tx_shape[0], tx_shape[1], tx_spacing, aod_az_deg[p], aod_el_deg[p])
        spatial = np.outer(rx_sv, np.conj(tx_sv))
        delay_phase = np.exp(-1j * 2.0 * np.pi * freqs * delay_s[p]).astype(np.complex64)
        channel += gain * spatial[:, :, None] * delay_phase[None, None, :]
    return channel


def _carrier_ghz_from_path(path: Path, fallback: float = 3.5) -> float:
    for parent in [path.parent, *path.parents]:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GHz", parent.name, flags=re.IGNORECASE)
        if match is not None:
            return float(match.group(1))
    return fallback


def read_rayverse_propbin(path: Path) -> RayVersePropbin:
    opener = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    with opener as f:
        data = f.read()

    if len(data) < PROP_HEADER_BYTES:
        raise ValueError(f"RayVerse propbin file is too small: {path}")
    if data[:4] not in PROP_MAGIC:
        raise ValueError(f"Unsupported RayVerse propbin magic {data[:4]!r} in {path}")

    version = int(np.frombuffer(data, dtype="<u4", count=1, offset=4)[0])
    map_extent_m = float(np.frombuffer(data, dtype="<f4", count=1, offset=8)[0])
    rx_count = int(np.frombuffer(data, dtype="<u4", count=1, offset=20)[0])
    path_count = int(np.frombuffer(data, dtype="<u4", count=1, offset=24)[0])
    interaction_count = int(np.frombuffer(data, dtype="<u4", count=1, offset=28)[0])

    rx_start = PROP_HEADER_BYTES
    path_start = rx_start + rx_count * PROP_RX_RECORD_BYTES
    interaction_start = path_start + path_count * PROP_PATH_RECORD_BYTES
    expected_bytes = interaction_start + interaction_count * np.dtype("<u2").itemsize
    if len(data) != expected_bytes:
        raise ValueError(
            f"Unexpected RayVerse propbin size for {path}: got {len(data)} bytes, "
            f"expected {expected_bytes}"
        )

    return RayVersePropbin(
        path=path,
        version=version,
        carrier_frequency_ghz=_carrier_ghz_from_path(path),
        map_extent_m=map_extent_m,
        rx_records=np.frombuffer(data, dtype=PROP_RX_DTYPE, count=rx_count, offset=rx_start).copy(),
        path_records=np.frombuffer(
            data,
            dtype=PROP_PATH_DTYPE,
            count=path_count,
            offset=path_start,
        ).copy(),
        interactions=np.frombuffer(
            data,
            dtype="<u2",
            count=interaction_count,
            offset=interaction_start,
        ).copy(),
    )


def _iter_d2los_propbin_files(
    d2los_root: Path,
    max_maps: int | None,
    max_sources_per_map: int | None,
) -> list[Path]:
    map_dirs = sorted(
        [path for path in d2los_root.glob("map_*") if path.is_dir()],
        key=lambda path: _natural_index(path, "map"),
    )
    if max_maps is not None:
        map_dirs = map_dirs[:max_maps]

    propbin_files: list[Path] = []
    for map_dir in map_dirs:
        propbin_dirs = sorted(map_dir.glob("special_points_propbin_*"))
        if not propbin_dirs:
            continue
        source_files = sorted(
            propbin_dirs[0].glob("source_*.propbin*"),
            key=lambda path: _natural_index(path, "source"),
        )
        if max_sources_per_map is not None:
            source_files = source_files[:max_sources_per_map]
        propbin_files.extend(source_files)
    return propbin_files


def is_d2los_root(path: Path) -> bool:
    return path.exists() and (path / "buildings_complete").is_dir() and any(path.glob("map_*"))


def _selected_rx_indices(
    rx_records: np.ndarray,
    max_rx_per_source: int | None,
    remaining_samples: int | None,
) -> np.ndarray:
    indices = np.flatnonzero(rx_records["path_count"] > 0)
    if max_rx_per_source is not None:
        indices = indices[:max_rx_per_source]
    if remaining_samples is not None:
        indices = indices[:remaining_samples]
    return indices


def load_d2los_dataset(
    d2los_root: Path,
    max_samples: int | None,
    max_maps: int | None,
    max_sources_per_map: int | None,
    max_rx_per_source: int | None,
    tx_shape: tuple[int, int],
    bandwidth_hz: float,
    total_subcarriers: int,
    tx_power_dbm: float,
    tx_spacing: float = 0.5,
) -> LoadedScenarioDataset:
    if not d2los_root.exists():
        raise FileNotFoundError(f"D2Los root not found: {d2los_root}")
    if max_samples is None and max_rx_per_source is None:
        raise ValueError(
            "D2Los/RayVerse preprocessing can expand to hundreds of millions of links. "
            "Set --max-samples or --max-rx-per-source for the conversion run."
        )

    propbin_files = _iter_d2los_propbin_files(
        d2los_root=d2los_root,
        max_maps=max_maps,
        max_sources_per_map=max_sources_per_map,
    )
    if not propbin_files:
        raise FileNotFoundError(f"No source_*.propbin(.gz) files found under {d2los_root}")

    selected_subcarriers = np.arange(total_subcarriers, dtype=np.int64)
    rx_shape = (1, 1)
    channels: list[np.ndarray] = []
    delays: list[np.ndarray] = []
    aoa_azs: list[np.ndarray] = []
    aoa_els: list[np.ndarray] = []
    aod_azs: list[np.ndarray] = []
    aod_els: list[np.ndarray] = []
    powers: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    inter_codes: list[np.ndarray] = []
    los_values: list[int] = []
    num_paths: list[int] = []
    skipped_propbin_files = 0

    for propbin_path in propbin_files:
        if max_samples is not None and len(channels) >= max_samples:
            break
        try:
            propbin = read_rayverse_propbin(propbin_path)
        except (EOFError, OSError, ValueError) as exc:
            skipped_propbin_files += 1
            warnings.warn(f"Skipping unreadable RayVerse propbin file {propbin_path}: {exc}")
            continue
        remaining = None if max_samples is None else max_samples - len(channels)
        rx_indices = _selected_rx_indices(
            propbin.rx_records,
            max_rx_per_source=max_rx_per_source,
            remaining_samples=remaining,
        )
        if rx_indices.size == 0:
            continue

        carrier_hz = propbin.carrier_frequency_ghz * 1e9
        wavelength_m = 299_792_458.0 / max(carrier_hz, 1.0)
        for rx_idx in rx_indices:
            rx_record = propbin.rx_records[int(rx_idx)]
            count = int(rx_record["path_count"])
            offset = int(rx_record["path_offset"])
            path_records = propbin.path_records[offset : offset + count]
            path_loss_db = path_records["path_loss_db"].astype(np.float32)
            distance_m = path_records["distance_m"].astype(np.float32)
            delay_s = path_records["delay_ns"].astype(np.float32) * 1e-9
            power_dbw = tx_power_dbm - path_loss_db - 30.0
            phase_deg = np.rad2deg(-2.0 * np.pi * distance_m / wavelength_m).astype(np.float32)
            aoa_az_deg = np.rad2deg(path_records["aoa_az_rad"].astype(np.float32))
            aoa_el_deg = np.rad2deg(path_records["aoa_el_rad"].astype(np.float32))
            aod_az_deg = np.rad2deg(path_records["aod_az_rad"].astype(np.float32))
            aod_el_deg = np.rad2deg(path_records["aod_el_rad"].astype(np.float32))
            interaction_counts = path_records["interaction_count"].astype(np.int32)

            channels.append(
                _synthesize_channel(
                    power_dbw=power_dbw,
                    phase_deg=phase_deg,
                    delay_s=delay_s,
                    aoa_az_deg=aoa_az_deg,
                    aoa_el_deg=aoa_el_deg,
                    aod_az_deg=aod_az_deg,
                    aod_el_deg=aod_el_deg,
                    rx_shape=rx_shape,
                    tx_shape=tx_shape,
                    rx_spacing=tx_spacing,
                    tx_spacing=tx_spacing,
                    bandwidth_hz=bandwidth_hz,
                    total_subcarriers=total_subcarriers,
                    selected_subcarriers=selected_subcarriers,
                )
            )
            delays.append(delay_s)
            aoa_azs.append(aoa_az_deg)
            aoa_els.append(aoa_el_deg)
            aod_azs.append(aod_az_deg)
            aod_els.append(aod_el_deg)
            powers.append(power_dbw)
            phases.append(phase_deg)
            inter_codes.append(_interactions_as_reflection_code(interaction_counts))
            los_values.append(1 if np.any(interaction_counts == 0) else 0)
            num_paths.append(count)

            if max_samples is not None and len(channels) >= max_samples:
                break

    if not channels:
        raise ValueError(
            f"No usable D2Los links found under {d2los_root}; "
            f"skipped {skipped_propbin_files} unreadable propbin files."
        )
    if skipped_propbin_files:
        warnings.warn(f"Skipped {skipped_propbin_files} unreadable RayVerse propbin files.")

    return LoadedScenarioDataset(
        channel=np.stack(channels, axis=0),
        los=np.asarray(los_values, dtype=np.int64),
        num_paths=np.asarray(num_paths, dtype=np.int64),
        delay=_pad_rows(delays),
        aoa_az=_pad_rows(aoa_azs),
        aoa_el=_pad_rows(aoa_els),
        aod_az=_pad_rows(aod_azs),
        aod_el=_pad_rows(aod_els),
        power=_pad_rows(powers),
        phase=_pad_rows(phases),
        inter=_pad_rows(inter_codes, pad_value=-1),
        ch_params={
            "bs_antenna": {"shape": list(tx_shape), "spacing": tx_spacing},
            "ue_antenna": {"shape": list(rx_shape), "spacing": tx_spacing},
            "ofdm": {"bandwidth": bandwidth_hz, "subcarriers": total_subcarriers},
            "selected_subcarriers": selected_subcarriers,
        },
    )


def load_json_scenario_dataset(
    scenario_dir: Path,
    max_samples: int | None = None,
) -> LoadedScenarioDataset:
    params_path = scenario_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"params.json not found in {scenario_dir}")

    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    power = _load_stacked_pattern(scenario_dir, "power", max_rows=max_samples)
    n_rows = int(power.shape[0])
    phase = _load_stacked_pattern(scenario_dir, "phase", max_rows=n_rows)
    delay = _load_stacked_pattern(scenario_dir, "delay", max_rows=n_rows)
    aoa_az = _load_stacked_pattern(scenario_dir, "aoa_az", max_rows=n_rows)
    aoa_el = _load_stacked_pattern(scenario_dir, "aoa_el", max_rows=n_rows)
    aod_az = _load_stacked_pattern(scenario_dir, "aod_az", max_rows=n_rows)
    aod_el = _load_stacked_pattern(scenario_dir, "aod_el", max_rows=n_rows)
    inter = _load_stacked_pattern(scenario_dir, "inter", max_rows=n_rows)

    tx_shape = _array_shape_from_json(params, "bs_antenna", fallback=(8, 8))
    rx_shape = _array_shape_from_json(params, "ue_antenna", fallback=(4, 2))
    tx_spacing = _spacing_from_json(params, "bs_antenna", default=0.5)
    rx_spacing = _spacing_from_json(params, "ue_antenna", default=0.5)
    bandwidth_hz, total_subcarriers, selected_subcarriers = _ofdm_from_json(params)

    channels = np.stack(
        [
            _synthesize_channel(
                power_dbw=power[i].reshape(-1),
                phase_deg=phase[i].reshape(-1),
                delay_s=delay[i].reshape(-1),
                aoa_az_deg=aoa_az[i].reshape(-1),
                aoa_el_deg=aoa_el[i].reshape(-1),
                aod_az_deg=aod_az[i].reshape(-1),
                aod_el_deg=aod_el[i].reshape(-1),
                rx_shape=rx_shape,
                tx_shape=tx_shape,
                rx_spacing=rx_spacing,
                tx_spacing=tx_spacing,
                bandwidth_hz=bandwidth_hz,
                total_subcarriers=total_subcarriers,
                selected_subcarriers=selected_subcarriers,
            )
            for i in range(n_rows)
        ],
        axis=0,
    )
    los, num_paths = _infer_los_and_num_paths(inter, power, delay)

    return LoadedScenarioDataset(
        channel=channels,
        los=los,
        num_paths=num_paths,
        delay=delay,
        aoa_az=aoa_az,
        aoa_el=aoa_el,
        aod_az=aod_az,
        aod_el=aod_el,
        power=power,
        phase=phase,
        inter=inter,
        ch_params={
            "bs_antenna": {"shape": list(tx_shape), "spacing": tx_spacing},
            "ue_antenna": {"shape": list(rx_shape), "spacing": rx_spacing},
            "ofdm": {"bandwidth": bandwidth_hz, "subcarriers": total_subcarriers},
            "selected_subcarriers": selected_subcarriers,
        },
    )


def import_deepmimo_backend():
    try:
        import DeepMIMOv3 as dm  # type: ignore

        return "v3", dm
    except ImportError:
        pass

    try:
        import deepmimo as dm  # type: ignore

        return "v4", dm
    except ImportError as exc:
        raise SystemExit(
            "Neither DeepMIMOv3 nor deepmimo is installed. "
            "Install the backend that matches your environment."
        ) from exc


def load_deepmimo_dataset(
    scenario: str,
    scenario_root: str | None = None,
    max_samples: int | None = None,
):
    if scenario_root is None:
        scenario_dir = ROOT / "Raytracing_scenarios" / scenario
    else:
        scenario_dir = Path(scenario_root) / scenario
    if (scenario_dir / "params.json").exists():
        return load_json_scenario_dataset(scenario_dir, max_samples=max_samples)
    if scenario_root is not None:
        available = []
        root_path = Path(scenario_root)
        if root_path.exists():
            available = sorted(path.name for path in root_path.iterdir() if path.is_dir())
        raise FileNotFoundError(
            f"Local scenario not found or missing params.json: {scenario_dir}. "
            f"Available scenario directories under {root_path}: {available}"
        )

    backend, dm = import_deepmimo_backend()

    if backend == "v4":
        dm.download(scenario)
        dataset = dm.load(scenario)
        dataset.compute_channels()
        return dataset

    if not hasattr(dm, "default_params") or not hasattr(dm, "generate_data"):
        raise SystemExit(
            "DeepMIMOv3 was imported, but the expected v3 API "
            "(`default_params`, `generate_data`) was not found."
        )

    params = dm.default_params()
    params["scenario"] = scenario
    dataset = dm.generate_data(params)

    if isinstance(dataset, (list, tuple)):
        if len(dataset) != 1:
            raise SystemExit(
                "DeepMIMOv3 returned multiple BS entries. This script currently expects "
                "a single dataset object or a single-entry list."
            )
        dataset = dataset[0]

    required_attrs = ("channel", "los", "num_paths", "delay", "aoa_az", "power", "ch_params")
    missing = [name for name in required_attrs if not hasattr(dataset, name)]
    if missing:
        raise SystemExit(
            "DeepMIMOv3 loaded successfully, but the returned object does not match the "
            f"current preprocessing adapter. Missing attributes: {missing}"
        )
    return dataset


def infer_env_type(scenario: str, fallback: str | None = None) -> str:
    if fallback is not None:
        return fallback
    lower = scenario.lower()
    if any(token in lower for token in ("indoor", "office", "mall", "o1", "i1", "i2", "i3")):
        return "indoor"
    if any(token in lower for token in ("o2i", "outdoor-to-indoor")):
        return "O2I"
    return "outdoor"


def compute_delay_spread(delay_s: np.ndarray, power_linear: np.ndarray) -> float:
    if delay_s.size == 0:
        return 0.0
    power_sum = float(power_linear.sum())
    if power_sum <= 0:
        return 0.0
    mean_delay = float((power_linear * delay_s).sum() / power_sum)
    second_moment = float((power_linear * (delay_s - mean_delay) ** 2).sum() / power_sum)
    return max(second_moment, 0.0) ** 0.5


def circular_azimuth_spread_deg(angles_deg: np.ndarray, power_linear: np.ndarray) -> float:
    if angles_deg.size == 0:
        return 0.0
    weights = power_linear / max(float(power_linear.sum()), 1e-12)
    radians = np.deg2rad(angles_deg)
    resultant = np.sum(weights * np.exp(1j * radians))
    magnitude = np.clip(np.abs(resultant), 1e-6, 1.0)
    circ_std_rad = np.sqrt(max(-2.0 * np.log(magnitude), 0.0))
    return float(np.rad2deg(circ_std_rad))


def estimate_k_factor_db(power_dbw: np.ndarray, los_flag: int) -> float:
    if power_dbw.size == 0:
        return -20.0
    power_linear = np.power(10.0, power_dbw / 10.0)
    if los_flag != 1:
        return -20.0
    los_power = float(power_linear[0])
    nlos_power = float(power_linear[1:].sum())
    if los_power <= 0:
        return -20.0
    if nlos_power <= 0:
        return 30.0
    return float(10.0 * np.log10(los_power / nlos_power))


def extract_semantic_observables_from_deepmimo(
    scenario: str,
    env_type: str,
    los_value: int,
    num_paths_value: int,
    delay_s: np.ndarray,
    aoa_az_deg: np.ndarray,
    power_dbw: np.ndarray,
    inter_code: np.ndarray,
) -> dict[str, float | int | str]:
    valid = _path_mask(power_dbw, delay_s)
    n_valid_paths = int(valid.sum())
    first_path_delay = float("nan")
    first_path_power_dbw = float("nan")
    first_path_aoa_az_deg = float("nan")
    interaction_counts = {"reflection": 0, "diffraction": 0}
    if bool(valid.any()):
        valid_indices = np.where(valid)[0]
        first_idx = int(valid_indices[np.argmin(delay_s[valid_indices])])
        first_path_delay = float(delay_s[first_idx])
        first_path_power_dbw = float(power_dbw[first_idx])
        if np.isfinite(aoa_az_deg[first_idx]):
            first_path_aoa_az_deg = float(aoa_az_deg[first_idx])
        interaction_counts = _summed_interaction_counts(inter_code, valid)

    angle_valid = valid & np.isfinite(aoa_az_deg)
    angle_aoa_az_deg = aoa_az_deg[angle_valid]
    angle_power_dbw = power_dbw[angle_valid]
    delay_s = delay_s[valid]
    power_dbw = power_dbw[valid]
    if num_paths_value > 0 and delay_s.size > num_paths_value:
        delay_s = delay_s[:num_paths_value]
        power_dbw = power_dbw[:num_paths_value]

    power_linear = np.power(10.0, power_dbw / 10.0) if power_dbw.size else np.zeros(0)
    delay_spread = compute_delay_spread(delay_s, power_linear)
    angle_power_linear = (
        np.power(10.0, angle_power_dbw / 10.0)
        if angle_power_dbw.size
        else np.zeros(0)
    )
    azimuth_spread_deg = (
        circular_azimuth_spread_deg(angle_aoa_az_deg, angle_power_linear)
        if angle_power_linear.size
        else 0.0
    )
    k_factor_db = estimate_k_factor_db(power_dbw, los_value)

    return {
        "environment_type": env_type or infer_env_type(scenario),
        "los_status": int(los_value == 1),
        "n_paths": max(int(num_paths_value), n_valid_paths),
        "delay_spread": delay_spread,
        "azimuth_spread_aoa": np.deg2rad(azimuth_spread_deg),
        "azimuth_spread_deg": azimuth_spread_deg,
        "k_factor_db": k_factor_db,
        "first_path_delay": first_path_delay,
        "first_path_power_dbw": first_path_power_dbw,
        "first_path_aoa_az_deg": first_path_aoa_az_deg,
        "reflection_count": interaction_counts["reflection"],
        "diffraction_count": interaction_counts["diffraction"],
    }


def build_semantic_key_from_deepmimo(
    scenario: str,
    env_type: str,
    los_value: int,
    num_paths_value: int,
    delay_s: np.ndarray,
    aoa_az_deg: np.ndarray,
    power_dbw: np.ndarray,
    inter_code: np.ndarray,
) -> SemanticKey:
    observables = extract_semantic_observables_from_deepmimo(
        scenario=scenario,
        env_type=env_type,
        los_value=los_value,
        num_paths_value=num_paths_value,
        delay_s=delay_s,
        aoa_az_deg=aoa_az_deg,
        power_dbw=power_dbw,
        inter_code=inter_code,
    )

    from data.semantic_key import build_semantic_key

    return build_semantic_key(observables)


def derive_config_info(dataset) -> tuple[str, int, int]:
    bs_shape = np.asarray(dataset.ch_params["bs_antenna"]["shape"]).astype(int)
    if bs_shape.size == 1:
        n_row, n_col = int(bs_shape[0]), 1
    else:
        n_row, n_col = int(bs_shape[0]), int(bs_shape[1])
    array_type = "ULA" if n_row == 1 or n_col == 1 else "UPA"
    return array_type, n_row, n_col


def derive_bw_bin(n_subcarriers: int) -> int:
    if n_subcarriers <= 64:
        return 0
    if n_subcarriers <= 128:
        return 1
    return 2


def derive_ant_bin(n_tx: int) -> int:
    if n_tx <= 16:
        return 0
    if n_tx <= 64:
        return 1
    return 2


def preprocess_deepmimo_dataset(
    dataset,
    scenario: str,
    freq_bin: int,
    rx_index: int,
    env_type: str | None,
    max_samples: int | None,
    patch_1d: int,
    patch_2d: tuple[int, int],
    target_nf: int,
    include_empty_samples: bool,
) -> list[PreprocessedSample]:
    caption_generator = CaptionGenerator()
    array_type, n_row, n_col = derive_config_info(dataset)
    config_key = f"{array_type}-{n_row}x{n_col}" if array_type == "UPA" else f"{array_type}-{max(n_row, n_col)}"
    n_tx = int(np.prod(np.asarray(dataset.ch_params["bs_antenna"]["shape"])))
    bw_hz = float(dataset.ch_params["ofdm"]["bandwidth"])
    total_subcarriers = int(dataset.ch_params["ofdm"]["subcarriers"])
    selected_subcarriers = np.asarray(dataset.ch_params["selected_subcarriers"])
    n_selected = int(selected_subcarriers.size) if selected_subcarriers.size else total_subcarriers
    subcarrier_spacing_hz = bw_hz / max(total_subcarriers, 1)
    bw_bin = derive_bw_bin(n_selected)
    ant_bin = derive_ant_bin(n_tx)
    array_label = 0 if array_type == "ULA" else 1
    n_samples = int(dataset.channel.shape[0])
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)

    samples: list[PreprocessedSample] = []
    inferred_env = infer_env_type(scenario, fallback=env_type)
    for idx in range(n_samples):
        delay_s = np.asarray(dataset.delay[idx]).reshape(-1)
        power_dbw = np.asarray(dataset.power[idx]).reshape(-1)
        valid_paths = _path_mask(power_dbw, delay_s)
        if not include_empty_samples and not bool(valid_paths.any()):
            continue

        raw_csi = torch.as_tensor(dataset.channel[idx])
        tokens, beam_positions, metadata = preprocess_sample(
            raw_csi=raw_csi,
            array_type=array_type,
            n_row=n_row,
            n_col=n_col,
            n_rx=1,
            target_nf=target_nf,
            patch_1d=patch_1d,
            patch_2d=patch_2d,
            rx_index=rx_index,
        )

        observables = extract_semantic_observables_from_deepmimo(
            scenario=scenario,
            env_type=inferred_env,
            los_value=int(dataset.los[idx]),
            num_paths_value=int(dataset.num_paths[idx]),
            delay_s=delay_s,
            aoa_az_deg=np.asarray(dataset.aoa_az[idx]).reshape(-1),
            power_dbw=power_dbw,
            inter_code=np.asarray(dataset.inter[idx]).reshape(-1),
        )
        from data.semantic_key import build_semantic_key

        semantic_key = build_semantic_key(observables)
        samples.append(
            PreprocessedSample(
                tokens=tokens,
                beam_positions=beam_positions,
                config_key=config_key,
                n_tokens=int(metadata["n_tokens"]),
                freq_bin=freq_bin,
                bw_bin=bw_bin,
                subcarrier_spacing_hz=subcarrier_spacing_hz,
                group_id=f"{scenario}-rx-{idx}",
                semantic_key=semantic_key,
                config_label=0,
                obs_array_label=array_label,
                obs_ant_label=ant_bin,
                obs_freq_label=freq_bin,
                obs_bw_label=bw_bin,
                prop_caption=caption_generator.generate(semantic_key),
                n_paths=int(observables["n_paths"]),
                delay_spread_s=float(observables["delay_spread"]),
                azimuth_spread_deg=float(observables["azimuth_spread_deg"]),
                k_factor_db=float(observables["k_factor_db"]),
                first_path_delay_s=float(observables["first_path_delay"]),
                first_path_power_dbw=float(observables["first_path_power_dbw"]),
                first_path_aoa_az_deg=float(observables["first_path_aoa_az_deg"]),
                reflection_count=int(observables["reflection_count"]),
                diffraction_count=int(observables["diffraction_count"]),
            )
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run preprocessing on synthetic CSI.")
    parser.add_argument(
        "--scenario",
        type=str,
        help="DeepMIMO scenario name, e.g. asu_campus_3p5, or D2Los_Data under --scenario-root.",
    )
    parser.add_argument(
        "--d2los-root",
        type=str,
        help="Path to RayVerse/D2Los_Data. If set, reads map_*/special_points_propbin_*/*.propbin.gz.",
    )
    parser.add_argument("--output", type=str, default="artifacts/preprocessed_samples.pt")
    parser.add_argument("--rx-index", type=int, default=0, help="Which Rx antenna to keep from DeepMIMO channel[i].")
    parser.add_argument("--freq-bin", type=int, default=0)
    parser.add_argument("--env-type", type=str, choices=["indoor", "outdoor", "O2I"])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--scenario-root", type=str, help="Root directory containing scenario folders.")
    parser.add_argument("--max-maps", type=int, help="D2Los only: maximum number of map_* directories.")
    parser.add_argument("--max-sources-per-map", type=int, help="D2Los only: maximum source files per map.")
    parser.add_argument("--max-rx-per-source", type=int, help="D2Los only: maximum RX points per source file.")
    parser.add_argument("--tx-shape", type=int, nargs=2, default=[8, 8], help="D2Los synthetic TX array shape.")
    parser.add_argument("--bandwidth-hz", type=float, default=100e6, help="D2Los synthetic OFDM bandwidth.")
    parser.add_argument("--total-subcarriers", type=int, default=128, help="D2Los synthetic OFDM subcarriers.")
    parser.add_argument("--tx-power-dbm", type=float, default=23.0, help="D2Los TX power used to convert path loss to power.")
    parser.add_argument("--target-nf", type=int, default=128)
    parser.add_argument("--patch-1d", type=int, default=4)
    parser.add_argument("--patch-2d", type=int, nargs=2, default=[2, 2])
    parser.add_argument(
        "--include-empty-samples",
        action="store_true",
        help="Keep samples with no finite path delay/power instead of skipping them.",
    )
    args = parser.parse_args()

    if args.demo:
        raw_csi = torch.randn(8, 64, 32) + 1j * torch.randn(8, 64, 32)
        tokens, beam_positions, metadata = preprocess_sample(
            raw_csi,
            array_type="UPA",
            n_row=8,
            n_col=8,
            rx_index=0,
        )
        print("tokens", tuple(tokens.shape))
        print("beam_positions", tuple(beam_positions.shape))
        print("metadata", metadata)
        return

    d2los_root = Path(args.d2los_root) if args.d2los_root else None
    scenario_name = args.scenario
    if d2los_root is None and args.scenario:
        candidate_root = (
            Path(args.scenario_root) / args.scenario
            if args.scenario_root is not None
            else ROOT / "Raytracing_scenarios" / args.scenario
        )
        if is_d2los_root(candidate_root):
            d2los_root = candidate_root
            scenario_name = args.scenario

    if d2los_root is not None:
        dataset = load_d2los_dataset(
            d2los_root=d2los_root,
            max_samples=args.max_samples,
            max_maps=args.max_maps,
            max_sources_per_map=args.max_sources_per_map,
            max_rx_per_source=args.max_rx_per_source,
            tx_shape=tuple(args.tx_shape),
            bandwidth_hz=args.bandwidth_hz,
            total_subcarriers=args.total_subcarriers,
            tx_power_dbm=args.tx_power_dbm,
        )
        scenario_name = scenario_name or d2los_root.name
    else:
        if not args.scenario:
            raise SystemExit(
                "Provide --scenario for DeepMIMO, provide --d2los-root for D2Los, or run with --demo."
            )
        dataset = load_deepmimo_dataset(
            args.scenario,
            scenario_root=args.scenario_root,
            max_samples=args.max_samples,
        )

    samples = preprocess_deepmimo_dataset(
        dataset=dataset,
        scenario=scenario_name or "D2Los_Data",
        freq_bin=args.freq_bin,
        rx_index=args.rx_index,
        env_type=args.env_type,
        max_samples=args.max_samples,
        patch_1d=args.patch_1d,
        patch_2d=tuple(args.patch_2d),
        target_nf=args.target_nf,
        include_empty_samples=args.include_empty_samples,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, output_path)
    print(f"saved {len(samples)} samples to {output_path}")
    if samples:
        print(f"first sample config={samples[0].config_key} n_tokens={samples[0].n_tokens}")
        print(f"first sample semantic_key={samples[0].semantic_key}")


if __name__ == "__main__":
    main()
