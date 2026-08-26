from __future__ import annotations

"""Evaluate training-free PDP and delay-angle channel estimators.

This script implements two classical baselines for the preprocessed CSI format
used by this repository:

1. IFFT-PDP peak detection for first-path delay, RMS delay spread, first-path
   power, resolvable path count, and a dominant-to-residual power ratio.
2. Bartlett scanning at the earliest detected delay for first-path azimuth,
   plus a delay-integrated angular power spectrum for azimuth spread.

All hyperparameters that depend on labels are selected on validation data and
then frozen.  The test labels are used only for metrics and target text.  The
script deliberately leaves environment and reflection fields unsupported in
the classical prediction record instead of copying their ground truth.

Example:

    python scripts/evaluate_classical_pdp_bartlett.py \
        --validation-data artifacts/d2los_80k_multiconfig_fit70k_val10k/d2los_100k_upa8x8_nf128_los50k_nlos50k_val.pt \
        --test-data artifacts/d2los_6k_multiconfig_final_holdout_seed45678/d2los_6k_upa8x8_nf128_final_test.pt \
        --output-dir output/classical_pdp_bartlett

To reuse a previously frozen calibration:

    python scripts/evaluate_classical_pdp_bartlett.py \
        --calibration-json output/classical_pdp_bartlett/calibration.json \
        --test-data artifacts/test_samples.pt \
        --output-dir output/classical_pdp_bartlett_test
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset  # noqa: E402
from scripts.evaluate import _render_signal_description  # noqa: E402


EPS = 1.0e-30
SUPPORTED_NUMERIC_FIELDS = (
    "path_count",
    "first_path_delay_ns",
    "first_path_angle_deg",
    "first_path_power_dbw",
    "k_factor_db",
    "delay_spread_ns",
    "angle_spread_deg",
)
ANGLE_MODES = (
    "x_sin",
    "y_sin",
    "xy_cos_sin",
    "xy_sin_cos",
)


@dataclass(frozen=True)
class EstimatorOptions:
    n_fft_factor: int = 4
    max_delay_ns: float = 3000.0
    peak_prominence_db: float = 6.0
    min_peak_distance_ns: float = 0.0
    noise_margin_db: float | None = None
    power_window_resolution: float = 1.0
    patch_1d: int = 4
    patch_rows: int = 2
    patch_cols: int = 2
    angle_min_deg: float = -180.0
    angle_max_deg: float = 180.0
    angle_step_deg: float = 1.0
    k_factor_min_db: float = -40.0
    k_factor_max_db: float = 60.0


@dataclass(frozen=True)
class AngleConvention:
    mode: str
    steering_sign: int


@dataclass(frozen=True)
class Calibration:
    pdp_threshold_db: float
    first_power_offset_db: float
    los_k_threshold_db: float
    angle_conventions: dict[str, dict[str, Any]]
    options: dict[str, Any]
    validation_samples: int
    selection_metrics: dict[str, Any]


@dataclass
class PreparedCSI:
    sample: Any
    antenna_delay_response: np.ndarray
    pdp: np.ndarray
    delays_ns: np.ndarray
    delay_step_ns: float
    native_delay_resolution_ns: float
    coordinates_wavelengths: np.ndarray
    array_type: str


@dataclass
class TemporalEstimate:
    first_peak_index: int
    peak_indices: np.ndarray
    retained_mask: np.ndarray
    detection_threshold: float
    first_path_delay_ns: float
    delay_spread_ns: float
    first_path_power_dbw_uncalibrated: float
    path_count: int
    dominant_to_residual_db: float


@dataclass
class ClassicalEstimate:
    first_path_delay_ns: float
    delay_spread_ns: float
    first_path_power_dbw: float
    path_count: int
    k_factor_db: float
    first_path_angle_deg: float
    angle_spread_deg: float
    los_status: str
    detection_threshold_db_below_peak: float


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated floating-point value.")
    return values


def parse_str_list(text: str, choices: Iterable[str]) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = sorted(set(values) - set(choices))
    if not values or unknown:
        raise ValueError(
            f"Invalid list {text!r}; choices are {', '.join(choices)}"
            + (f"; unknown values: {unknown}" if unknown else "")
        )
    return values


def load_samples(path: str, limit: int | None) -> list[Any]:
    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"CSI split not found: {input_path.resolve()}. "
            "The paths in this script's usage example must match the split files "
            "available on the current server; list them with: "
            "find artifacts -type f -name '*.pt'"
        )
    samples = PreprocessedCSIDataset.from_pt(str(input_path)).samples
    if limit is not None:
        if limit <= 0:
            raise ValueError("Dataset limits must be positive.")
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No samples loaded from {path}.")
    return samples


def complex_token_chunks(sample: Any) -> np.ndarray:
    tokens = sample.tokens[: int(sample.n_tokens)].detach().cpu().float()
    if tokens.ndim != 3 or tokens.shape[1] % 2:
        raise ValueError(
            "Expected tokens with shape [K, 2*C, F] and real/imaginary halves, "
            f"got {tuple(tokens.shape)} for group_id={getattr(sample, 'group_id', '')!r}."
        )
    half = tokens.shape[1] // 2
    return (
        tokens[:, :half].numpy().astype(np.float64, copy=False)
        + 1j * tokens[:, half:].numpy().astype(np.float64, copy=False)
    )


def reconstruct_beam_csi(sample: Any, options: EstimatorOptions) -> np.ndarray:
    """Undo beamspace patch tokenization and return [N_beam, N_frequency]."""
    chunks = complex_token_chunks(sample)
    array_type = str(getattr(sample, "array_type", "")).upper()
    rows = int(getattr(sample, "array_rows", 0) or 0)
    cols = int(getattr(sample, "array_cols", 0) or 0)
    if rows <= 0 or cols <= 0:
        raise ValueError(
            f"Missing array geometry for group_id={getattr(sample, 'group_id', '')!r}."
        )

    n_freq = chunks.shape[-1]
    if array_type == "ULA":
        n_antennas = rows * cols
        patch = int(options.patch_1d)
        if chunks.shape[1] != patch:
            raise ValueError(
                f"ULA token has {chunks.shape[1]} complex channels but --patch-1d={patch}."
            )
        expected_tokens = math.ceil(n_antennas / patch)
        if chunks.shape[0] != expected_tokens:
            raise ValueError(
                f"ULA token count mismatch: got {chunks.shape[0]}, expected {expected_tokens}."
            )
        beam = np.zeros((n_antennas, n_freq), dtype=np.complex128)
        for token_index in range(expected_tokens):
            start = token_index * patch
            count = min(patch, n_antennas - start)
            beam[start : start + count] = chunks[token_index, :count]
        return beam

    if array_type != "UPA":
        raise ValueError(
            f"Unsupported array_type={array_type!r}; only ULA and UPA are supported."
        )

    patch_rows = int(options.patch_rows)
    patch_cols = int(options.patch_cols)
    if chunks.shape[1] != patch_rows * patch_cols:
        raise ValueError(
            "UPA token complex-channel count does not match --patch-rows * --patch-cols: "
            f"{chunks.shape[1]} != {patch_rows} * {patch_cols}."
        )
    token_rows = math.ceil(rows / patch_rows)
    token_cols = math.ceil(cols / patch_cols)
    if chunks.shape[0] != token_rows * token_cols:
        raise ValueError(
            f"UPA token count mismatch: got {chunks.shape[0]}, "
            f"expected {token_rows * token_cols} for {rows}x{cols}."
        )
    beam_grid = np.zeros((rows, cols, n_freq), dtype=np.complex128)
    token_index = 0
    for token_row in range(token_rows):
        for token_col in range(token_cols):
            patch = chunks[token_index].reshape(patch_rows, patch_cols, n_freq)
            row_start = token_row * patch_rows
            col_start = token_col * patch_cols
            valid_rows = min(patch_rows, rows - row_start)
            valid_cols = min(patch_cols, cols - col_start)
            beam_grid[
                row_start : row_start + valid_rows,
                col_start : col_start + valid_cols,
            ] = patch[:valid_rows, :valid_cols]
            token_index += 1
    return beam_grid.reshape(rows * cols, n_freq)


def beam_to_antenna_csi(sample: Any, beam_csi: np.ndarray) -> np.ndarray:
    rows = int(sample.array_rows)
    cols = int(sample.array_cols)
    array_type = str(sample.array_type).upper()
    if array_type == "ULA":
        return np.fft.ifft(beam_csi, axis=0)
    beam_grid = beam_csi.reshape(rows, cols, beam_csi.shape[-1])
    return np.fft.ifft2(beam_grid, axes=(0, 1)).reshape(rows * cols, beam_csi.shape[-1])


def frequency_to_delay(
    antenna_csi: np.ndarray,
    n_fft_factor: int,
) -> np.ndarray:
    """Convert centered frequency samples to delay while preserving amplitude."""
    n_freq = int(antenna_csi.shape[-1])
    n_fft = n_freq * int(n_fft_factor)
    if n_fft_factor <= 0:
        raise ValueError("--n-fft-factor must be positive.")
    left = (n_fft - n_freq) // 2
    centered = np.zeros((*antenna_csi.shape[:-1], n_fft), dtype=np.complex128)
    centered[..., left : left + n_freq] = antenna_csi
    # NumPy's IFFT divides by n_fft.  Multiplication by n_fft / n_freq keeps
    # the physical amplitude invariant when the centered spectrum is padded.
    return (
        np.fft.ifft(np.fft.ifftshift(centered, axes=-1), axis=-1)
        * (float(n_fft) / float(n_freq))
    )


def sample_coordinates(sample: Any, antenna_count: int) -> np.ndarray:
    coordinates = getattr(sample, "antenna_coordinates_wavelengths", None)
    if isinstance(coordinates, torch.Tensor):
        result = coordinates.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        result = np.asarray(coordinates, dtype=np.float64)
    if result.shape != (antenna_count, 3):
        raise ValueError(
            f"Expected antenna coordinates [{antenna_count}, 3], got {result.shape}."
        )
    return result


def prepare_csi(sample: Any, options: EstimatorOptions) -> PreparedCSI:
    beam_csi = reconstruct_beam_csi(sample, options)
    antenna_csi = beam_to_antenna_csi(sample, beam_csi)
    delay_response = frequency_to_delay(antenna_csi, options.n_fft_factor)
    pdp = np.sum(np.abs(delay_response) ** 2, axis=0).real
    bandwidth_hz = finite_float(getattr(sample, "bandwidth_hz", math.nan))
    if bandwidth_hz is None or bandwidth_hz <= 0.0:
        source_spacing_hz = finite_float(
            getattr(sample, "subcarrier_spacing_hz", math.nan)
        )
        source_n_freq = int(getattr(sample, "source_n_freq", 0) or 0)
        if source_spacing_hz is None or source_spacing_hz <= 0.0 or source_n_freq <= 0:
            raise ValueError(
                "Each sample must contain positive bandwidth_hz, or both "
                "subcarrier_spacing_hz and source_n_freq."
            )
        bandwidth_hz = source_spacing_hz * source_n_freq
    # preprocess_sample resamples the centered frequency response at fixed
    # bandwidth.  Its output spacing is therefore bandwidth / token_n_freq,
    # which can differ from the original source subcarrier spacing.
    spacing_hz = bandwidth_hz / beam_csi.shape[-1]
    n_fft = delay_response.shape[-1]
    delay_step_ns = 1.0e9 / (n_fft * spacing_hz)
    native_resolution_ns = 1.0e9 / (beam_csi.shape[-1] * spacing_hz)
    delays_ns = np.arange(n_fft, dtype=np.float64) * delay_step_ns
    return PreparedCSI(
        sample=sample,
        antenna_delay_response=delay_response,
        pdp=pdp,
        delays_ns=delays_ns,
        delay_step_ns=delay_step_ns,
        native_delay_resolution_ns=native_resolution_ns,
        coordinates_wavelengths=sample_coordinates(sample, antenna_csi.shape[0]),
        array_type=str(sample.array_type).upper(),
    )


def local_peak_indices(
    profile: np.ndarray,
    threshold: float,
    prominence: float,
    distance_bins: int,
) -> np.ndarray:
    peaks, _ = find_peaks(
        profile,
        height=threshold,
        prominence=prominence,
        distance=max(int(distance_bins), 1),
    )
    candidates = list(int(index) for index in peaks)
    if profile.size == 1 or (
        profile[0] >= threshold and profile[0] >= profile[min(1, profile.size - 1)]
    ):
        candidates.append(0)
    if profile.size > 1 and profile[-1] >= threshold and profile[-1] >= profile[-2]:
        candidates.append(profile.size - 1)
    return np.asarray(sorted(set(candidates)), dtype=np.int64)


def temporal_estimate(
    prepared: PreparedCSI,
    threshold_db: float,
    options: EstimatorOptions,
) -> TemporalEstimate:
    search_mask = prepared.delays_ns <= float(options.max_delay_ns)
    if not bool(np.any(search_mask)):
        raise ValueError("--max-delay-ns excludes every delay bin.")
    search_count = int(np.flatnonzero(search_mask)[-1]) + 1
    profile = prepared.pdp[:search_count]
    peak_power = float(np.max(profile))
    if not math.isfinite(peak_power) or peak_power <= 0.0:
        raise ValueError(
            f"CSI has zero/non-finite power for group_id={getattr(prepared.sample, 'group_id', '')!r}."
        )

    relative_threshold = peak_power * 10.0 ** (-float(threshold_db) / 10.0)
    detection_threshold = relative_threshold
    if options.noise_margin_db is not None:
        finite_power = profile[np.isfinite(profile)]
        noise_floor = float(np.median(finite_power)) if finite_power.size else 0.0
        detection_threshold = max(
            detection_threshold,
            noise_floor * 10.0 ** (float(options.noise_margin_db) / 10.0),
        )
    # Express prominence relative to the selected detection floor.  Using the
    # global peak as its reference would reject every sufficiently weak path
    # regardless of the separately validated height threshold.
    prominence = detection_threshold * 10.0 ** (
        -float(options.peak_prominence_db) / 10.0
    )
    distance_bins = max(
        1,
        int(math.ceil(float(options.min_peak_distance_ns) / prepared.delay_step_ns)),
    )
    peaks = local_peak_indices(profile, detection_threshold, prominence, distance_bins)
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(profile))], dtype=np.int64)
    first_index = int(peaks[0])

    retained = np.zeros_like(prepared.pdp, dtype=bool)
    retained[:search_count] = profile >= detection_threshold
    retained[peaks] = True
    retained_power = prepared.pdp[retained]
    retained_delays = prepared.delays_ns[retained]
    total_retained = float(np.sum(retained_power))
    if total_retained <= 0.0:
        delay_spread_ns = 0.0
    else:
        mean_delay = float(np.sum(retained_power * retained_delays) / total_retained)
        variance = float(
            np.sum(retained_power * (retained_delays - mean_delay) ** 2)
            / total_retained
        )
        delay_spread_ns = math.sqrt(max(variance, 0.0))

    half_window_ns = (
        0.5
        * float(options.power_window_resolution)
        * prepared.native_delay_resolution_ns
    )
    half_window_bins = max(0, int(math.ceil(half_window_ns / prepared.delay_step_ns)))

    def peak_window(index: int) -> tuple[int, int]:
        start = max(0, int(index) - half_window_bins)
        stop = min(prepared.pdp.size, int(index) + half_window_bins + 1)
        return start, stop

    def integrated_peak_power(index: int) -> float:
        start, stop = peak_window(index)
        # Mean over antennas maps an ideal single path back to its per-antenna
        # channel power; summing would introduce an array-size-dependent gain.
        # Oversampled IFFT energy is n_fft_factor times the native-grid energy.
        return float(np.sum(prepared.pdp[start:stop])) / (
            float(prepared.antenna_delay_response.shape[0])
            * float(options.n_fft_factor)
        )

    first_power = integrated_peak_power(first_index)
    peak_powers = np.asarray([integrated_peak_power(int(index)) for index in peaks])
    dominant_power = float(np.max(peak_powers))
    signal_mask = retained.copy()
    for index in peaks:
        start, stop = peak_window(int(index))
        signal_mask[start:stop] = True
    total_per_antenna = float(np.sum(prepared.pdp[signal_mask])) / (
        float(prepared.antenna_delay_response.shape[0])
        * float(options.n_fft_factor)
    )
    residual_power = max(total_per_antenna - dominant_power, EPS)
    ratio_db = 10.0 * math.log10(max(dominant_power, EPS) / residual_power)
    ratio_db = float(
        np.clip(ratio_db, options.k_factor_min_db, options.k_factor_max_db)
    )
    return TemporalEstimate(
        first_peak_index=first_index,
        peak_indices=peaks,
        retained_mask=retained,
        detection_threshold=detection_threshold,
        first_path_delay_ns=float(prepared.delays_ns[first_index]),
        delay_spread_ns=delay_spread_ns,
        first_path_power_dbw_uncalibrated=10.0 * math.log10(max(first_power, EPS)),
        path_count=int(peaks.size),
        dominant_to_residual_db=ratio_db,
    )


def direction_vectors(angles_deg: np.ndarray, mode: str) -> np.ndarray:
    radians = np.deg2rad(angles_deg)
    zeros = np.zeros_like(radians)
    if mode == "x_sin":
        return np.stack([np.sin(radians), zeros, zeros], axis=1)
    if mode == "y_sin":
        return np.stack([zeros, np.sin(radians), zeros], axis=1)
    if mode == "xy_cos_sin":
        return np.stack([np.cos(radians), np.sin(radians), zeros], axis=1)
    if mode == "xy_sin_cos":
        return np.stack([np.sin(radians), np.cos(radians), zeros], axis=1)
    raise ValueError(f"Unsupported angle convention mode={mode!r}.")


def steering_matrix(
    coordinates_wavelengths: np.ndarray,
    angles_deg: np.ndarray,
    convention: AngleConvention,
) -> np.ndarray:
    directions = direction_vectors(angles_deg, convention.mode)
    phase = 2.0 * np.pi * directions @ coordinates_wavelengths.T
    return np.exp(1j * float(convention.steering_sign) * phase)


def circular_angle_error_deg(prediction: float, target: float) -> float:
    delta = math.radians(float(prediction) - float(target))
    return abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))


def circular_spread_deg(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return math.nan
    resultant = np.sum(weights * np.exp(1j * np.deg2rad(angles_deg))) / total
    magnitude = float(np.clip(abs(resultant), 1.0e-12, 1.0))
    return min(math.degrees(math.sqrt(max(-2.0 * math.log(magnitude), 0.0))), 180.0)


def bartlett_estimate(
    prepared: PreparedCSI,
    temporal: TemporalEstimate,
    convention: AngleConvention,
    options: EstimatorOptions,
    *,
    compute_spread: bool,
    steering_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, float]:
    cache_key = (
        prepared.array_type,
        prepared.coordinates_wavelengths.shape,
        prepared.coordinates_wavelengths.tobytes(),
        convention.mode,
        convention.steering_sign,
        options.angle_min_deg,
        options.angle_max_deg,
        options.angle_step_deg,
    )
    cached = steering_cache.get(cache_key) if steering_cache is not None else None
    if cached is None:
        angles = np.arange(
            options.angle_min_deg,
            options.angle_max_deg + options.angle_step_deg * 0.5,
            options.angle_step_deg,
            dtype=np.float64,
        )
        if angles.size < 2:
            raise ValueError("The angle scan grid must contain at least two values.")
        steering = steering_matrix(
            prepared.coordinates_wavelengths,
            angles,
            convention,
        )
        if steering_cache is not None:
            steering_cache[cache_key] = (angles, steering)
    else:
        angles, steering = cached
    first_snapshot = prepared.antenna_delay_response[:, temporal.first_peak_index]
    first_spectrum = np.abs(steering.conj() @ first_snapshot) ** 2
    first_angle = float(angles[int(np.argmax(first_spectrum))])
    if not compute_spread:
        return first_angle, math.nan

    # Integrating the Bartlett spectra at the detected delay peaks is the
    # classical resolvable-path approximation.  It also avoids counting many
    # oversampled samples from the same delay lobe as separate observations.
    snapshots = prepared.antenna_delay_response[:, temporal.peak_indices]
    angular_spectrum = np.sum(np.abs(steering.conj() @ snapshots) ** 2, axis=1)
    return first_angle, circular_spread_deg(angles, angular_spectrum)


def true_los_status(sample: Any) -> str:
    status = str(getattr(sample.semantic_key, "los_status", "")).lower()
    return "los" if status == "los" else "nlos"


def true_first_angle_deg(sample: Any) -> float | None:
    return finite_float(getattr(sample, "first_path_aoa_az_deg", math.nan))


def select_pdp_threshold(
    samples: list[Any],
    candidates_db: tuple[float, ...],
    options: EstimatorOptions,
) -> tuple[float, dict[str, Any]]:
    results: dict[str, Any] = {}
    best: tuple[float, float] | None = None
    errors_by_threshold = {float(value): [] for value in candidates_db}
    for sample in samples:
        target_s = finite_float(getattr(sample, "first_path_delay_s", math.nan))
        if target_s is None:
            continue
        prepared = prepare_csi(sample, options)
        for threshold_db in candidates_db:
            estimate = temporal_estimate(prepared, threshold_db, options)
            errors_by_threshold[float(threshold_db)].append(
                abs(estimate.first_path_delay_ns - target_s * 1.0e9)
            )
    for threshold_db in candidates_db:
        errors = errors_by_threshold[float(threshold_db)]
        mae = float(np.mean(errors)) if errors else math.inf
        results[f"{threshold_db:g}"] = {
            "first_path_delay_mae_ns": mae,
            "valid_samples": len(errors),
        }
        score = (mae, float(threshold_db))
        if best is None or score < best:
            best = score
    if best is None or not math.isfinite(best[0]):
        raise ValueError("No finite first-path delay labels are available for calibration.")
    return best[1], results


def select_angle_conventions_from_errors(
    errors: dict[str, dict[str, list[float]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    for array_type in sorted(errors):
        candidate_metrics: dict[str, Any] = {}
        best: tuple[float, str, int] | None = None
        for key, values in errors[array_type].items():
            mode, sign_text = key.split(":sign=")
            sign = int(sign_text)
            mae = float(np.mean(values)) if values else math.inf
            candidate_metrics[key] = {
                "first_path_angle_mae_deg": mae,
                "valid_samples": len(values),
            }
            score = (mae, mode, sign)
            if best is None or score < best:
                best = score
        if best is None or not math.isfinite(best[0]):
            # This is a geometry convention, not a learned predictor.  A
            # documented default keeps inference possible if angle labels are
            # absent from validation data.
            default_mode = "y_sin" if array_type == "ULA" else "xy_cos_sin"
            selected[array_type] = {"mode": default_mode, "steering_sign": 1}
        else:
            selected[array_type] = {"mode": best[1], "steering_sign": best[2]}
        diagnostics[array_type] = candidate_metrics
    return selected, diagnostics


def select_los_threshold(
    ratios_db: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    finite = np.isfinite(ratios_db)
    ratios = ratios_db[finite]
    labels = labels[finite].astype(bool)
    if ratios.size == 0:
        return 0.0, {"balanced_accuracy": math.nan, "accuracy": math.nan}
    unique = np.unique(ratios)
    if unique.size == 1:
        candidates = np.asarray([unique[0]], dtype=np.float64)
    else:
        candidates = np.concatenate(
            [
                [unique[0] - 1.0e-6],
                (unique[:-1] + unique[1:]) / 2.0,
                [unique[-1] + 1.0e-6],
            ]
        )
    best: tuple[float, float, float] | None = None
    best_stats: dict[str, Any] = {}
    positives = labels
    negatives = ~labels
    for threshold in candidates:
        predictions = ratios >= threshold
        tpr = float(np.mean(predictions[positives])) if bool(np.any(positives)) else math.nan
        tnr = float(np.mean(~predictions[negatives])) if bool(np.any(negatives)) else math.nan
        recalls = [value for value in (tpr, tnr) if math.isfinite(value)]
        balanced_accuracy = float(np.mean(recalls)) if recalls else math.nan
        accuracy = float(np.mean(predictions == labels))
        score = (balanced_accuracy, accuracy, -abs(float(threshold)))
        if best is None or score > best:
            best = score
            best_stats = {
                "balanced_accuracy": balanced_accuracy,
                "accuracy": accuracy,
                "true_positive_rate": tpr,
                "true_negative_rate": tnr,
            }
            best_threshold = float(threshold)
    return best_threshold, best_stats


def calibrate(
    samples: list[Any],
    threshold_candidates: tuple[float, ...],
    angle_modes: tuple[str, ...],
    options: EstimatorOptions,
    calibrate_power: bool,
) -> Calibration:
    threshold_db, threshold_metrics = select_pdp_threshold(
        samples,
        threshold_candidates,
        options,
    )
    power_residuals = []
    ratios = []
    los_labels = []
    angle_errors: dict[str, dict[str, list[float]]] = {}
    steering_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    for sample in samples:
        prepared = prepare_csi(sample, options)
        estimate = temporal_estimate(prepared, threshold_db, options)
        target = finite_float(getattr(sample, "first_path_power_dbw", math.nan))
        if target is not None:
            power_residuals.append(target - estimate.first_path_power_dbw_uncalibrated)
        ratios.append(estimate.dominant_to_residual_db)
        los_labels.append(true_los_status(sample) == "los")

        target_angle = true_first_angle_deg(sample)
        if target_angle is None:
            continue
        type_errors = angle_errors.setdefault(prepared.array_type, {})
        for mode in angle_modes:
            for sign in (-1, 1):
                key = f"{mode}:sign={sign:+d}"
                predicted, _ = bartlett_estimate(
                    prepared,
                    estimate,
                    AngleConvention(mode=mode, steering_sign=sign),
                    options,
                    compute_spread=False,
                    steering_cache=steering_cache,
                )
                type_errors.setdefault(key, []).append(
                    circular_angle_error_deg(predicted, target_angle)
                )
    power_offset = (
        float(np.median(power_residuals)) if calibrate_power and power_residuals else 0.0
    )

    los_threshold, los_metrics = select_los_threshold(
        np.asarray(ratios, dtype=np.float64),
        np.asarray(los_labels, dtype=bool),
    )
    conventions, angle_metrics = select_angle_conventions_from_errors(angle_errors)
    return Calibration(
        pdp_threshold_db=threshold_db,
        first_power_offset_db=power_offset,
        los_k_threshold_db=los_threshold,
        angle_conventions=conventions,
        options=asdict(options),
        validation_samples=len(samples),
        selection_metrics={
            "pdp_threshold_candidates": threshold_metrics,
            "first_power_calibration": {
                "enabled": calibrate_power,
                "median_offset_db": power_offset,
                "valid_samples": len(power_residuals),
            },
            "los_threshold": los_metrics,
            "angle_conventions": angle_metrics,
        },
    )


def calibration_from_json(path: str) -> Calibration:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Calibration(**payload)


def convention_for(prepared: PreparedCSI, calibration: Calibration) -> AngleConvention:
    raw = calibration.angle_conventions.get(prepared.array_type)
    if raw is None:
        raw = {
            "mode": "y_sin" if prepared.array_type == "ULA" else "xy_cos_sin",
            "steering_sign": 1,
        }
    return AngleConvention(mode=str(raw["mode"]), steering_sign=int(raw["steering_sign"]))


def estimate_sample(
    sample: Any,
    calibration: Calibration,
    options: EstimatorOptions,
    steering_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] | None = None,
) -> ClassicalEstimate:
    prepared = prepare_csi(sample, options)
    temporal = temporal_estimate(prepared, calibration.pdp_threshold_db, options)
    first_angle, angle_spread = bartlett_estimate(
        prepared,
        temporal,
        convention_for(prepared, calibration),
        options,
        compute_spread=True,
        steering_cache=steering_cache,
    )
    ratio = temporal.dominant_to_residual_db
    los_status = "los" if ratio >= calibration.los_k_threshold_db else "nlos"
    return ClassicalEstimate(
        first_path_delay_ns=temporal.first_path_delay_ns,
        delay_spread_ns=temporal.delay_spread_ns,
        first_path_power_dbw=(
            temporal.first_path_power_dbw_uncalibrated
            + calibration.first_power_offset_db
        ),
        path_count=temporal.path_count,
        k_factor_db=ratio,
        first_path_angle_deg=first_angle,
        angle_spread_deg=angle_spread,
        los_status=los_status,
        detection_threshold_db_below_peak=calibration.pdp_threshold_db,
    )


def prediction_record(
    estimate: ClassicalEstimate,
    *,
    include_angles: bool,
) -> dict[str, float | str]:
    is_los = estimate.los_status == "los"
    return {
        "environment": "unknown",
        "los_status": estimate.los_status,
        "path_count": float(estimate.path_count),
        "first_path_delay_ns": estimate.first_path_delay_ns,
        "first_path_angle_deg": estimate.first_path_angle_deg if include_angles else math.nan,
        "first_path_power_dbw": estimate.first_path_power_dbw,
        "k_factor_db": estimate.k_factor_db,
        "delay_spread_ns": estimate.delay_spread_ns,
        "angle_spread_deg": estimate.angle_spread_deg if include_angles else math.nan,
        "los_delay_ns": estimate.first_path_delay_ns if is_los else math.nan,
        "los_angle_deg": (
            estimate.first_path_angle_deg if is_los and include_angles else math.nan
        ),
        "reflection_count": math.nan,
        "reflection_path_count": math.nan,
    }


def target_record(sample: Any) -> dict[str, float | str]:
    first_delay_s = finite_float(getattr(sample, "first_path_delay_s", math.nan))
    los_delay_s = finite_float(getattr(sample, "los_delay_s", math.nan))
    return {
        "environment": str(getattr(sample.semantic_key, "env_type", "unknown")),
        "los_status": true_los_status(sample),
        "path_count": float(getattr(sample, "n_paths", math.nan)),
        "first_path_delay_ns": (
            first_delay_s * 1.0e9 if first_delay_s is not None else math.nan
        ),
        "first_path_angle_deg": float(
            getattr(sample, "first_path_aoa_az_deg", math.nan)
        ),
        "first_path_power_dbw": float(
            getattr(sample, "first_path_power_dbw", math.nan)
        ),
        "k_factor_db": float(getattr(sample, "k_factor_db", math.nan)),
        "delay_spread_ns": float(getattr(sample, "delay_spread_s", math.nan))
        * 1.0e9,
        "angle_spread_deg": float(
            getattr(sample, "azimuth_spread_deg", math.nan)
        ),
        "los_delay_ns": los_delay_s * 1.0e9 if los_delay_s is not None else math.nan,
        "los_angle_deg": float(getattr(sample, "los_aoa_az_deg", math.nan)),
        "reflection_count": float(getattr(sample, "reflection_count", math.nan)),
        "reflection_path_count": float(
            getattr(sample, "reflection_path_count", math.nan)
        ),
    }


def error_value(field: str, prediction: float, target: float) -> float:
    if field == "first_path_angle_deg":
        return circular_angle_error_deg(prediction, target)
    return float(prediction) - float(target)


def regression_metrics(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    signed_errors = []
    absolute_errors = []
    for prediction, target in zip(predictions, targets):
        pred_value = finite_float(prediction.get(field))
        target_value = finite_float(target.get(field))
        if pred_value is None or target_value is None:
            continue
        error = error_value(field, pred_value, target_value)
        # Circular angle error is intrinsically unsigned.
        signed_errors.append(error)
        absolute_errors.append(abs(error))
    if not absolute_errors:
        return {
            "count": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "median_absolute_error": math.nan,
            "p90_absolute_error": math.nan,
            "bias": math.nan,
        }
    return {
        "count": len(absolute_errors),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(np.square(absolute_errors)))),
        "median_absolute_error": float(np.median(absolute_errors)),
        "p90_absolute_error": float(np.quantile(absolute_errors, 0.9)),
        "bias": (
            math.nan
            if field == "first_path_angle_deg"
            else float(np.mean(signed_errors))
        ),
    }


def classification_metrics(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    pred_los = np.asarray(
        [str(record["los_status"]).lower() == "los" for record in predictions]
    )
    true_los = np.asarray(
        [str(record["los_status"]).lower() == "los" for record in targets]
    )
    tp = int(np.sum(pred_los & true_los))
    tn = int(np.sum(~pred_los & ~true_los))
    fp = int(np.sum(pred_los & ~true_los))
    fn = int(np.sum(~pred_los & true_los))
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision)
        and math.isfinite(recall)
        and precision + recall > 0.0
        else math.nan
    )
    return {
        "count": int(true_los.size),
        "accuracy": float(np.mean(pred_los == true_los)),
        "balanced_accuracy": float(np.nanmean([recall, specificity])),
        "precision_los": precision,
        "recall_los": recall,
        "specificity_nlos": specificity,
        "f1_los": f1,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def build_metrics(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "numeric": {
            field: regression_metrics(predictions, targets, field)
            for field in SUPPORTED_NUMERIC_FIELDS
        },
        "los_nlos": classification_metrics(predictions, targets),
        "notes": {
            "path_count": "Number of resolvable PDP peaks, not ray-tracing path count.",
            "k_factor_db": (
                "Dominant-to-residual power ratio heuristic, not a classical "
                "multi-snapshot Rician moment estimator."
            ),
            "first_path_power_dbw": (
                "Computed from unnormalized CSI with one validation-set dB offset."
            ),
            "unsupported": [
                "environment",
                "reflection_count",
                "reflection_path_count",
                "diffraction_count",
            ],
        },
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_metrics_csv(path: Path, metrics_by_method: dict[str, dict[str, Any]]) -> None:
    rows = []
    for method, metrics in metrics_by_method.items():
        for field, values in metrics["numeric"].items():
            rows.append({"method": method, "task": field, **values})
        los = metrics["los_nlos"]
        rows.append(
            {
                "method": method,
                "task": "los_nlos",
                "count": los["count"],
                "accuracy": los["accuracy"],
                "balanced_accuracy": los["balanced_accuracy"],
                "f1_los": los["f1_los"],
            }
        )
    fieldnames = sorted(
        {key for row in rows for key in row},
        key=lambda key: (key not in {"method", "task"}, key),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    samples: list[Any],
    calibration: Calibration,
    options: EstimatorOptions,
    output_dir: Path,
) -> dict[str, Any]:
    method_names = ("classical_pdp", "classical_delay_angle_bartlett")
    predicted_records: dict[str, list[dict[str, Any]]] = {
        method: [] for method in method_names
    }
    target_records: list[dict[str, Any]] = []
    predicted_texts: dict[str, list[str]] = {method: [] for method in method_names}
    target_texts: list[str] = []
    comparisons: dict[str, list[dict[str, Any]]] = {
        method: [] for method in method_names
    }
    steering_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as jsonl:
        for index, sample in enumerate(samples):
            estimate = estimate_sample(
                sample,
                calibration,
                options,
                steering_cache=steering_cache,
            )
            target = target_record(sample)
            target_text = _render_signal_description(target)
            target_records.append(target)
            target_texts.append(target_text)
            for method in method_names:
                predicted = prediction_record(
                    estimate,
                    include_angles=(method == "classical_delay_angle_bartlett"),
                )
                predicted_text = _render_signal_description(predicted)
                predicted_records[method].append(predicted)
                predicted_texts[method].append(predicted_text)
                comparison = {
                    "index": index,
                    "group_id": str(getattr(sample, "group_id", "")),
                    "config_key": str(getattr(sample, "config_key", "")),
                    "method": method,
                    "predicted_signal_description": predicted_text,
                    "target_signal_description": target_text,
                    "predicted_record": predicted,
                    "target_record": target,
                }
                comparisons[method].append(comparison)
                jsonl.write(
                    json.dumps(json_safe(comparison), ensure_ascii=False) + "\n"
                )

    metrics = {
        method: build_metrics(predicted_records[method], target_records)
        for method in method_names
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metrics), handle, indent=2, ensure_ascii=False)
    write_metrics_csv(output_dir / "metrics.csv", metrics)

    for method in method_names:
        payload = {
            "predicted_signal_descriptions": predicted_texts[method],
            "target_signal_descriptions": target_texts,
            "predicted_signal_records": predicted_records[method],
            "target_signal_records": target_records,
            "comparisons": comparisons[method],
            "metadata": {
                "baseline": method,
                "test_samples": len(samples),
                "calibration": asdict(calibration),
                "supported_numeric_fields": [
                    field
                    for field in SUPPORTED_NUMERIC_FIELDS
                    if method == "classical_delay_angle_bartlett"
                    or field not in {"first_path_angle_deg", "angle_spread_deg"}
                ],
                "unsupported_fields_are_not_imputed_from_ground_truth": True,
            },
        }
        torch.save(payload, output_dir / f"signal_descriptions_{method}.pt")
    return metrics


def validate_options(options: EstimatorOptions) -> None:
    if options.n_fft_factor <= 0:
        raise ValueError("--n-fft-factor must be positive.")
    if options.max_delay_ns <= 0.0:
        raise ValueError("--max-delay-ns must be positive.")
    if options.peak_prominence_db < 0.0:
        raise ValueError("--peak-prominence-db must be non-negative.")
    if options.min_peak_distance_ns < 0.0:
        raise ValueError("--min-peak-distance-ns must be non-negative.")
    if options.power_window_resolution < 0.0:
        raise ValueError("--power-window-resolution must be non-negative.")
    if options.patch_1d <= 0 or options.patch_rows <= 0 or options.patch_cols <= 0:
        raise ValueError("Patch sizes must be positive.")
    if options.angle_step_deg <= 0.0 or options.angle_min_deg >= options.angle_max_deg:
        raise ValueError("Invalid angle scan range/step.")
    if options.k_factor_min_db >= options.k_factor_max_db:
        raise ValueError("Invalid K-factor clipping range.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate training-free IFFT-PDP and Bartlett CSI baselines."
    )
    calibration_group = parser.add_mutually_exclusive_group(required=True)
    calibration_group.add_argument(
        "--validation-data",
        help="Validation .pt used only to freeze classical estimator settings.",
    )
    calibration_group.add_argument(
        "--calibration-json",
        help="Previously frozen calibration.json; no validation labels are read.",
    )
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument(
        "--threshold-db-candidates",
        default="15,20,25,30",
        help="Validation candidates for the PDP height threshold below its maximum.",
    )
    parser.add_argument(
        "--angle-mode-candidates",
        default=",".join(ANGLE_MODES),
        help="Validation candidates for mapping azimuth to the stored array coordinates.",
    )
    parser.add_argument(
        "--disable-power-calibration",
        action="store_true",
        help="Do not fit the single validation-set first-power dB offset.",
    )
    parser.add_argument("--n-fft-factor", type=int, default=4)
    parser.add_argument("--max-delay-ns", type=float, default=3000.0)
    parser.add_argument(
        "--peak-prominence-db",
        type=float,
        default=6.0,
        help="Required prominence in dB below the selected PDP detection floor.",
    )
    parser.add_argument("--min-peak-distance-ns", type=float, default=0.0)
    parser.add_argument(
        "--noise-margin-db",
        type=float,
        help="Optional floor: median PDP noise plus this many dB.",
    )
    parser.add_argument(
        "--power-window-resolution",
        type=float,
        default=1.0,
        help="Peak integration window width in native (pre-zero-padding) delay bins.",
    )
    parser.add_argument("--patch-1d", type=int, default=4)
    parser.add_argument("--patch-rows", type=int, default=2)
    parser.add_argument("--patch-cols", type=int, default=2)
    parser.add_argument("--angle-min-deg", type=float, default=-180.0)
    parser.add_argument("--angle-max-deg", type=float, default=180.0)
    parser.add_argument("--angle-step-deg", type=float, default=1.0)
    parser.add_argument("--k-factor-min-db", type=float, default=-40.0)
    parser.add_argument("--k-factor-max-db", type=float, default=60.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cli_options = EstimatorOptions(
        n_fft_factor=args.n_fft_factor,
        max_delay_ns=args.max_delay_ns,
        peak_prominence_db=args.peak_prominence_db,
        min_peak_distance_ns=args.min_peak_distance_ns,
        noise_margin_db=args.noise_margin_db,
        power_window_resolution=args.power_window_resolution,
        patch_1d=args.patch_1d,
        patch_rows=args.patch_rows,
        patch_cols=args.patch_cols,
        angle_min_deg=args.angle_min_deg,
        angle_max_deg=args.angle_max_deg,
        angle_step_deg=args.angle_step_deg,
        k_factor_min_db=args.k_factor_min_db,
        k_factor_max_db=args.k_factor_max_db,
    )
    validate_options(cli_options)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.calibration_json:
        calibration = calibration_from_json(args.calibration_json)
        frozen_options = EstimatorOptions(**calibration.options)
        if frozen_options != cli_options:
            print(
                "warning=CLI estimator options are ignored because --calibration-json "
                "contains the frozen validation settings."
            )
        options = frozen_options
        validate_options(options)
    else:
        options = cli_options
        validation_samples = load_samples(args.validation_data, args.limit_validation)
        calibration = calibrate(
            validation_samples,
            threshold_candidates=parse_float_list(args.threshold_db_candidates),
            angle_modes=parse_str_list(args.angle_mode_candidates, ANGLE_MODES),
            options=options,
            calibrate_power=not args.disable_power_calibration,
        )

    with (output_dir / "calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(asdict(calibration)), handle, indent=2, ensure_ascii=False)

    test_samples = load_samples(args.test_data, args.limit_test)
    metrics = evaluate(test_samples, calibration, options, output_dir)
    run_config = {
        "validation_data": args.validation_data,
        "calibration_json_input": args.calibration_json,
        "test_data": args.test_data,
        "output_dir": str(output_dir),
        "limit_validation": args.limit_validation,
        "limit_test": args.limit_test,
        "estimator_options": asdict(options),
        "calibration": asdict(calibration),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(run_config), handle, indent=2, ensure_ascii=False)

    print(f"classical_baseline_output_dir={output_dir}")
    print(f"classical_baseline_test_samples={len(test_samples)}")
    for method, method_metrics in metrics.items():
        for field, values in method_metrics["numeric"].items():
            print(f"{method}_{field}_MAE={values['mae']:.6g}")
        print(
            f"{method}_los_nlos_accuracy="
            f"{method_metrics['los_nlos']['accuracy']:.6g}"
        )


if __name__ == "__main__":
    main()
