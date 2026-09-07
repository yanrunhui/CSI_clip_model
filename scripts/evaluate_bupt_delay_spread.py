from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def parse_thresholds(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("Thresholds must be positive comma-separated dB values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare checkpoint delay-spread predictions with classical RMS delay "
            "spread estimates from the measured BUPT mean PDP. The PDP estimate is "
            "a reference estimator, not laboratory ground truth."
        )
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--thresholds-db-below-peak",
        type=parse_thresholds,
        default=(15.0, 20.0, 25.0, 30.0),
    )
    parser.add_argument("--primary-threshold-db", type=float, default=25.0)
    parser.add_argument("--noise-margin-db", type=float, default=6.0)
    parser.add_argument("--peak-prominence-db", type=float, default=3.0)
    parser.add_argument("--max-excess-delay-ns", type=float, default=3000.0)
    return parser.parse_args()


def read_prediction_means(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    required = {"timestamp", "pred_delay_spread_ns"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = float(row["pred_delay_spread_ns"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite delay-spread prediction at timestamp {row['timestamp']}")
        grouped.setdefault(str(row["timestamp"]), []).append(value)
    return {
        timestamp: {
            "prediction_mean_ns": float(np.mean(values)),
            "prediction_std_ns": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "prediction_snapshots": float(len(values)),
        }
        for timestamp, values in grouped.items()
    }


def scalar(data: np.lib.npyio.NpzFile, name: str, default: float) -> float:
    if name not in data:
        return default
    values = np.asarray(data[name]).reshape(-1)
    return float(values[0]) if values.size else default


def rms_delay_spread_from_pdp(
    pdp: np.ndarray,
    sample_period_ns: float,
    threshold_db_below_peak: float,
    noise_margin_db: float,
    peak_prominence_db: float,
    max_excess_delay_ns: float,
    stored_noise_floor: float,
) -> dict[str, float | int]:
    pdp = np.asarray(pdp, dtype=np.float64).reshape(-1)
    if pdp.size == 0 or not np.isfinite(pdp).all() or np.max(pdp) <= 0.0:
        raise ValueError("PDP must be finite, non-empty, and contain positive power")
    peak_index = int(np.argmax(pdp))
    peak_power = float(pdp[peak_index])
    noise_floor = stored_noise_floor
    if not math.isfinite(noise_floor) or noise_floor < 0.0:
        tail_count = max(1, int(round(0.2 * len(pdp))))
        noise_floor = float(np.median(pdp[-tail_count:]))
    threshold = max(
        peak_power * 10.0 ** (-threshold_db_below_peak / 10.0),
        noise_floor * 10.0 ** (noise_margin_db / 10.0),
    )
    prominence = threshold * 10.0 ** (-peak_prominence_db / 10.0)
    peaks, _ = find_peaks(pdp, height=threshold, prominence=prominence)
    if pdp[0] >= threshold and pdp[0] >= pdp[1]:
        peaks = np.unique(np.concatenate([np.asarray([0]), peaks]))
    if peaks.size == 0:
        peaks = np.asarray([peak_index], dtype=np.int64)
    first_path_index = int(peaks[0])

    max_excess_bins = max(1, int(math.floor(max_excess_delay_ns / sample_period_ns)))
    stop = min(len(pdp), first_path_index + max_excess_bins + 1)
    indexes = np.arange(first_path_index, stop, dtype=np.int64)
    retained = pdp[indexes] >= threshold
    retained_indexes = indexes[retained]
    if retained_indexes.size == 0:
        retained_indexes = np.asarray([peak_index], dtype=np.int64)
    weights = np.maximum(pdp[retained_indexes] - noise_floor, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = pdp[retained_indexes]
    relative_delays_ns = (retained_indexes - first_path_index) * sample_period_ns
    power_sum = float(weights.sum())
    mean_delay_ns = float(np.sum(weights * relative_delays_ns) / power_sum)
    variance_ns2 = float(
        np.sum(weights * np.square(relative_delays_ns - mean_delay_ns)) / power_sum
    )
    retained_energy = float(np.sum(np.maximum(pdp[retained_indexes] - noise_floor, 0.0)))
    total_clean_energy = float(np.sum(np.maximum(pdp - noise_floor, 0.0)))
    return {
        "reference_delay_spread_ns": math.sqrt(max(variance_ns2, 0.0)),
        "mean_excess_delay_ns": mean_delay_ns,
        "first_path_tap": first_path_index,
        "peak_tap": peak_index,
        "detected_peak_count": int(np.count_nonzero((peaks >= first_path_index) & (peaks < stop))),
        "retained_bin_count": int(retained_indexes.size),
        "retained_clean_energy_fraction": (
            retained_energy / total_clean_energy if total_clean_energy > 0.0 else 0.0
        ),
        "noise_floor": noise_floor,
        "detection_threshold": threshold,
        "threshold_db_below_peak": threshold_db_below_peak,
    }


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    error = prediction - reference
    if len(reference) >= 2 and float(np.std(reference)) > 0.0 and float(np.std(prediction)) > 0.0:
        correlation = float(np.corrcoef(reference, prediction)[0, 1])
    else:
        correlation = float("nan")
    return {
        "mat_count": int(len(reference)),
        "reference_mean_ns": float(np.mean(reference)),
        "reference_std_ns": float(np.std(reference, ddof=1)) if len(reference) > 1 else 0.0,
        "prediction_mean_ns": float(np.mean(prediction)),
        "prediction_std_ns": float(np.std(prediction, ddof=1)) if len(prediction) > 1 else 0.0,
        "mae_ns": float(np.mean(np.abs(error))),
        "rmse_ns": float(np.sqrt(np.mean(np.square(error)))),
        "bias_ns": float(np.mean(error)),
        "pearson_r": correlation,
    }


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
    thresholds = args.thresholds_db_below_peak
    if not any(math.isclose(args.primary_threshold_db, value) for value in thresholds):
        raise ValueError("primary-threshold-db must be included in thresholds-db-below-peak")
    if args.noise_margin_db < 0.0 or args.peak_prominence_db < 0.0:
        raise ValueError("noise-margin-db and peak-prominence-db must be non-negative")
    if args.max_excess_delay_ns <= 0.0:
        raise ValueError("max-excess-delay-ns must be positive")

    prediction_means = read_prediction_means(args.predictions_csv)
    npz_paths = sorted(args.processed_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No NPZ files found under {args.processed_dir}")
    output_dir = args.output_dir or args.processed_dir / "delay_spread_evaluation"

    detail_rows: list[dict[str, object]] = []
    for path in npz_paths:
        timestamp = path.stem
        if timestamp not in prediction_means:
            raise ValueError(f"No model predictions found for NPZ timestamp {timestamp}")
        with np.load(path) as data:
            pdp = np.asarray(data["mean_pdp"], dtype=np.float64)
            sample_rate_mhz = scalar(data, "sampleRateMHz", 200.0)
            stored_noise_floor = scalar(data, "noise_floor", float("nan"))
        if sample_rate_mhz <= 0.0:
            raise ValueError(f"Invalid sampleRateMHz={sample_rate_mhz} in {path}")
        sample_period_ns = 1000.0 / sample_rate_mhz
        for threshold_db in thresholds:
            estimate = rms_delay_spread_from_pdp(
                pdp=pdp,
                sample_period_ns=sample_period_ns,
                threshold_db_below_peak=threshold_db,
                noise_margin_db=args.noise_margin_db,
                peak_prominence_db=args.peak_prominence_db,
                max_excess_delay_ns=args.max_excess_delay_ns,
                stored_noise_floor=stored_noise_floor,
            )
            prediction = prediction_means[timestamp]
            detail_rows.append(
                {
                    "timestamp": timestamp,
                    "source_npz": str(path),
                    "sample_period_ns": sample_period_ns,
                    **estimate,
                    **prediction,
                    "error_ns": prediction["prediction_mean_ns"]
                    - float(estimate["reference_delay_spread_ns"]),
                    "absolute_error_ns": abs(
                        prediction["prediction_mean_ns"]
                        - float(estimate["reference_delay_spread_ns"])
                    ),
                }
            )

    summary_rows: list[dict[str, object]] = []
    for threshold_db in thresholds:
        selected = [
            row
            for row in detail_rows
            if math.isclose(float(row["threshold_db_below_peak"]), threshold_db)
        ]
        reference = np.asarray(
            [float(row["reference_delay_spread_ns"]) for row in selected], dtype=np.float64
        )
        prediction = np.asarray(
            [float(row["prediction_mean_ns"]) for row in selected], dtype=np.float64
        )
        summary_rows.append(
            {
                "threshold_db_below_peak": threshold_db,
                "noise_margin_db": args.noise_margin_db,
                "peak_prominence_db": args.peak_prominence_db,
                "max_excess_delay_ns": args.max_excess_delay_ns,
                "is_primary": int(math.isclose(threshold_db, args.primary_threshold_db)),
                **metrics(reference, prediction),
            }
        )

    write_csv(output_dir / "delay_spread_per_mat.csv", detail_rows)
    write_csv(output_dir / "delay_spread_threshold_summary.csv", summary_rows)
    primary = next(row for row in summary_rows if row["is_primary"] == 1)
    with (output_dir / "delay_spread_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "status": "reference_estimator_comparison_not_ground_truth",
                "primary": primary,
                "threshold_sensitivity": summary_rows,
            },
            stream,
            indent=2,
            allow_nan=True,
        )

    print("Delay-spread reference comparison complete")
    print(f"Independent MAT samples: {primary['mat_count']}")
    print(f"Primary threshold: {args.primary_threshold_db:.1f} dB below peak")
    print(f"Reference mean: {primary['reference_mean_ns']:.3f} ns")
    print(f"Prediction mean: {primary['prediction_mean_ns']:.3f} ns")
    print(f"MAE: {primary['mae_ns']:.3f} ns")
    print(f"RMSE: {primary['rmse_ns']:.3f} ns")
    print(f"Bias (prediction-reference): {primary['bias_ns']:.3f} ns")
    print(f"Pearson r: {primary['pearson_r']:.6f}")
    print(f"Output directory: {output_dir.resolve()}")
    print("Important: PDP-derived values are classical reference estimates, not ground truth.")


if __name__ == "__main__":
    main()
