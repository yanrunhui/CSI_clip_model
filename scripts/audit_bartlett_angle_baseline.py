from __future__ import annotations

"""Audit whether the Bartlett angle baseline matches the paper's angle target.

The audit separates physical validity from numerical performance.  It checks
which link side supplies the CSI spatial aperture, which side the angle label
describes, whether stored array coordinates form the declared UPA plane, and
whether a two-dimensional azimuth/elevation Bartlett scan with a validation-
selected angular sidelobe threshold is available.

The current repository preprocessing contract treats the retained spatial
dimension as the BS/Tx array and selects one Rx element.  Its
``first_path_aoa_az_deg`` target is an Rx-side angle of arrival.  Those defaults
therefore intentionally produce a side-mismatch FAIL unless the raw channel
was generated in the reverse link and that fact is supplied explicitly with
evidence on the command line.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_classical_pdp_bartlett import (  # noqa: E402
    Calibration,
    EstimatorOptions,
    calibration_from_json,
    circular_angle_error_deg,
    circular_spread_deg,
    finite_float,
    load_samples,
    parse_float_list,
    prepare_csi,
    temporal_estimate,
)


STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_UNKNOWN = "UNKNOWN"


def status_record(status: str, explanation: str, **details: Any) -> dict[str, Any]:
    return {"status": status, "explanation": explanation, **details}


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


def infer_spatial_role(samples: list[Any], requested: str) -> tuple[str, str]:
    if requested != "auto":
        return requested, "explicit_command_line"
    persisted = {
        str(getattr(sample, "spatial_array_role", "")).strip().lower()
        for sample in samples
        if str(getattr(sample, "spatial_array_role", "")).strip()
    }
    if len(persisted) == 1:
        return next(iter(persisted)), "persisted_sample_metadata"
    if len(persisted) > 1:
        return "unknown", "conflicting_persisted_sample_metadata"
    # Contract in data/preprocess.py + scripts/preprocess_all.py:
    # raw CSI is documented as [N_rx, N_tx, N_f], one Rx is selected, and the
    # remaining array geometry is obtained from ch_params['bs_antenna'].
    return "tx", "current_repository_preprocessing_contract"


def infer_target_role(samples: list[Any], requested: str) -> tuple[str, str]:
    if requested != "auto":
        return requested, "explicit_command_line"
    persisted = {
        str(getattr(sample, "first_path_angle_role", "")).strip().lower()
        for sample in samples
        if str(getattr(sample, "first_path_angle_role", "")).strip()
    }
    if len(persisted) == 1:
        return next(iter(persisted)), "persisted_sample_metadata"
    if len(persisted) > 1:
        return "unknown", "conflicting_persisted_sample_metadata"
    if any(hasattr(sample, "first_path_aoa_az_deg") for sample in samples):
        return "rx", "target_field_name:first_path_aoa_az_deg"
    if any(hasattr(sample, "first_path_aod_az_deg") for sample in samples):
        return "tx", "target_field_name:first_path_aod_az_deg"
    return "unknown", "no_angle_role_metadata_or_recognized_target_field"


def role_check(
    spatial_role: str,
    spatial_source: str,
    target_role: str,
    target_source: str,
    role_evidence: str | None,
) -> dict[str, Any]:
    details = {
        "csi_spatial_role": spatial_role,
        "csi_spatial_role_source": spatial_source,
        "target_angle_role": target_role,
        "target_angle_role_source": target_source,
        "external_evidence": role_evidence or "",
    }
    if "unknown" in {spatial_role, target_role}:
        return status_record(
            STATUS_UNKNOWN,
            "At least one link side is unknown; AoA/AoD correspondence is unverified.",
            **details,
        )
    if spatial_role != target_role:
        return status_record(
            STATUS_FAIL,
            "The CSI spatial aperture and target angle refer to different link sides.",
            **details,
        )
    if spatial_source == "explicit_command_line" and not role_evidence:
        return status_record(
            STATUS_UNKNOWN,
            "The roles match only by an unsupported command-line assertion; provide --role-evidence.",
            **details,
        )
    return status_record(
        STATUS_PASS,
        "The CSI spatial aperture and angle target refer to the same link side.",
        **details,
    )


def numerical_plane(coordinates: np.ndarray, tolerance: float = 1.0e-8) -> str:
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    active = np.ptp(centered, axis=0) > tolerance
    active_axes = "".join(axis for axis, keep in zip("xyz", active) if keep)
    return active_axes if active_axes in {"x", "y", "z", "xy", "xz", "yz", "xyz"} else "unknown"


def coordinate_audit(
    samples: list[Any],
    physical_array_plane: str,
    coordinate_evidence: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_configuration: dict[str, Any] = {}
    geometry_ok = True
    declared_plane_ok = True
    observed_planes: set[str] = set()
    for sample in samples:
        config_key = str(getattr(sample, "config_key", "unknown"))
        if config_key in per_configuration:
            continue
        rows = int(getattr(sample, "array_rows", 0) or 0)
        cols = int(getattr(sample, "array_cols", 0) or 0)
        coordinates = getattr(sample, "antenna_coordinates_wavelengths", None)
        if hasattr(coordinates, "detach"):
            coordinates = coordinates.detach().cpu().numpy()
        coordinates = np.asarray(coordinates, dtype=np.float64)
        expected_shape = (rows * cols, 3)
        shape_ok = coordinates.shape == expected_shape and rows > 0 and cols > 0
        finite_ok = bool(np.isfinite(coordinates).all()) if shape_ok else False
        unique_count = (
            int(np.unique(np.round(coordinates, decimals=9), axis=0).shape[0])
            if shape_ok
            else 0
        )
        unique_ok = unique_count == rows * cols and rows * cols > 0
        rank = (
            int(np.linalg.matrix_rank(coordinates - coordinates.mean(axis=0)))
            if shape_ok
            else 0
        )
        expected_rank = 1 if str(getattr(sample, "array_type", "")).upper() == "ULA" else 2
        rank_ok = rank == expected_rank
        plane = numerical_plane(coordinates) if shape_ok else "unknown"
        observed_planes.add(plane)
        spacing_metadata = finite_float(
            getattr(sample, "antenna_spacing_wavelengths", math.nan)
        )
        pairwise = None
        spacing_observed = math.nan
        if shape_ok and coordinates.shape[0] > 1:
            differences = coordinates[:, None, :] - coordinates[None, :, :]
            distances = np.linalg.norm(differences, axis=-1)
            pairwise = distances[distances > 1.0e-8]
            if pairwise.size:
                spacing_observed = float(np.min(pairwise))
        spacing_ok = (
            spacing_metadata is not None
            and math.isfinite(spacing_observed)
            and math.isclose(spacing_metadata, spacing_observed, rel_tol=1.0e-4, abs_tol=1.0e-6)
        )
        config_ok = shape_ok and finite_ok and unique_ok and rank_ok and spacing_ok
        geometry_ok &= config_ok
        if physical_array_plane != "unknown":
            declared_plane_ok &= plane == physical_array_plane
        per_configuration[config_key] = {
            "array_type": str(getattr(sample, "array_type", "")),
            "rows": rows,
            "cols": cols,
            "coordinate_shape": list(coordinates.shape),
            "coordinate_rank": rank,
            "expected_rank": expected_rank,
            "stored_numerical_plane": plane,
            "unique_coordinates": unique_count,
            "spacing_metadata_wavelengths": spacing_metadata,
            "spacing_observed_wavelengths": spacing_observed,
            "geometry_checks_pass": config_ok,
        }

    geometry_check = status_record(
        STATUS_PASS if geometry_ok else STATUS_FAIL,
        (
            "Stored coordinate count, rank, uniqueness, and spacing match the array metadata."
            if geometry_ok
            else "At least one stored array geometry is inconsistent with its metadata."
        ),
        configurations=per_configuration,
    )
    if physical_array_plane == "unknown":
        convention_check = status_record(
            STATUS_UNKNOWN,
            "The numerical coordinate plane is visible, but its physical azimuth/elevation axis convention was not supplied.",
            observed_numerical_planes=sorted(observed_planes),
            physical_array_plane="unknown",
            external_evidence=coordinate_evidence or "",
        )
    elif not coordinate_evidence:
        convention_check = status_record(
            STATUS_UNKNOWN,
            "A physical plane was asserted without --coordinate-evidence.",
            observed_numerical_planes=sorted(observed_planes),
            physical_array_plane=physical_array_plane,
            external_evidence="",
        )
    elif not declared_plane_ok:
        convention_check = status_record(
            STATUS_FAIL,
            "The declared physical plane does not match the stored coordinate plane.",
            observed_numerical_planes=sorted(observed_planes),
            physical_array_plane=physical_array_plane,
            external_evidence=coordinate_evidence,
        )
    else:
        convention_check = status_record(
            STATUS_PASS,
            "The declared physical array plane matches the stored coordinates and has external provenance.",
            observed_numerical_planes=sorted(observed_planes),
            physical_array_plane=physical_array_plane,
            external_evidence=coordinate_evidence,
        )
    return geometry_check, convention_check


def spherical_directions(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    azimuth = np.deg2rad(azimuth_deg)[None, :]
    elevation = np.deg2rad(elevation_deg)[:, None]
    cos_elevation = np.cos(elevation)
    x = cos_elevation * np.cos(azimuth)
    y = cos_elevation * np.sin(azimuth)
    z = np.sin(elevation) * np.ones_like(azimuth)
    return np.stack([x, y, z], axis=-1)


def steering_2d(
    coordinates: np.ndarray,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    sign: int,
) -> np.ndarray:
    directions = spherical_directions(azimuth_deg, elevation_deg)
    phase = 2.0 * np.pi * np.einsum("eac,nc->ean", directions, coordinates)
    return np.exp(1j * float(sign) * phase)


def marginalize_elevation(power: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return np.max(power, axis=0)
    if mode == "sum":
        return np.sum(power, axis=0)
    raise ValueError(f"Unsupported elevation marginalization mode={mode!r}.")


def thresholded_spread(
    azimuth_deg: np.ndarray,
    spectrum: np.ndarray,
    threshold_db: float,
) -> float:
    spectrum = np.maximum(np.asarray(spectrum, dtype=np.float64), 0.0)
    peak = float(np.max(spectrum))
    if peak <= 0.0:
        return math.nan
    floor = peak * 10.0 ** (-float(threshold_db) / 10.0)
    weights = np.where(spectrum >= floor, spectrum, 0.0)
    return circular_spread_deg(azimuth_deg, weights)


def spectrum_for_sample(
    prepared,
    temporal,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    sign: int,
    marginalization: str,
    cache: dict[tuple[Any, ...], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    key = (
        prepared.array_type,
        prepared.coordinates_wavelengths.shape,
        prepared.coordinates_wavelengths.tobytes(),
        sign,
        azimuth_deg.tobytes(),
        elevation_deg.tobytes(),
    )
    steering = cache.get(key)
    if steering is None:
        steering = steering_2d(
            prepared.coordinates_wavelengths,
            azimuth_deg,
            elevation_deg,
            sign,
        )
        cache[key] = steering
    flattened = steering.reshape(-1, steering.shape[-1])
    first_snapshot = prepared.antenna_delay_response[:, temporal.first_peak_index]
    first_power = np.abs(flattened.conj() @ first_snapshot) ** 2
    first_power_2d = first_power.reshape(elevation_deg.size, azimuth_deg.size)
    first_azimuth_spectrum = marginalize_elevation(first_power_2d, marginalization)

    path_snapshots = prepared.antenna_delay_response[:, temporal.peak_indices]
    path_power = np.sum(np.abs(flattened.conj() @ path_snapshots) ** 2, axis=1)
    path_power_2d = path_power.reshape(elevation_deg.size, azimuth_deg.size)
    spread_azimuth_spectrum = marginalize_elevation(path_power_2d, marginalization)
    elevation_index, _ = np.unravel_index(int(np.argmax(first_power_2d)), first_power_2d.shape)
    return (
        first_azimuth_spectrum,
        spread_azimuth_spectrum,
        float(elevation_deg[elevation_index]),
    )


def metric_summary(errors: list[float]) -> dict[str, Any]:
    if not errors:
        return {"count": 0, "mae": math.nan, "median": math.nan, "p90": math.nan}
    values = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(values.size),
        "mae": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def run_two_dimensional_audit(
    samples: list[Any],
    calibration: Calibration,
    options: EstimatorOptions,
    azimuth_step_deg: float,
    elevation_min_deg: float,
    elevation_max_deg: float,
    elevation_step_deg: float,
    marginalization: str,
    angular_thresholds_db: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    azimuth_deg = np.arange(-180.0, 180.0, azimuth_step_deg, dtype=np.float64)
    elevation_deg = np.arange(
        elevation_min_deg,
        elevation_max_deg + elevation_step_deg * 0.5,
        elevation_step_deg,
        dtype=np.float64,
    )
    if azimuth_deg.size < 2 or elevation_deg.size < 2:
        raise ValueError("The 2D Bartlett grid must contain at least two bins per axis.")

    cache: dict[tuple[Any, ...], np.ndarray] = {}
    by_sign: dict[int, dict[str, Any]] = {
        sign: {
            "first_errors": [],
            "spread_errors": {threshold: [] for threshold in angular_thresholds_db},
            "predictions": [],
            "elevation_edges": 0,
            "valid_elevations": 0,
        }
        for sign in (-1, 1)
    }
    for index, sample in enumerate(samples):
        prepared = prepare_csi(sample, options)
        temporal = temporal_estimate(prepared, calibration.pdp_threshold_db, options)
        true_angle = finite_float(getattr(sample, "first_path_aoa_az_deg", math.nan))
        true_spread = finite_float(getattr(sample, "azimuth_spread_deg", math.nan))
        for sign in (-1, 1):
            first_spectrum, spread_spectrum, estimated_elevation = spectrum_for_sample(
                prepared,
                temporal,
                azimuth_deg,
                elevation_deg,
                sign,
                marginalization,
                cache,
            )
            predicted_angle = float(azimuth_deg[int(np.argmax(first_spectrum))])
            first_error = (
                circular_angle_error_deg(predicted_angle, true_angle)
                if true_angle is not None
                else math.nan
            )
            if math.isfinite(first_error):
                by_sign[sign]["first_errors"].append(first_error)
            by_sign[sign]["valid_elevations"] += 1
            if math.isclose(estimated_elevation, elevation_deg[0]) or math.isclose(
                estimated_elevation, elevation_deg[-1]
            ):
                by_sign[sign]["elevation_edges"] += 1
            spread_predictions = {}
            for threshold in angular_thresholds_db:
                predicted_spread = thresholded_spread(
                    azimuth_deg,
                    spread_spectrum,
                    threshold,
                )
                spread_predictions[threshold] = predicted_spread
                if true_spread is not None and math.isfinite(predicted_spread):
                    by_sign[sign]["spread_errors"][threshold].append(
                        abs(predicted_spread - true_spread)
                    )
            by_sign[sign]["predictions"].append(
                {
                    "index": index,
                    "group_id": str(getattr(sample, "group_id", "")),
                    "config_key": str(getattr(sample, "config_key", "")),
                    "steering_sign": sign,
                    "true_first_angle_deg": true_angle,
                    "predicted_first_angle_deg": predicted_angle,
                    "first_angle_error_deg": first_error,
                    "estimated_elevation_deg": estimated_elevation,
                    "true_angle_spread_deg": true_spread,
                    "spread_predictions": spread_predictions,
                }
            )

    sign_metrics = {
        sign: metric_summary(values["first_errors"])
        for sign, values in by_sign.items()
    }
    selected_sign = min(
        (-1, 1),
        key=lambda sign: (sign_metrics[sign]["mae"], sign),
    )
    selected = by_sign[selected_sign]
    threshold_rows = []
    best_threshold: tuple[float, float] | None = None
    for threshold in angular_thresholds_db:
        summary = metric_summary(selected["spread_errors"][threshold])
        row = {"threshold_db": threshold, **summary}
        threshold_rows.append(row)
        score = (summary["mae"], threshold)
        if best_threshold is None or score < best_threshold:
            best_threshold = score
    if best_threshold is None or not math.isfinite(best_threshold[0]):
        selected_threshold = math.nan
    else:
        selected_threshold = best_threshold[1]

    prediction_rows = []
    for row in selected["predictions"]:
        threshold_key = selected_threshold
        prediction_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "spread_predictions"
            }
            | {
                "selected_angular_threshold_db": selected_threshold,
                "predicted_angle_spread_deg": (
                    row["spread_predictions"].get(threshold_key, math.nan)
                    if math.isfinite(selected_threshold)
                    else math.nan
                ),
            }
        )
    edge_rate = selected["elevation_edges"] / max(selected["valid_elevations"], 1)
    diagnostics = {
        "scan": {
            "azimuth_range_deg": [-180.0, 180.0],
            "azimuth_step_deg": azimuth_step_deg,
            "elevation_range_deg": [elevation_min_deg, elevation_max_deg],
            "elevation_step_deg": elevation_step_deg,
            "elevation_marginalization": marginalization,
        },
        "steering_sign_candidates": {
            str(sign): sign_metrics[sign] for sign in (-1, 1)
        },
        "selected_steering_sign": selected_sign,
        "first_path_angle": sign_metrics[selected_sign],
        "angular_threshold_candidates": threshold_rows,
        "selected_angular_threshold_db": selected_threshold,
        "selected_angle_spread": (
            next(
                (row for row in threshold_rows if row["threshold_db"] == selected_threshold),
                metric_summary([]),
            )
        ),
        "estimated_elevation_boundary_rate": edge_rate,
        "validation_samples": len(samples),
    }
    return diagnostics, threshold_rows, prediction_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit physical and numerical validity of the Bartlett angle baseline."
    )
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--spatial-array-role",
        choices=("auto", "tx", "rx", "unknown"),
        default="auto",
    )
    parser.add_argument(
        "--target-angle-role",
        choices=("auto", "tx", "rx", "unknown"),
        default="auto",
    )
    parser.add_argument(
        "--role-evidence",
        help="Config/manual citation supporting an explicitly overridden link-side role.",
    )
    parser.add_argument(
        "--physical-array-plane",
        choices=("unknown", "xy", "xz", "yz"),
        default="unknown",
    )
    parser.add_argument(
        "--coordinate-evidence",
        help="Config/manual citation defining physical axes and azimuth/elevation convention.",
    )
    parser.add_argument("--azimuth-step-deg", type=float, default=2.0)
    parser.add_argument("--elevation-min-deg", type=float, default=-90.0)
    parser.add_argument("--elevation-max-deg", type=float, default=90.0)
    parser.add_argument("--elevation-step-deg", type=float, default=10.0)
    parser.add_argument(
        "--elevation-marginalization",
        choices=("max", "sum"),
        default="max",
    )
    parser.add_argument(
        "--angular-threshold-db-candidates",
        default="6,10,15,20,25,30",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.azimuth_step_deg <= 0.0 or args.elevation_step_deg <= 0.0:
        raise ValueError("Angular grid steps must be positive.")
    if args.elevation_min_deg >= args.elevation_max_deg:
        raise ValueError("Invalid elevation range.")

    samples = load_samples(args.validation_data, args.limit)
    calibration = calibration_from_json(args.calibration_json)
    options = EstimatorOptions(**calibration.options)
    spatial_role, spatial_source = infer_spatial_role(samples, args.spatial_array_role)
    target_role, target_source = infer_target_role(samples, args.target_angle_role)
    side_check = role_check(
        spatial_role,
        spatial_source,
        target_role,
        target_source,
        args.role_evidence,
    )
    geometry_check, coordinate_check = coordinate_audit(
        samples,
        args.physical_array_plane,
        args.coordinate_evidence,
    )
    array_types = Counter(str(getattr(sample, "array_type", "unknown")) for sample in samples)
    upa_only = set(array_types) == {"UPA"}
    array_type_check = status_record(
        STATUS_PASS if upa_only else STATUS_FAIL,
        (
            "Every audited sample uses a UPA, so a 2D scan is applicable."
            if upa_only
            else "The audit set contains a non-UPA geometry; one shared 2D UPA interpretation is invalid."
        ),
        histogram=dict(array_types),
    )

    two_d, threshold_rows, prediction_rows = run_two_dimensional_audit(
        samples,
        calibration,
        options,
        azimuth_step_deg=args.azimuth_step_deg,
        elevation_min_deg=args.elevation_min_deg,
        elevation_max_deg=args.elevation_max_deg,
        elevation_step_deg=args.elevation_step_deg,
        marginalization=args.elevation_marginalization,
        angular_thresholds_db=parse_float_list(args.angular_threshold_db_candidates),
    )
    threshold_valid = math.isfinite(float(two_d["selected_angular_threshold_db"]))
    two_d_check = status_record(
        STATUS_PASS if threshold_valid else STATUS_FAIL,
        (
            "A 2D azimuth/elevation scan ran and a sidelobe threshold was selected on validation labels."
            if threshold_valid
            else "The 2D scan did not yield a finite validation-selected angular threshold."
        ),
        **two_d,
    )

    checks = {
        "csi_and_target_same_link_side": side_check,
        "stored_array_geometry": geometry_check,
        "physical_coordinate_convention": coordinate_check,
        "upa_two_dimensional_applicability": array_type_check,
        "two_dimensional_scan_and_sidelobe_threshold": two_d_check,
    }
    blocking_checks = [
        "csi_and_target_same_link_side",
        "stored_array_geometry",
        "physical_coordinate_convention",
        "upa_two_dimensional_applicability",
        "two_dimensional_scan_and_sidelobe_threshold",
    ]
    blocking_statuses = {name: checks[name]["status"] for name in blocking_checks}
    paper_eligible = all(status == STATUS_PASS for status in blocking_statuses.values())
    if paper_eligible:
        conclusion = (
            "The Bartlett angle baseline is physically aligned and numerically specified. "
            "Freeze the reported sign and angular threshold before test evaluation."
        )
    elif STATUS_FAIL in blocking_statuses.values():
        conclusion = (
            "Do not report the Bartlett errors as an estimator of the paper's target angle: "
            "at least one required physical/numerical check failed."
        )
    else:
        conclusion = (
            "Do not report the Bartlett errors yet: required physical provenance remains unknown."
        )

    report = {
        "paper_eligible": paper_eligible,
        "conclusion": conclusion,
        "blocking_statuses": blocking_statuses,
        "checks": checks,
        "inputs": {
            "validation_data": args.validation_data,
            "calibration_json": args.calibration_json,
            "audited_samples": len(samples),
            "estimator_options": asdict(options),
        },
        "frozen_2d_settings_for_test": {
            "steering_sign": two_d["selected_steering_sign"],
            "angular_threshold_db": two_d["selected_angular_threshold_db"],
            "azimuth_step_deg": args.azimuth_step_deg,
            "elevation_min_deg": args.elevation_min_deg,
            "elevation_max_deg": args.elevation_max_deg,
            "elevation_step_deg": args.elevation_step_deg,
            "elevation_marginalization": args.elevation_marginalization,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "bartlett_angle_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, ensure_ascii=False)
    write_csv(output_dir / "angular_threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "validation_angle_predictions.csv", prediction_rows)

    print(f"bartlett_angle_paper_eligible={str(paper_eligible).lower()}")
    for name, status in blocking_statuses.items():
        print(f"bartlett_angle_check_{name}={status}")
    print(f"bartlett_2d_first_angle_MAE={two_d['first_path_angle']['mae']:.6g}")
    print(
        "bartlett_2d_angle_spread_MAE="
        f"{two_d['selected_angle_spread']['mae']:.6g}"
    )
    print(f"bartlett_angle_audit_report={output_dir / 'bartlett_angle_audit.json'}")


if __name__ == "__main__":
    main()
