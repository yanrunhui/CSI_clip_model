from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from diagnose_bupt_beam_scan import (
    extract_sequences,
    materialized_mat,
    read_manifest,
    read_zip_sources,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fold consecutive BUPT CIR snapshots by a candidate scan period and test "
            "whether each within-cycle position is repeatable and distinct."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--subset-manifest", type=Path)
    source.add_argument("--cir-zip-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--period", type=int, default=56)
    parser.add_argument("--max-mats", type=int, default=20)
    parser.add_argument("--sample-mode", choices=("uniform", "first"), default="first")
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--feature-bins", type=int, default=64)
    parser.add_argument("--chunk-columns", type=int, default=128)
    parser.add_argument(
        "--comparison-radius",
        type=int,
        default=8,
        help="Also evaluate candidate periods in period +/- this radius.",
    )
    return parser.parse_args()


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0


def eta_squared(matrix: np.ndarray) -> float:
    """Fraction of power variance explained by position within the scan cycle."""
    grand_mean = float(matrix.mean())
    state_means = matrix.mean(axis=0)
    between = matrix.shape[0] * float(np.square(state_means - grand_mean).sum())
    total = float(np.square(matrix - grand_mean).sum())
    return between / total if total > 0.0 else 0.0


def evaluate_period(
    features: np.ndarray,
    power_db: np.ndarray,
    period: int,
) -> dict[str, float | int]:
    cycle_count = len(features) // period
    if cycle_count < 3:
        raise ValueError(f"Period {period} leaves fewer than three complete cycles")
    usable = cycle_count * period
    feature_cycles = features[:usable].reshape(cycle_count, period, features.shape[1])
    power_cycles = power_db[:usable].reshape(cycle_count, period)

    # Compare corresponding positions in adjacent cycles.
    same_pdp = np.einsum(
        "cpf,cpf->cp", feature_cycles[:-1], feature_cycles[1:]
    )
    same_power_correlation = np.asarray(
        [safe_correlation(power_cycles[index], power_cycles[index + 1])
         for index in range(cycle_count - 1)],
        dtype=np.float64,
    )

    # Wrong-position controls: cyclically shift the following cycle by 1..min(8,P/2).
    max_shift = min(8, period // 2)
    wrong_pdp_values: list[float] = []
    wrong_power_values: list[float] = []
    for shift in range(1, max_shift + 1):
        shifted_features = np.roll(feature_cycles[1:], shift=shift, axis=1)
        wrong_pdp_values.append(float(np.einsum(
            "cpf,cpf->", feature_cycles[:-1], shifted_features
        ) / ((cycle_count - 1) * period)))
        shifted_power = np.roll(power_cycles[1:], shift=shift, axis=1)
        wrong_power_values.extend(
            safe_correlation(power_cycles[index], shifted_power[index])
            for index in range(cycle_count - 1)
        )

    mean_same_pdp = float(same_pdp.mean())
    mean_wrong_pdp = float(np.mean(wrong_pdp_values))
    mean_same_power = float(same_power_correlation.mean())
    mean_wrong_power = float(np.mean(wrong_power_values))
    return {
        "period_snapshots": period,
        "complete_cycles": cycle_count,
        "used_snapshots": usable,
        "pdp_same_position_cosine": mean_same_pdp,
        "pdp_wrong_position_cosine": mean_wrong_pdp,
        "pdp_position_margin": mean_same_pdp - mean_wrong_pdp,
        "power_cycle_correlation": mean_same_power,
        "power_wrong_shift_correlation": mean_wrong_power,
        "power_position_margin": mean_same_power - mean_wrong_power,
        "power_state_eta_squared": eta_squared(power_cycles),
    }


def per_state_rows(
    timestamp: str,
    source_label: str,
    features: np.ndarray,
    power_db: np.ndarray,
    period: int,
) -> list[dict[str, object]]:
    cycle_count = len(features) // period
    usable = cycle_count * period
    feature_cycles = features[:usable].reshape(cycle_count, period, features.shape[1])
    power_cycles = power_db[:usable].reshape(cycle_count, period)
    prototypes = feature_cycles.mean(axis=0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    similarity = np.einsum("cpf,pf->cp", feature_cycles, prototypes)
    return [
        {
            "timestamp": timestamp,
            "source_mat": source_label,
            "state_index": state,
            "cycle_count": cycle_count,
            "mean_power_db": float(power_cycles[:, state].mean()),
            "std_power_db": float(power_cycles[:, state].std()),
            "mean_pdp_to_state_prototype_cosine": float(similarity[:, state].mean()),
            "std_pdp_to_state_prototype_cosine": float(similarity[:, state].std()),
        }
        for state in range(period)
    ]


def main() -> None:
    args = parse_args()
    if args.period < 2 or args.comparison_radius < 1:
        raise ValueError("period must be >=2 and comparison-radius must be positive")
    if args.subset_manifest is not None:
        rows = read_manifest(args.subset_manifest, args.max_mats)
    else:
        rows = read_zip_sources(args.cir_zip_dir, args.max_mats, args.sample_mode)

    candidate_periods = range(
        max(2, args.period - args.comparison_radius),
        args.period + args.comparison_radius + 1,
    )
    per_mat: list[dict[str, object]] = []
    per_state: list[dict[str, object]] = []
    comparison: list[dict[str, object]] = []

    for index, row in enumerate(rows):
        with materialized_mat(row, args.temp_dir) as (path, source_label):
            features, power_db, time_gap = extract_sequences(
                path, args.feature_bins, args.chunk_columns
            )
        primary = evaluate_period(features, power_db, args.period)
        per_mat.append({
            "timestamp": row["timestamp"],
            "source_mat": source_label,
            "cir_time_gap_seconds": time_gap,
            **primary,
        })
        per_state.extend(per_state_rows(
            row["timestamp"], source_label, features, power_db, args.period
        ))
        for period in candidate_periods:
            comparison.append({
                "timestamp": row["timestamp"],
                "source_mat": source_label,
                **evaluate_period(features, power_db, period),
            })
        print(f"[{index + 1:03d}/{len(rows):03d}] folded {Path(source_label).name}")

    numeric_fields = [
        "pdp_same_position_cosine",
        "pdp_wrong_position_cosine",
        "pdp_position_margin",
        "power_cycle_correlation",
        "power_wrong_shift_correlation",
        "power_position_margin",
        "power_state_eta_squared",
    ]
    aggregate_periods: list[dict[str, object]] = []
    for period in candidate_periods:
        selected = [row for row in comparison if row["period_snapshots"] == period]
        aggregate_periods.append({
            "period_snapshots": period,
            "mat_count": len(selected),
            **{
                f"median_{field}": float(np.median([float(row[field]) for row in selected]))
                for field in numeric_fields
            },
        })

    primary_aggregate = next(
        row for row in aggregate_periods if row["period_snapshots"] == args.period
    )
    best_pdp = max(aggregate_periods, key=lambda row: float(row["median_pdp_position_margin"]))
    best_power = max(aggregate_periods, key=lambda row: float(row["median_power_position_margin"]))
    pdp_rank = 1 + sum(
        float(row["median_pdp_position_margin"])
        > float(primary_aggregate["median_pdp_position_margin"])
        for row in aggregate_periods
    )
    power_rank = 1 + sum(
        float(row["median_power_position_margin"])
        > float(primary_aggregate["median_power_position_margin"])
        for row in aggregate_periods
    )
    positive_mats = sum(
        float(row["pdp_position_margin"]) > 0.0
        and float(row["power_position_margin"]) > 0.0
        for row in per_mat
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "scan_state_per_mat.csv", per_mat)
    write_csv(args.output_dir / "scan_state_per_position.csv", per_state)
    write_csv(args.output_dir / "candidate_period_comparison.csv", aggregate_periods)
    report = {
        "candidate_period": args.period,
        "mat_count": len(rows),
        "positive_margin_mat_count": positive_mats,
        "positive_margin_fraction": positive_mats / len(rows),
        "candidate_metrics": primary_aggregate,
        "candidate_pdp_margin_rank": pdp_rank,
        "candidate_power_margin_rank": power_rank,
        "tested_period_count": len(aggregate_periods),
        "best_pdp_margin_period": best_pdp,
        "best_power_margin_period": best_power,
        "interpretation": (
            "Positive same-position margins and a sharp optimum near 56 support "
            "repeatable, distinct acquisition states. They still do not map state "
            "indices to physical angles without the beam codebook."
        ),
    }
    with (args.output_dir / "scan_state_validation.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(f"Candidate period: {args.period}")
    print(f"Positive PDP+power margins: {positive_mats}/{len(rows)}")
    print(f"PDP margin rank among nearby periods: {pdp_rank}/{len(aggregate_periods)}")
    print(f"Power margin rank among nearby periods: {power_rank}/{len(aggregate_periods)}")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
