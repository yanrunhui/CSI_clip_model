from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DELAY_FIELDS = (
    "first_path_delay_ns",
    "los_delay_ns",
)

DELAY_BINS = (
    ("0_100", 0.0, 100.0),
    ("100_300", 100.0, 300.0),
    ("300_600", 300.0, 600.0),
    ("600_960", 600.0, 960.0),
    ("960_1920", 960.0, 1920.0),
)


def finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_records(path: Path) -> dict[str, tuple[dict, dict]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError(f"Payload has no non-empty comparisons list: {path}")

    records = {}
    for comparison in comparisons:
        group_id = str(comparison.get("group_id", "")).strip()
        if not group_id:
            raise ValueError(f"Comparison without group_id in {path}.")
        if group_id in records:
            raise ValueError(f"Duplicate group_id={group_id!r} in {path}.")
        predicted = comparison.get("predicted_record")
        target = comparison.get("target_record")
        if not isinstance(predicted, dict) or not isinstance(target, dict):
            raise ValueError(
                f"Comparison for group_id={group_id!r} is missing structured records."
            )
        records[group_id] = (predicted, target)
    return records


def circular_difference(
    value: torch.Tensor,
    reference: torch.Tensor,
    period: float,
) -> torch.Tensor:
    return torch.remainder(value - reference + period / 2.0, period) - period / 2.0


def robust_period_fusion(
    residue_a: torch.Tensor,
    residue_b: torch.Tensor,
    *,
    period_a_ns: float,
    period_b_ns: float,
    max_delay_ns: float,
    grid_step_ns: float,
    chunk_size: int,
) -> torch.Tensor:
    candidates = torch.arange(
        0.0,
        max_delay_ns,
        grid_step_ns,
        dtype=torch.float32,
    )
    outputs = []
    for start in range(0, residue_a.numel(), chunk_size):
        end = min(start + chunk_size, residue_a.numel())
        candidate_grid = candidates.unsqueeze(0)
        error_a = circular_difference(
            candidate_grid,
            residue_a[start:end].unsqueeze(1),
            period_a_ns,
        )
        error_b = circular_difference(
            candidate_grid,
            residue_b[start:end].unsqueeze(1),
            period_b_ns,
        )
        cost = error_a.square() + error_b.square()
        outputs.append(candidates[cost.argmin(dim=1)])
    return torch.cat(outputs, dim=0)


def safe_pearson(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    if predictions.numel() < 2:
        return math.nan
    prediction_centered = predictions.float() - predictions.float().mean()
    target_centered = targets.float() - targets.float().mean()
    denominator = (
        torch.linalg.vector_norm(prediction_centered)
        * torch.linalg.vector_norm(target_centered)
    )
    if float(denominator) <= 0.0:
        return 0.0
    return float((prediction_centered * target_centered).sum() / denominator)


def metric_rows(
    *,
    field: str,
    method: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    rows = []
    groups = [("all", torch.ones_like(targets, dtype=torch.bool))]
    for label, lower, upper in DELAY_BINS:
        groups.append((label, (targets >= lower) & (targets < upper)))

    for target_range, mask in groups:
        count = int(mask.sum().item())
        row: dict[str, float | int | str] = {
            "field": field,
            "method": method,
            "target_range": target_range,
            "count": count,
            "MAE": math.nan,
            "RMSE": math.nan,
            "signed_mean": math.nan,
            "pearson": math.nan,
            "accuracy_at_50ns": math.nan,
        }
        if count:
            prediction = predictions[mask].float()
            target = targets[mask].float()
            error = prediction - target
            row.update(
                {
                    "MAE": float(error.abs().mean()),
                    "RMSE": float(torch.sqrt(error.square().mean())),
                    "signed_mean": float(error.mean()),
                    "pearson": safe_pearson(prediction, target),
                    "accuracy_at_50ns": float((error.abs() <= 50.0).float().mean()),
                }
            )
        rows.append(row)
    return rows


def format_metric(value) -> str:
    parsed = finite_float(value)
    return "nan" if parsed is None else f"{parsed:.6g}"


def collect_field(
    *,
    field: str,
    records_a: dict[str, tuple[dict, dict]],
    records_b: dict[str, tuple[dict, dict]],
    shared_ids: list[str],
    target_tolerance_ns: float,
    max_delay_ns: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    predictions_a = []
    predictions_b = []
    targets = []
    missing_count = 0
    out_of_range_count = 0
    for group_id in shared_ids:
        predicted_a, target_a = records_a[group_id]
        predicted_b, target_b = records_b[group_id]
        prediction_a = finite_float(predicted_a.get(field))
        prediction_b = finite_float(predicted_b.get(field))
        target_value_a = finite_float(target_a.get(field))
        target_value_b = finite_float(target_b.get(field))
        if None in (prediction_a, prediction_b, target_value_a, target_value_b):
            missing_count += 1
            continue
        assert target_value_a is not None and target_value_b is not None
        if abs(target_value_a - target_value_b) > target_tolerance_ns:
            raise ValueError(
                f"Target mismatch for group_id={group_id!r}, field={field}: "
                f"A={target_value_a} B={target_value_b}."
            )
        if target_value_a < 0.0 or target_value_a >= max_delay_ns:
            out_of_range_count += 1
            continue
        predictions_a.append(float(prediction_a))
        predictions_b.append(float(prediction_b))
        targets.append(float(target_value_a))

    if not targets:
        raise ValueError(f"No valid paired targets for {field}.")
    return (
        torch.tensor(predictions_a, dtype=torch.float32),
        torch.tensor(predictions_b, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        missing_count,
        out_of_range_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-a", type=Path, required=True)
    parser.add_argument("--payload-b", type=Path, required=True)
    parser.add_argument("--name-a", default="nf64")
    parser.add_argument("--name-b", default="nf96")
    parser.add_argument("--period-a-ns", type=float, default=640.0)
    parser.add_argument("--period-b-ns", type=float, default=960.0)
    parser.add_argument("--max-delay-ns", type=float, default=1920.0)
    parser.add_argument("--grid-step-ns", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--target-tolerance-ns", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.period_a_ns <= 0.0 or args.period_b_ns <= 0.0:
        raise ValueError("Numerology periods must be positive.")
    if args.max_delay_ns <= 0.0 or args.grid_step_ns <= 0.0:
        raise ValueError("Delay range and grid step must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    records_a = load_records(args.payload_a)
    records_b = load_records(args.payload_b)
    shared_ids = sorted(set(records_a).intersection(records_b))
    if not shared_ids:
        raise ValueError("The payloads have no shared group IDs.")

    print(f"payload_a={args.payload_a}")
    print(f"payload_b={args.payload_b}")
    print(f"paired_group_count={len(shared_ids)}")
    print(f"period_a_ns={args.period_a_ns:g}")
    print(f"period_b_ns={args.period_b_ns:g}")
    print(f"fusion_range_ns=0,{args.max_delay_ns:g}")

    all_rows = []
    summary = {
        "payload_a": str(args.payload_a),
        "payload_b": str(args.payload_b),
        "name_a": args.name_a,
        "name_b": args.name_b,
        "period_a_ns": args.period_a_ns,
        "period_b_ns": args.period_b_ns,
        "max_delay_ns": args.max_delay_ns,
        "grid_step_ns": args.grid_step_ns,
        "paired_group_count": len(shared_ids),
        "fields": {},
    }

    for field in DELAY_FIELDS:
        prediction_a, prediction_b, target, missing_count, out_of_range_count = collect_field(
            field=field,
            records_a=records_a,
            records_b=records_b,
            shared_ids=shared_ids,
            target_tolerance_ns=args.target_tolerance_ns,
            max_delay_ns=args.max_delay_ns,
        )
        residue_a = torch.remainder(prediction_a, args.period_a_ns)
        residue_b = torch.remainder(prediction_b, args.period_b_ns)
        target_residue_a = torch.remainder(target, args.period_a_ns)
        target_residue_b = torch.remainder(target, args.period_b_ns)

        fused = robust_period_fusion(
            residue_a,
            residue_b,
            period_a_ns=args.period_a_ns,
            period_b_ns=args.period_b_ns,
            max_delay_ns=args.max_delay_ns,
            grid_step_ns=args.grid_step_ns,
            chunk_size=args.chunk_size,
        )
        oracle_fused = robust_period_fusion(
            target_residue_a,
            target_residue_b,
            period_a_ns=args.period_a_ns,
            period_b_ns=args.period_b_ns,
            max_delay_ns=args.max_delay_ns,
            grid_step_ns=args.grid_step_ns,
            chunk_size=args.chunk_size,
        )

        methods = {
            args.name_a: prediction_a,
            args.name_b: prediction_b,
            "period_fusion": fused,
            "oracle_period_fusion": oracle_fused,
        }
        field_rows = []
        for method, prediction in methods.items():
            rows = metric_rows(
                field=field,
                method=method,
                predictions=prediction,
                targets=target,
            )
            field_rows.extend(rows)
            overall = rows[0]
            print(f"{field}_{method}_count={overall['count']}")
            print(f"{field}_{method}_MAE={format_metric(overall['MAE'])}")
            print(f"{field}_{method}_RMSE={format_metric(overall['RMSE'])}")
            print(
                f"{field}_{method}_accuracy_at_50ns="
                f"{format_metric(overall['accuracy_at_50ns'])}"
            )
        all_rows.extend(field_rows)
        summary["fields"][field] = {
            "valid_count": target.numel(),
            "missing_count": missing_count,
            "out_of_range_count": out_of_range_count,
            "metrics": {
                row["method"]: row
                for row in field_rows
                if row["target_range"] == "all"
            },
        }
        print(f"{field}_missing_pair_count={missing_count}")
        print(f"{field}_outside_fusion_range_count={out_of_range_count}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "multinumerology_delay_fusion_metrics.csv"
    fieldnames = [
        "field",
        "method",
        "target_range",
        "count",
        "MAE",
        "RMSE",
        "signed_mean",
        "pearson",
        "accuracy_at_50ns",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = args.output_dir / "multinumerology_delay_fusion_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved_metrics_csv={csv_path}")
    print(f"saved_summary_json={json_path}")


if __name__ == "__main__":
    main()
