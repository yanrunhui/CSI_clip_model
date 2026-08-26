from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CATEGORICAL_FIELDS = ("environment", "los_status")

NUMERIC_FIELDS = (
    "path_count",
    "first_path_delay_ns",
    "first_path_angle_deg",
    "first_path_power_dbw",
    "k_factor_db",
    "delay_spread_ns",
    "angle_spread_deg",
    "los_delay_ns",
    "los_angle_deg",
    "reflection_count",
    "reflection_path_count",
)

ANGLE_FIELDS = {"first_path_angle_deg", "los_angle_deg"}
LOS_ONLY_NUMERIC_FIELDS = {"los_delay_ns", "los_angle_deg"}
DELAY_NUMERIC_FIELDS = {"first_path_delay_ns", "los_delay_ns"}

# Labels that represent the same physical delay can differ slightly after
# floating-point serialization. Keep the ordering diagnostic insensitive to
# sub-picosecond numerical noise.
RELATIONAL_DELAY_EPSILON_NS = 1.0e-3

DEFAULT_TOLERANCES = {
    "path_count": 1.0,
    "first_path_delay_ns": 50.0,
    "first_path_angle_deg": 15.0,
    "first_path_power_dbw": 5.0,
    "k_factor_db": 3.0,
    "delay_spread_ns": 50.0,
    "angle_spread_deg": 15.0,
    "los_delay_ns": 50.0,
    "los_angle_deg": 15.0,
    "reflection_count": 1.0,
    "reflection_path_count": 1.0,
}


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def numeric_error(field: str, prediction: float, target: float) -> float:
    if field in ANGLE_FIELDS:
        diff = math.radians(prediction - target)
        return abs(math.degrees(math.atan2(math.sin(diff), math.cos(diff))))
    return abs(prediction - target)


def numeric_field_applicable(field: str, record: dict[str, Any]) -> bool:
    if field in LOS_ONLY_NUMERIC_FIELDS:
        return str(record.get("los_status", "")).lower() == "los"
    return True


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else math.nan


def safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else math.nan


def f1(precision: float, recall: float) -> float:
    if not math.isfinite(precision) or not math.isfinite(recall):
        return math.nan
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def categorical_macro_f1(predictions: list[str], targets: list[str]) -> float:
    labels = sorted(set(predictions) | set(targets))
    if not labels:
        return math.nan
    values = []
    for label in labels:
        tp = sum(pred == label and target == label for pred, target in zip(predictions, targets))
        fp = sum(pred == label and target != label for pred, target in zip(predictions, targets))
        fn = sum(pred != label and target == label for pred, target in zip(predictions, targets))
        denominator = 2 * tp + fp + fn
        values.append(2.0 * tp / denominator if denominator else 0.0)
    return mean(values)


def mean_finite(values: list[float]) -> float:
    return mean([value for value in values if math.isfinite(value)])


def format_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.6g}"


def metric_role(metric: str) -> str:
    if metric.startswith("physical_consistency"):
        return "diagnostic"
    if metric == "description_exact_match":
        return "auxiliary"
    return "primary"


def metric_aspect(metric: str) -> str:
    if metric in {"attribute_accuracy", "categorical_macro_f1"}:
        return "categorical_facts"
    if metric.startswith("numerical_"):
        return "numerical_facts"
    if metric.startswith("slot_"):
        return "information_completeness"
    if metric == "hallucination_rate":
        return "unsupported_facts"
    if metric in {
        "description_factual_accuracy",
        "entire_record_factual_accuracy",
    }:
        return "overall_factuality"
    if metric.startswith("physical_consistency"):
        return "physical_consistency_diagnostic"
    if metric == "description_exact_match":
        return "template_exact_match"
    return "other"


def finalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        row.setdefault("role", metric_role(row["metric"]))
        row.setdefault("aspect", metric_aspect(row["metric"]))
        row.setdefault("notes", "")
    return rows


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict payload in {path}.")
    required = {
        "predicted_signal_records",
        "target_signal_records",
        "predicted_signal_descriptions",
        "target_signal_descriptions",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Missing required signal-description fields: {', '.join(missing)}")
    return payload


def consistency_violations(
    record: dict[str, Any],
    *,
    los_delay_tolerance_ns: float,
    los_angle_tolerance_deg: float,
) -> list[str]:
    violations = []

    reflection_count = finite_float(record.get("reflection_count"))
    if reflection_count is not None and reflection_count < 0.0:
        violations.append("reflection_count_negative")
    reflection_path_count = finite_float(record.get("reflection_path_count"))
    if reflection_path_count is not None and reflection_path_count < 0.0:
        violations.append("reflection_path_count_negative")

    path_count = finite_float(record.get("path_count"))
    if path_count is not None and path_count < 1.0:
        violations.append("path_count_below_1")

    for field in ("first_path_delay_ns", "delay_spread_ns", "angle_spread_deg"):
        value = finite_float(record.get(field))
        if value is not None and value < 0.0:
            violations.append(f"{field}_negative")

    first_delay = finite_float(record.get("first_path_delay_ns"))

    los_status = str(record.get("los_status", "")).lower()
    if los_status == "los":
        los_delay = finite_float(record.get("los_delay_ns"))
        if los_delay is not None and los_delay <= 0.0:
            violations.append("los_delay_ns_nonpositive")

        if first_delay is None or los_delay is None:
            violations.append("los_missing_delay")
        else:
            delay_difference = abs(first_delay - los_delay)
            if delay_difference > RELATIONAL_DELAY_EPSILON_NS:
                violations.append("los_first_delay_unequal")
            if delay_difference > los_delay_tolerance_ns:
                violations.append("los_first_delay_outside_tolerance")

        first_angle = finite_float(record.get("first_path_angle_deg"))
        los_angle = finite_float(record.get("los_angle_deg"))
        if first_angle is None or los_angle is None:
            violations.append("los_missing_angle")
        elif numeric_error("first_path_angle_deg", first_angle, los_angle) > los_angle_tolerance_deg:
            violations.append("los_first_angle_inconsistent")

    return violations


def evaluate_payload(
    payload: dict,
    *,
    tolerances: dict[str, float],
    los_delay_tolerance_ns: float,
    los_angle_tolerance_deg: float,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]]]:
    predicted_records = payload["predicted_signal_records"]
    target_records = payload["target_signal_records"]
    predicted_texts = payload["predicted_signal_descriptions"]
    target_texts = payload["target_signal_descriptions"]

    if len(predicted_records) != len(target_records):
        raise ValueError("Predicted and target record counts differ.")

    rows: list[dict[str, str]] = []
    failures: list[dict[str, Any]] = []
    categorical_accuracies: list[float] = []
    categorical_macro_f1s: list[float] = []

    exact_matches = [
        str(predicted_texts[idx]) == str(target_texts[idx])
        for idx in range(len(predicted_records))
    ]
    rows.append(
        {
            "metric": "description_exact_match",
            "field": "text",
            "value": format_float(mean([1.0 if hit else 0.0 for hit in exact_matches])),
            "count": str(len(exact_matches)),
        }
    )

    for field in CATEGORICAL_FIELDS:
        predictions = [str(pred.get(field)) for pred in predicted_records]
        targets = [str(target.get(field)) for target in target_records]
        hits = [prediction == target for prediction, target in zip(predictions, targets)]
        accuracy = mean([1.0 if hit else 0.0 for hit in hits])
        macro_f1 = categorical_macro_f1(predictions, targets)
        categorical_accuracies.append(accuracy)
        categorical_macro_f1s.append(macro_f1)
        rows.append(
            {
                "metric": "attribute_accuracy",
                "field": field,
                "value": format_float(accuracy),
                "count": str(len(hits)),
            }
        )
        rows.append(
            {
                "metric": "categorical_macro_f1",
                "field": field,
                "value": format_float(macro_f1),
                "count": str(len(hits)),
            }
        )
    rows.append(
        {
            "metric": "categorical_macro_f1",
            "field": "all_categorical_fields",
            "value": format_float(mean_finite(categorical_macro_f1s)),
            "count": str(len(categorical_macro_f1s)),
        }
    )

    slot_tp = slot_fp = slot_fn = 0
    numeric_correct = 0
    numeric_total = 0
    field_errors: dict[str, list[float]] = {field: [] for field in NUMERIC_FIELDS}
    field_correct: dict[str, int] = {field: 0 for field in NUMERIC_FIELDS}
    field_total: dict[str, int] = {field: 0 for field in NUMERIC_FIELDS}
    delay_slot_tp = delay_slot_fp = delay_slot_fn = 0
    delay_numeric_correct = delay_numeric_total = 0
    entire_record_correct = 0

    consistency_ok = 0
    target_consistency_ok = 0
    violation_counts: dict[str, int] = {}
    target_violation_counts: dict[str, int] = {}
    prediction_only_violation_counts: dict[str, int] = {}
    target_only_violation_counts: dict[str, int] = {}
    shared_violation_counts: dict[str, int] = {}
    prediction_only_samples = 0
    target_only_samples = 0
    shared_violation_samples = 0

    for idx, (pred, target) in enumerate(zip(predicted_records, target_records)):
        sample_errors = []
        sample_numeric_correct = True
        for field in NUMERIC_FIELDS:
            pred_value = (
                finite_float(pred.get(field))
                if numeric_field_applicable(field, pred)
                else None
            )
            target_value = (
                finite_float(target.get(field))
                if numeric_field_applicable(field, target)
                else None
            )
            pred_present = pred_value is not None
            target_present = target_value is not None

            if pred_present and target_present:
                slot_tp += 1
                error = numeric_error(field, pred_value, target_value)
                field_errors[field].append(error)
                field_total[field] += 1
                numeric_total += 1
                if error <= tolerances[field]:
                    field_correct[field] += 1
                    numeric_correct += 1
                else:
                    sample_numeric_correct = False
                sample_errors.append((field, error))
            elif pred_present and not target_present:
                slot_fp += 1
                sample_numeric_correct = False
            elif target_present and not pred_present:
                slot_fn += 1
                sample_numeric_correct = False

            if field in DELAY_NUMERIC_FIELDS:
                if pred_present and target_present:
                    delay_slot_tp += 1
                    delay_numeric_total += 1
                    if error <= tolerances[field]:
                        delay_numeric_correct += 1
                elif pred_present and not target_present:
                    delay_slot_fp += 1
                elif target_present and not pred_present:
                    delay_slot_fn += 1

        sample_categorical_correct = all(
            str(pred.get(field)) == str(target.get(field))
            for field in CATEGORICAL_FIELDS
        )
        if sample_categorical_correct and sample_numeric_correct:
            entire_record_correct += 1

        violations = consistency_violations(
            pred,
            los_delay_tolerance_ns=los_delay_tolerance_ns,
            los_angle_tolerance_deg=los_angle_tolerance_deg,
        )
        target_violations = consistency_violations(
            target,
            los_delay_tolerance_ns=los_delay_tolerance_ns,
            los_angle_tolerance_deg=los_angle_tolerance_deg,
        )
        if not violations:
            consistency_ok += 1
        if not target_violations:
            target_consistency_ok += 1
        for violation in violations:
            violation_counts[violation] = violation_counts.get(violation, 0) + 1
        for violation in target_violations:
            target_violation_counts[violation] = target_violation_counts.get(violation, 0) + 1

        predicted_set = set(violations)
        target_set = set(target_violations)
        prediction_only_set = predicted_set - target_set
        target_only_set = target_set - predicted_set
        shared_set = predicted_set & target_set
        if prediction_only_set:
            prediction_only_samples += 1
        if target_only_set:
            target_only_samples += 1
        if shared_set:
            shared_violation_samples += 1
        for violation in prediction_only_set:
            prediction_only_violation_counts[violation] = (
                prediction_only_violation_counts.get(violation, 0) + 1
            )
        for violation in target_only_set:
            target_only_violation_counts[violation] = (
                target_only_violation_counts.get(violation, 0) + 1
            )
        for violation in shared_set:
            shared_violation_counts[violation] = shared_violation_counts.get(violation, 0) + 1

        categorical_errors = [
            field
            for field in CATEGORICAL_FIELDS
            if str(pred.get(field)) != str(target.get(field))
        ]
        if categorical_errors or violations or sample_errors:
            worst_numeric = max((error for _, error in sample_errors), default=0.0)
            failures.append(
                {
                    "index": idx,
                    "worst_numeric_error": worst_numeric,
                    "categorical_errors": categorical_errors,
                    "consistency_violations": violations,
                    "target_consistency_violations": target_violations,
                    "prediction_only_consistency_violations": sorted(prediction_only_set),
                    "target_only_consistency_violations": sorted(target_only_set),
                    "shared_consistency_violations": sorted(shared_set),
                    "predicted_text": predicted_texts[idx],
                    "target_text": target_texts[idx],
                    "predicted_record": pred,
                    "target_record": target,
                }
    )

    slot_precision = safe_ratio(slot_tp, slot_tp + slot_fp)
    slot_recall = safe_ratio(slot_tp, slot_tp + slot_fn)
    slot_f1_value = f1(slot_precision, slot_recall)
    hallucination_rate = safe_ratio(slot_fp, slot_tp + slot_fp)
    numerical_slot_accuracy = safe_ratio(numeric_correct, numeric_total)
    delay_slot_precision = safe_ratio(delay_slot_tp, delay_slot_tp + delay_slot_fp)
    delay_slot_recall = safe_ratio(delay_slot_tp, delay_slot_tp + delay_slot_fn)
    delay_slot_f1 = f1(delay_slot_precision, delay_slot_recall)
    delay_hallucination_rate = safe_ratio(delay_slot_fp, delay_slot_tp + delay_slot_fp)
    delay_numerical_slot_accuracy = safe_ratio(
        delay_numeric_correct, delay_numeric_total
    )
    delay_factual_accuracy = mean_finite(
        [
            delay_slot_f1,
            delay_numerical_slot_accuracy,
            (
                1.0 - delay_hallucination_rate
                if math.isfinite(delay_hallucination_rate)
                else math.nan
            ),
        ]
    )
    description_factual_accuracy = mean_finite(
        [
            *categorical_accuracies,
            slot_f1_value,
            numerical_slot_accuracy,
            1.0 - hallucination_rate if math.isfinite(hallucination_rate) else math.nan,
        ]
    )
    rows.extend(
        [
            {
                "metric": "description_factual_accuracy",
                "field": "primary_factual_metrics",
                "value": format_float(description_factual_accuracy),
                "count": str(len(predicted_records)),
                "notes": (
                    "Mean of categorical accuracies, slot F1, numerical slot accuracy, "
                    "and 1-hallucination rate."
                ),
            },
            {
                "metric": "entire_record_factual_accuracy",
                "field": "all_factual_fields",
                "value": format_float(
                    safe_ratio(entire_record_correct, len(predicted_records))
                ),
                "count": str(len(predicted_records)),
                "notes": (
                    "Fraction of samples with every categorical field correct, exact "
                    "numeric-slot presence, and every applicable numeric value within "
                    "its field tolerance."
                ),
            },
            {
                "metric": "slot_precision",
                "field": "all_numeric_slots",
                "value": format_float(slot_precision),
                "count": str(slot_tp + slot_fp),
            },
            {
                "metric": "slot_recall",
                "field": "all_numeric_slots",
                "value": format_float(slot_recall),
                "count": str(slot_tp + slot_fn),
            },
            {
                "metric": "slot_f1",
                "field": "all_numeric_slots",
                "value": format_float(slot_f1_value),
                "count": str(slot_tp + slot_fp + slot_fn),
            },
            {
                "metric": "hallucination_rate",
                "field": "all_numeric_slots",
                "value": format_float(hallucination_rate),
                "count": str(slot_tp + slot_fp),
            },
            {
                "metric": "numerical_slot_accuracy",
                "field": "all_numeric_slots",
                "value": format_float(numerical_slot_accuracy),
                "count": str(numeric_total),
            },
            {
                "metric": "description_factual_accuracy",
                "field": "delay_numeric_slots",
                "value": format_float(delay_factual_accuracy),
                "count": str(len(predicted_records)),
                "notes": (
                    "Delay-only factuality: mean of delay-slot F1, delay numerical "
                    "tolerance accuracy, and one minus delay hallucination rate."
                ),
            },
            {
                "metric": "slot_precision",
                "field": "delay_numeric_slots",
                "value": format_float(delay_slot_precision),
                "count": str(delay_slot_tp + delay_slot_fp),
            },
            {
                "metric": "slot_recall",
                "field": "delay_numeric_slots",
                "value": format_float(delay_slot_recall),
                "count": str(delay_slot_tp + delay_slot_fn),
            },
            {
                "metric": "slot_f1",
                "field": "delay_numeric_slots",
                "value": format_float(delay_slot_f1),
                "count": str(delay_slot_tp + delay_slot_fp + delay_slot_fn),
            },
            {
                "metric": "hallucination_rate",
                "field": "delay_numeric_slots",
                "value": format_float(delay_hallucination_rate),
                "count": str(delay_slot_tp + delay_slot_fp),
            },
            {
                "metric": "numerical_slot_accuracy",
                "field": "delay_numeric_slots",
                "value": format_float(delay_numerical_slot_accuracy),
                "count": str(delay_numeric_total),
            },
            {
                "metric": "physical_consistency_rate",
                "field": "predicted_description",
                "value": format_float(safe_ratio(consistency_ok, len(predicted_records))),
                "count": str(len(predicted_records)),
            },
            {
                "metric": "physical_consistency_violation_rate",
                "field": "predicted_description",
                "value": format_float(1.0 - safe_ratio(consistency_ok, len(predicted_records))),
                "count": str(len(predicted_records)),
            },
            {
                "metric": "physical_consistency_rate",
                "field": "target_description",
                "value": format_float(safe_ratio(target_consistency_ok, len(predicted_records))),
                "count": str(len(predicted_records)),
                "notes": "Consistency diagnostics applied to target records.",
            },
            {
                "metric": "physical_consistency_violation_rate",
                "field": "target_description",
                "value": format_float(1.0 - safe_ratio(target_consistency_ok, len(predicted_records))),
                "count": str(len(predicted_records)),
                "notes": "If high, the diagnostic rule is stricter than the target data semantics.",
            },
            {
                "metric": "physical_consistency_violation_sample_rate",
                "field": "prediction_only",
                "value": format_float(safe_ratio(prediction_only_samples, len(predicted_records))),
                "count": str(len(predicted_records)),
                "notes": "Samples with predicted-record violations not present in the target record.",
            },
            {
                "metric": "physical_consistency_violation_sample_rate",
                "field": "target_only",
                "value": format_float(safe_ratio(target_only_samples, len(predicted_records))),
                "count": str(len(predicted_records)),
                "notes": "Samples with target-record violations not present in the predicted record.",
            },
            {
                "metric": "physical_consistency_violation_sample_rate",
                "field": "shared_predicted_and_target",
                "value": format_float(safe_ratio(shared_violation_samples, len(predicted_records))),
                "count": str(len(predicted_records)),
                "notes": "Samples where predicted and target records trigger at least one same consistency rule.",
            },
        ]
    )

    for field in NUMERIC_FIELDS:
        errors = field_errors[field]
        rows.append(
            {
                "metric": "numerical_mae",
                "field": field,
                "value": format_float(mean(errors)),
                "count": str(len(errors)),
            }
        )
        rows.append(
            {
                "metric": "numerical_rmse",
                "field": field,
                "value": format_float(rmse(errors)),
                "count": str(len(errors)),
            }
        )
        rows.append(
            {
                "metric": f"numerical_accuracy@{tolerances[field]:g}",
                "field": field,
                "value": format_float(safe_ratio(field_correct[field], field_total[field])),
                "count": str(field_total[field]),
            }
        )

    for violation, count in sorted(violation_counts.items()):
        rows.append(
            {
                "metric": "physical_consistency_predicted_violation_count",
                "field": violation,
                "value": str(count),
                "count": str(len(predicted_records)),
            }
        )
    for violation, count in sorted(target_violation_counts.items()):
        rows.append(
            {
                "metric": "physical_consistency_target_violation_count",
                "field": violation,
                "value": str(count),
                "count": str(len(predicted_records)),
            }
        )
    for violation, count in sorted(prediction_only_violation_counts.items()):
        rows.append(
            {
                "metric": "physical_consistency_prediction_only_violation_count",
                "field": violation,
                "value": str(count),
                "count": str(len(predicted_records)),
                "notes": "Violation type appears in prediction but not in target for the same sample.",
            }
        )
    for violation, count in sorted(target_only_violation_counts.items()):
        rows.append(
            {
                "metric": "physical_consistency_target_only_violation_count",
                "field": violation,
                "value": str(count),
                "count": str(len(predicted_records)),
                "notes": "Violation type appears in target but not in prediction for the same sample.",
            }
        )
    for violation, count in sorted(shared_violation_counts.items()):
        rows.append(
            {
                "metric": "physical_consistency_shared_violation_count",
                "field": violation,
                "value": str(count),
                "count": str(len(predicted_records)),
                "notes": "Violation type appears in both prediction and target for the same sample.",
            }
        )

    failures.sort(
        key=lambda item: (
            len(item["categorical_errors"]) + len(item["consistency_violations"]),
            item["worst_numeric_error"],
        ),
        reverse=True,
    )

    spec = [
        {
            "type": "overall_factuality",
            "role": "primary",
            "metric": "description_factual_accuracy",
            "definition": (
                "Mean of categorical accuracies, numeric slot F1, numerical tolerance "
                "accuracy, and one minus hallucination rate."
            ),
        },
        {
            "type": "entire_record_factuality",
            "role": "primary",
            "metric": "entire_record_factual_accuracy",
            "definition": (
                "All categorical fields must match, numeric-slot presence must match, "
                "and all applicable numeric errors must be within field tolerances."
            ),
        },
        {
            "type": "categorical_attribute",
            "role": "primary",
            "fields": list(CATEGORICAL_FIELDS),
            "metric": "attribute accuracy and macro-F1 against target record",
        },
        {
            "type": "numeric_slot",
            "role": "primary",
            "fields": list(NUMERIC_FIELDS),
            "metric": "slot precision/recall/F1, MAE/RMSE, tolerance accuracy",
            "tolerances": tolerances,
            "angle_error": "circular absolute error in degrees",
        },
        {
            "type": "physical_consistency",
            "role": "diagnostic",
            "rules": [
                "delay/spread values must be non-negative when finite",
                "reflection count must be non-negative when finite",
                "reflected-path count must be non-negative when finite",
                "path count must be at least one when finite",
                "LoS delay must be positive when finite",
                "first-path delay must not be before LoS delay when both are finite",
                "LoS descriptions must include finite LoS delay/angle",
                "LoS delay must be close to first-path delay",
                "LoS angle must be close to first-path angle",
            ],
            "breakdown": [
                "predicted_description: rules applied to predicted records",
                "target_description: rules applied to target records",
                "prediction_only: predicted violation type not present in the matching target record",
                "target_only: target violation type not present in the matching predicted record",
                "shared_predicted_and_target: same violation type appears in both records",
            ],
        },
        {
            "type": "exact_match",
            "role": "auxiliary",
            "metric": "description_exact_match",
            "note": "Only meaningful for identical template strings; not used as the main text accuracy.",
        },
    ]
    return finalize_rows(rows), failures, spec


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ("role", "aspect", "metric", "field", "value", "count", "notes")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="Path produced by evaluate.py --save-signal-descriptions.")
    parser.add_argument("--output-dir", help="Defaults to the payload directory.")
    parser.add_argument("--failure-examples", type=int, default=20)
    parser.add_argument("--los-delay-tolerance-ns", type=float, default=50.0)
    parser.add_argument("--los-angle-tolerance-deg", type=float, default=30.0)
    args = parser.parse_args()

    payload_path = Path(args.payload)
    output_dir = Path(args.output_dir) if args.output_dir else payload_path.parent
    payload = load_payload(payload_path)
    rows, failures, spec = evaluate_payload(
        payload,
        tolerances=dict(DEFAULT_TOLERANCES),
        los_delay_tolerance_ns=args.los_delay_tolerance_ns,
        los_angle_tolerance_deg=args.los_angle_tolerance_deg,
    )

    metrics_csv = output_dir / "signal_description_text_metrics.csv"
    metrics_json = output_dir / "signal_description_text_metrics.json"
    failures_json = output_dir / "signal_description_failure_examples.json"
    spec_json = output_dir / "signal_description_eval_spec.json"

    write_csv(metrics_csv, rows)
    metrics_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures_json.write_text(
        json.dumps(failures[: args.failure_examples], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    spec_json.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"saved_text_metrics_csv={metrics_csv}")
    print(f"saved_text_metrics_json={metrics_json}")
    print(f"saved_failure_examples_json={failures_json}")
    print(f"saved_eval_spec_json={spec_json}")
    printed_metrics = {
        "description_factual_accuracy",
        "entire_record_factual_accuracy",
        "attribute_accuracy",
        "categorical_macro_f1",
        "slot_f1",
        "hallucination_rate",
        "numerical_slot_accuracy",
        "description_exact_match",
        "physical_consistency_rate",
        "physical_consistency_violation_rate",
    }
    for row in rows:
        if (
            row["metric"] not in printed_metrics
            and not row["metric"].startswith("physical_consistency_")
        ):
            continue
        print(f"{row['role']}_{row['metric']}_{row['field']}={row['value']}")


if __name__ == "__main__":
    main()
