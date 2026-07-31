from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset  # noqa: E402


def _finite_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _format_counter(values, limit: int = 20) -> str:
    counts = Counter(values)
    items = sorted(counts.items(), key=lambda item: str(item[0]))
    text = ",".join(f"{key}:{count}" for key, count in items[:limit])
    if len(items) > limit:
        text += f",...({len(items) - limit}_more)"
    return text or "none"


def _format_float_counter(values, decimals: int = 6) -> str:
    rounded = [
        round(float(value), decimals)
        for value in values
        if _finite_float(value) is not None
    ]
    return _format_counter(rounded)


def _quantile_text(values: torch.Tensor) -> str:
    if values.numel() == 0:
        return "nan"
    values = values.float()
    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    result = torch.quantile(values, quantiles)
    return ",".join(
        f"{name}:{float(value):.6g}"
        for name, value in zip(("min", "p25", "p50", "p75", "max"), result)
    )


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return math.nan
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 0.0:
        return math.nan
    return float((x * y).sum() / denominator)


def _los_status(sample) -> str:
    return str(getattr(sample.semantic_key, "los_status", "unknown"))


def summarize_dataset(
    name: str,
    path: Path,
    expected_bandwidth_hz: float | None,
) -> tuple[list, dict[str, object]]:
    samples = PreprocessedCSIDataset.from_pt(path).samples
    if not samples:
        raise ValueError(f"No samples loaded from {path}.")

    token_n_freq = [int(sample.tokens.shape[-1]) for sample in samples]
    d_token = [int(sample.tokens.shape[-2]) for sample in samples]
    spacings = [float(sample.subcarrier_spacing_hz) for sample in samples]
    los_delays_ns = []
    first_delays_ns = []
    los_first_diffs_ns = []
    los_group_ids = []
    for sample in samples:
        if _los_status(sample) != "los":
            continue
        los_group_ids.append(str(getattr(sample, "group_id", "")))
        los_delay = _finite_float(getattr(sample, "los_delay_s", math.nan))
        first_delay = _finite_float(getattr(sample, "first_path_delay_s", math.nan))
        if los_delay is not None:
            los_delays_ns.append(los_delay * 1e9)
        if first_delay is not None:
            first_delays_ns.append(first_delay * 1e9)
        if los_delay is not None and first_delay is not None:
            los_first_diffs_ns.append((first_delay - los_delay) * 1e9)

    group_ids = [str(getattr(sample, "group_id", "")) for sample in samples]
    config_keys = [str(getattr(sample, "config_key", "")) for sample in samples]
    implied_spans_hz = [
        spacing * n_freq
        for spacing, n_freq in zip(spacings, token_n_freq)
    ]
    expected_spacing_ratios = []
    if expected_bandwidth_hz is not None:
        expected_spacing_ratios = [
            spacing / (expected_bandwidth_hz / max(n_freq, 1))
            for spacing, n_freq in zip(spacings, token_n_freq)
        ]

    print(f"[{name}]")
    print(f"path={path}")
    print(f"sample_count={len(samples)}")
    print(f"los_status_histogram={_format_counter(_los_status(sample) for sample in samples)}")
    print(f"config_key_histogram={_format_counter(config_keys)}")
    print(f"token_shape_histogram={_format_counter(tuple(sample.tokens.shape) for sample in samples)}")
    print(f"n_tokens_histogram={_format_counter(int(sample.n_tokens) for sample in samples)}")
    print(f"d_token_histogram={_format_counter(d_token)}")
    print(f"token_n_freq_histogram={_format_counter(token_n_freq)}")
    print(f"freq_bin_histogram={_format_counter(int(sample.freq_bin) for sample in samples)}")
    print(f"bw_bin_histogram={_format_counter(int(sample.bw_bin) for sample in samples)}")
    print(f"subcarrier_spacing_hz_histogram={_format_float_counter(spacings, decimals=3)}")
    print(f"spacing_times_token_nf_hz_histogram={_format_float_counter(implied_spans_hz, decimals=3)}")
    if expected_spacing_ratios:
        print(
            "stored_to_expected_resampled_spacing_ratio_histogram="
            f"{_format_float_counter(expected_spacing_ratios, decimals=6)}"
        )
    print(f"group_id_unique_count={len(set(group_ids))}")
    print(f"group_id_duplicate_count={len(group_ids) - len(set(group_ids))}")
    print(f"los_delay_ns_quantiles={_quantile_text(torch.tensor(los_delays_ns))}")
    print(f"first_path_delay_los_ns_quantiles={_quantile_text(torch.tensor(first_delays_ns))}")
    print(
        "first_minus_los_delay_ns_quantiles="
        f"{_quantile_text(torch.tensor(los_first_diffs_ns))}"
    )
    if los_first_diffs_ns:
        diffs = torch.tensor(los_first_diffs_ns).abs()
        print(
            "first_los_delay_equal_within_1e-3ns_rate="
            f"{float((diffs <= 1e-3).float().mean()):.6g}"
        )
    print()

    return samples, {
        "group_ids": set(group_ids),
        "los_group_ids": set(los_group_ids),
    }


def summarize_checkpoint(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    state = checkpoint.get("model_state", {})
    critical_prefixes = (
        "csi_delay_context_encoder.",
        "first_path_delay_context_encoder.",
        "los_delay_context_head.",
        "first_path_delay_context_head.",
    )
    print("[checkpoint]")
    print(f"path={path}")
    print(f"checkpoint_data_path={args.get('data_path', 'missing')}")
    print(f"checkpoint_seed={args.get('seed', 'missing')}")
    print(f"checkpoint_token_norm_mode={args.get('token_norm_mode', 'missing')}")
    print(
        "checkpoint_use_delay_specific_encoder="
        f"{args.get('use_delay_specific_encoder', 'missing')}"
    )
    for prefix in critical_prefixes:
        keys = sorted(key for key in state if key.startswith(prefix))
        print(f"checkpoint_{prefix.rstrip('.').replace('.', '_')}_key_count={len(keys)}")
        if keys:
            print(
                f"checkpoint_{prefix.rstrip('.').replace('.', '_')}_key_examples="
                + ",".join(keys[:3])
            )
    print()


def _record_value(record: dict, field: str) -> float | None:
    return _finite_float(record.get(field))


def summarize_payload(path: Path, test_samples: list) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    comparisons = payload.get("comparisons", [])
    sample_by_group = {
        str(getattr(sample, "group_id", "")): sample
        for sample in test_samples
    }
    matched = 0
    missing_group = 0
    target_los_status_mismatch = 0
    target_los_delay_mismatch = 0
    predictions = []
    targets = []

    for comparison in comparisons:
        group_id = str(comparison.get("group_id", ""))
        sample = sample_by_group.get(group_id)
        if sample is None:
            missing_group += 1
            continue
        matched += 1
        target_record = comparison.get("target_record", {})
        predicted_record = comparison.get("predicted_record", {})
        if str(target_record.get("los_status")) != _los_status(sample):
            target_los_status_mismatch += 1

        sample_los_delay_s = _finite_float(getattr(sample, "los_delay_s", math.nan))
        target_los_delay_ns = _record_value(target_record, "los_delay_ns")
        if sample_los_delay_s is not None and target_los_delay_ns is not None:
            if abs(sample_los_delay_s * 1e9 - target_los_delay_ns) > 1e-3:
                target_los_delay_mismatch += 1

        if str(target_record.get("los_status")) != "los":
            continue
        prediction = _record_value(predicted_record, "los_delay_ns")
        target = _record_value(target_record, "los_delay_ns")
        if prediction is not None and target is not None:
            predictions.append(prediction)
            targets.append(target)

    print("[payload]")
    print(f"path={path}")
    print(f"comparison_count={len(comparisons)}")
    print(f"payload_dataset_group_id_match_count={matched}")
    print(f"payload_dataset_missing_group_id_count={missing_group}")
    print(f"payload_target_los_status_mismatch_count={target_los_status_mismatch}")
    print(f"payload_target_los_delay_mismatch_count={target_los_delay_mismatch}")
    if not predictions:
        print("payload_los_delay_valid_count=0")
        return

    prediction = torch.tensor(predictions)
    target = torch.tensor(targets)
    error = prediction - target
    print(f"payload_los_delay_valid_count={target.numel()}")
    print(f"payload_los_delay_MAE_ns={float(error.abs().mean()):.6g}")
    print(f"payload_los_delay_RMSE_ns={float(error.square().mean().sqrt()):.6g}")
    print(f"payload_los_delay_signed_mean_ns={float(error.mean()):.6g}")
    print(f"payload_los_delay_pearson={_safe_pearson(prediction, target):.6g}")
    print(f"payload_los_delay_target_quantiles_ns={_quantile_text(target)}")
    print(f"payload_los_delay_prediction_quantiles_ns={_quantile_text(prediction)}")
    print(f"payload_los_delay_error_quantiles_ns={_quantile_text(error)}")

    edges = (0.0, 100.0, 300.0, 600.0, 1000.0, float("inf"))
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (target >= lower) & (target < upper)
        label = f"{int(lower)}_{'inf' if math.isinf(upper) else int(upper)}"
        if not bool(mask.any()):
            print(f"payload_los_delay_target_bin_{label}_count=0")
            continue
        bin_error = error[mask]
        print(f"payload_los_delay_target_bin_{label}_count={int(mask.sum())}")
        print(
            f"payload_los_delay_target_bin_{label}_MAE_ns="
            f"{float(bin_error.abs().mean()):.6g}"
        )
        print(
            f"payload_los_delay_target_bin_{label}_signed_mean_ns="
            f"{float(bin_error.mean()):.6g}"
        )


def _payload_los_delay_by_group(path: Path) -> dict[str, tuple[float, float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = {}
    for comparison in payload.get("comparisons", []):
        target_record = comparison.get("target_record", {})
        predicted_record = comparison.get("predicted_record", {})
        if str(target_record.get("los_status")) != "los":
            continue
        prediction = _record_value(predicted_record, "los_delay_ns")
        target = _record_value(target_record, "los_delay_ns")
        group_id = str(comparison.get("group_id", ""))
        if group_id and prediction is not None and target is not None:
            result[group_id] = (prediction, target)
    return result


def summarize_paired_payloads(reference_path: Path, test_path: Path) -> None:
    reference = _payload_los_delay_by_group(reference_path)
    test = _payload_los_delay_by_group(test_path)
    shared_ids = sorted(set(reference) & set(test))

    print("[paired_payloads]")
    print(f"reference_payload_path={reference_path}")
    print(f"test_payload_path={test_path}")
    print(f"paired_los_group_count={len(shared_ids)}")
    if not shared_ids:
        return

    reference_prediction = torch.tensor([reference[group_id][0] for group_id in shared_ids])
    reference_target = torch.tensor([reference[group_id][1] for group_id in shared_ids])
    test_prediction = torch.tensor([test[group_id][0] for group_id in shared_ids])
    test_target = torch.tensor([test[group_id][1] for group_id in shared_ids])
    target_delta = test_target - reference_target
    prediction_delta = test_prediction - reference_prediction
    reference_error = reference_prediction - reference_target
    test_error = test_prediction - test_target

    print(
        "paired_target_delay_mismatch_rate_gt_1e-3ns="
        f"{float((target_delta.abs() > 1e-3).float().mean()):.6g}"
    )
    print(f"paired_target_delay_delta_MAE_ns={float(target_delta.abs().mean()):.6g}")
    print(f"paired_target_delay_delta_max_abs_ns={float(target_delta.abs().max()):.6g}")
    print(
        "paired_reference_los_delay_MAE_ns="
        f"{float(reference_error.abs().mean()):.6g}"
    )
    print(f"paired_test_los_delay_MAE_ns={float(test_error.abs().mean()):.6g}")
    print(
        "paired_test_minus_reference_prediction_MAE_ns="
        f"{float(prediction_delta.abs().mean()):.6g}"
    )
    print(
        "paired_test_minus_reference_prediction_signed_mean_ns="
        f"{float(prediction_delta.mean()):.6g}"
    )
    print(
        "paired_test_reference_prediction_pearson="
        f"{_safe_pearson(test_prediction, reference_prediction):.6g}"
    )
    print(
        "paired_reference_error_quantiles_ns="
        f"{_quantile_text(reference_error)}"
    )
    print(f"paired_test_error_quantiles_ns={_quantile_text(test_error)}")
    print(
        "paired_prediction_delta_quantiles_ns="
        f"{_quantile_text(prediction_delta)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data", type=Path)
    parser.add_argument("--reference-payload", type=Path)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--payload", type=Path)
    parser.add_argument(
        "--expected-bandwidth-hz",
        type=float,
        help=(
            "Expected occupied bandwidth after preprocessing. When provided, "
            "the script checks stored SCS against bandwidth/token_n_freq."
        ),
    )
    args = parser.parse_args()

    reference_info = None
    if args.reference_data is not None:
        _, reference_info = summarize_dataset(
            "reference_dataset",
            args.reference_data,
            args.expected_bandwidth_hz,
        )
    test_samples, test_info = summarize_dataset(
        "test_dataset",
        args.test_data,
        args.expected_bandwidth_hz,
    )
    if reference_info is not None:
        print("[reference_test_overlap]")
        print(
            "group_id_overlap_count="
            f"{len(reference_info['group_ids'] & test_info['group_ids'])}"
        )
        print(
            "los_group_id_overlap_count="
            f"{len(reference_info['los_group_ids'] & test_info['los_group_ids'])}"
        )
        print()

    summarize_checkpoint(args.checkpoint)
    if args.payload is not None:
        summarize_payload(args.payload, test_samples)
    if args.reference_payload is not None:
        if args.payload is None:
            raise ValueError("--reference-payload requires --payload.")
        summarize_paired_payloads(args.reference_payload, args.payload)


if __name__ == "__main__":
    main()
