from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import (  # noqa: E402
    _apply_signal_description_correction,
    _render_signal_description,
    _signal_description_record,
)
from data.dataset import physics_raw_values  # noqa: E402
from scripts.train_multinumerology_delay_fusion import move_nested  # noqa: E402
from scripts.train_multinumerology_generalization import (  # noqa: E402
    FINAL_OUTPUT_KEY,
    FINAL_OUTPUT_METHOD,
    PeriodConditionedDelayFusion,
    make_loaders,
    parse_pair,
)


def load_base_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict payload in {path}.")
    required = {
        "predicted_signal_records",
        "target_signal_records",
        "comparisons",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "Base payload is missing required fields: " + ", ".join(missing)
        )
    record_count = len(payload["predicted_signal_records"])
    if len(payload["target_signal_records"]) != record_count:
        raise ValueError("Base payload predicted/target record counts differ.")
    if len(payload["comparisons"]) != record_count:
        raise ValueError("Base payload comparisons are not aligned with its records.")
    return payload


def payload_group_index(payload: dict) -> dict[str, int]:
    index: dict[str, int] = {}
    for row, comparison in enumerate(payload["comparisons"]):
        group_id = str(comparison.get("group_id", "")).strip()
        if not group_id:
            raise ValueError(f"Base payload comparison {row} has no group_id.")
        if group_id in index:
            raise ValueError(f"Duplicate group_id={group_id!r} in base payload.")
        index[group_id] = row
    return index


def target_record_from_sample(sample) -> dict[str, float | str]:
    los_angle_deg = float(getattr(sample, "los_aoa_az_deg", math.nan))
    los_angle_sincos = None
    if math.isfinite(los_angle_deg):
        los_angle_rad = math.radians(los_angle_deg)
        los_angle_sincos = torch.tensor(
            [math.sin(los_angle_rad), math.cos(los_angle_rad)],
            dtype=torch.float32,
        )
    los_delay_s = float(getattr(sample, "los_delay_s", math.nan))
    los_delay_ns = los_delay_s * 1e9 if math.isfinite(los_delay_s) else math.nan
    return _signal_description_record(
        sample.semantic_key,
        physics_raw_values(sample),
        los_delay_ns=los_delay_ns,
        los_angle_sincos=los_angle_sincos,
        reflection_count=float(getattr(sample, "reflection_count", math.nan)),
        reflection_path_count=float(getattr(sample, "reflection_path_count", math.nan)),
    )


def build_oracle_base_payload(samples: list[object], fused_group_ids: set[str]) -> dict:
    records = []
    comparisons = []
    for sample in samples:
        group_id = str(getattr(sample, "group_id", "")).strip()
        if group_id not in fused_group_ids:
            continue
        record = target_record_from_sample(sample)
        records.append(record)
        comparisons.append(
            {
                "group_id": group_id,
                "config_key": str(getattr(sample, "config_key", "")),
            }
        )
    return {
        "predicted_signal_records": copy.deepcopy(records),
        "target_signal_records": records,
        "comparisons": comparisons,
    }


@torch.no_grad()
def collect_fused_predictions(
    model, loader, device: torch.device
) -> dict[str, dict[str, float]]:
    model.eval()
    predictions: dict[str, dict[str, float]] = {}
    for batch in loader:
        moved = move_nested(batch, device)
        outputs = model(
            moved["view_a"],
            moved["view_b"],
            moved["period_a_ns"],
            moved["period_b_ns"],
            moved["availability_a"],
            moved["availability_b"],
        )
        first_path = outputs["first_path_delay_ns"][FINAL_OUTPUT_KEY].detach().cpu()
        los_delay = outputs["los_delay_ns"][FINAL_OUTPUT_KEY].detach().cpu()
        for row, group_id in enumerate(batch["group_ids"]):
            group_id = str(group_id)
            if group_id in predictions:
                raise ValueError(
                    f"Duplicate fused prediction for group_id={group_id!r}."
                )
            predictions[group_id] = {
                "first_path_delay_ns": float(first_path[row]),
                "los_delay_ns": float(los_delay[row]),
            }
    return predictions


def build_payload(
    *,
    base_payload: dict,
    fused_predictions: dict[str, dict[str, float]],
    context_mode: str,
    correction: str,
    checkpoint_path: Path,
    base_payload_path: Path,
    pair_name: str,
) -> dict:
    group_index = payload_group_index(base_payload)
    missing_from_base = sorted(set(fused_predictions) - set(group_index))
    if missing_from_base:
        preview = ", ".join(missing_from_base[:5])
        raise ValueError(
            f"Base payload lacks {len(missing_from_base)} paired group IDs; first: {preview}"
        )

    predicted_records = []
    target_records = []
    predicted_texts = []
    target_texts = []
    comparisons = []
    for base_row, comparison in enumerate(base_payload["comparisons"]):
        group_id = str(comparison.get("group_id", "")).strip()
        fused = fused_predictions.get(group_id)
        if fused is None:
            continue
        target_record = copy.deepcopy(base_payload["target_signal_records"][base_row])
        if context_mode == "base_prediction":
            predicted_record = copy.deepcopy(
                base_payload["predicted_signal_records"][base_row]
            )
        elif context_mode == "oracle_context":
            predicted_record = copy.deepcopy(target_record)
        else:
            raise ValueError(f"Unsupported context_mode={context_mode!r}.")

        predicted_record["first_path_delay_ns"] = fused["first_path_delay_ns"]
        predicted_record["los_delay_ns"] = fused["los_delay_ns"]
        predicted_record = _apply_signal_description_correction(
            predicted_record, correction
        )
        predicted_text = _render_signal_description(predicted_record)
        target_text = _render_signal_description(target_record)
        predicted_records.append(predicted_record)
        target_records.append(target_record)
        predicted_texts.append(predicted_text)
        target_texts.append(target_text)
        comparisons.append(
            {
                "index": len(comparisons),
                "group_id": group_id,
                "config_key": comparison.get("config_key", ""),
                "predicted_signal_description": predicted_text,
                "target_signal_description": target_text,
                "predicted_record": predicted_record,
                "target_record": target_record,
            }
        )

    if not predicted_records:
        raise ValueError("No overlapping records were available for text evaluation.")
    return {
        "predicted_signal_descriptions": predicted_texts,
        "target_signal_descriptions": target_texts,
        "predicted_signal_records": predicted_records,
        "target_signal_records": target_records,
        "comparisons": comparisons,
        "metadata": {
            "task": "delay_specific_cross_numerology_text_extension",
            "context_mode": context_mode,
            "base_payload": str(base_payload_path),
            "fusion_checkpoint": str(checkpoint_path),
            "fusion_pair": pair_name,
            "fusion_output_method": FINAL_OUTPUT_METHOD,
            "signal_description_correction": correction,
            "record_count": len(predicted_records),
            "interpretation": (
                "End-to-end text evaluation with fused delay slots replacing the base "
                "system predictions."
                if context_mode == "base_prediction"
                else "Delay-slot isolation with oracle non-delay context; not a complete "
                "end-to-end CSI-to-language result."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-pair",
        nargs=6,
        required=True,
        metavar=(
            "NAME_A",
            "PERIOD_A_NS",
            "PATH_A",
            "NAME_B",
            "PERIOD_B_NS",
            "PATH_B",
        ),
    )
    parser.add_argument(
        "--base-payload",
        type=Path,
        help=(
            "Existing full-system signal-description payload. Required for "
            "base_prediction; optional for oracle_context."
        ),
    )
    parser.add_argument(
        "--context-mode",
        choices=("base_prediction", "oracle_context"),
        default="base_prediction",
    )
    parser.add_argument(
        "--signal-description-correction",
        choices=("none", "bounds", "relational"),
        default="relational",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.context_mode == "base_prediction" and args.base_payload is None:
        raise ValueError(
            "--base-payload is required for --context-mode base_prediction."
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model = PeriodConditionedDelayFusion(
        max_delay_ns=float(checkpoint_args.get("max_delay_ns", 1920.0)),
        hidden_dim=int(checkpoint_args.get("hidden_dim", 256)),
        residual_scale_ns=float(checkpoint_args.get("residual_scale_ns", 50.0)),
        period_prior_strength=float(checkpoint_args.get("period_prior_strength", 0.0)),
        fallback_confidence_threshold=float(
            checkpoint_args.get("fallback_confidence_threshold", 0.0)
        ),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    spec = parse_pair(list(args.test_pair))
    sample_cache: dict[str, tuple[list[object], dict[str, object]]] = {}
    loaders, pair_info = make_loaders(
        [spec],
        max_delay_ns=float(checkpoint_args.get("max_delay_ns", 1920.0)),
        max_delay_spread_ns=(
            None
            if checkpoint_args.get("max_delay_spread_ns", 400.0) is None
            else float(checkpoint_args.get("max_delay_spread_ns", 400.0))
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=int(checkpoint_args.get("seed", 0)),
        shuffle=False,
        sample_cache=sample_cache,
    )
    _, loader = loaders[0]
    fused_predictions = collect_fused_predictions(model, loader, device)
    if args.base_payload is not None:
        base_payload = load_base_payload(args.base_payload)
        base_payload_path = args.base_payload
    else:
        source_samples = sample_cache[spec.path_a][0]
        base_payload = build_oracle_base_payload(source_samples, set(fused_predictions))
        base_payload_path = Path("<constructed-from-target-samples>")
    output_payload = build_payload(
        base_payload=base_payload,
        fused_predictions=fused_predictions,
        context_mode=args.context_mode,
        correction=args.signal_description_correction,
        checkpoint_path=args.checkpoint,
        base_payload_path=base_payload_path,
        pair_name=spec.name,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)
    print(f"device={device}")
    print(f"fusion_pair={spec.name}")
    print(f"fusion_output_method={FINAL_OUTPUT_METHOD}")
    print(f"context_mode={args.context_mode}")
    print(f"paired_sample_count={pair_info[spec.name]['paired_count']}")
    print(f"text_record_count={len(output_payload['predicted_signal_records'])}")
    print("metadata=" + json.dumps(output_payload["metadata"], sort_keys=True))
    print(f"saved_signal_descriptions={args.output}")
    for index in range(min(3, len(output_payload["predicted_signal_descriptions"]))):
        print(
            f"signal_description_example_{index + 1}_pred_text="
            f"{output_payload['predicted_signal_descriptions'][index]}"
        )
        print(
            f"signal_description_example_{index + 1}_true_text="
            f"{output_payload['target_signal_descriptions'][index]}"
        )


if __name__ == "__main__":
    main()
