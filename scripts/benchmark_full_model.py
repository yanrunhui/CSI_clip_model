from __future__ import annotations

import argparse
import json
import math
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    PHYSICS_TARGET_NAMES,
    PHYSICS_TARGET_OFFSETS,
    PHYSICS_TARGET_SCALES,
    PreprocessedCSIDataset,
    collate_fn,
)
from models.encoder import CSIEncoder  # noqa: E402
from models.model import CSIClip  # noqa: E402
from models.text_encoder import PhysicsTextEncoder  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    _apply_signal_description_correction,
    _infer_first_path_power_gate_mode,
    _infer_first_path_power_mode,
    _infer_first_path_power_use_internal_gate,
    _infer_shared_physics_token_residual_scale,
    _infer_token_norm_mode,
    _infer_use_array_invariant_delay_encoder,
    _infer_use_continuous_config_encoding,
    _infer_use_delay_specific_encoder,
    _infer_use_delay_spread_head,
    _infer_use_first_path_angle_context_encoder,
    _infer_use_los_angle_context_encoder,
    _infer_use_power_branch,
    _infer_use_shared_physics_token,
    _load_model_state_compatible,
    _physics_raw_predictions,
    _render_signal_description,
    _signal_description_record,
    build_tokenizer,
    move_batch,
)
from scripts.inference_benchmark_common import (  # noqa: E402
    parameter_counts,
    resolve_file_sha256,
    sha256_file,
    summarize_cost,
    timed_call,
    write_benchmark_outputs,
    write_environment,
)
from scripts.pretrain import deserialize_prototype_keys  # noqa: E402
from scripts.qwen_csi_text_common import target_response  # noqa: E402


def physics_index(name: str) -> int:
    return PHYSICS_TARGET_NAMES.index(name)


def checkpoint_attribute_shapes(checkpoint: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    pattern = re.compile(r"^attribute_classifiers\.([^.]+)\.3\.weight$")
    for name, value in checkpoint["model_state"].items():
        match = pattern.match(name)
        if match and isinstance(value, torch.Tensor):
            result[match.group(1)] = int(value.shape[0])
    return result


def build_model(checkpoint: dict[str, Any], device: torch.device) -> CSIClip:
    prototype_keys = deserialize_prototype_keys(checkpoint.get("prototype_keys"))
    if not prototype_keys:
        raise ValueError("Full-model checkpoint is missing prototype_keys.")
    token_norm_mode = _infer_token_norm_mode(checkpoint, None)
    tokenizer_word2id = checkpoint.get("tokenizer_word2id", {})
    # CaptionTokenizer.next_id is max(id) + 1; evaluate.py then reserves 8 ids.
    vocab_size = max(max(tokenizer_word2id.values(), default=291) + 9, 300)
    model = CSIClip(
        CSIEncoder(
            d_token=8,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
            use_continuous_config_encoding=_infer_use_continuous_config_encoding(
                checkpoint
            ),
        ),
        PhysicsTextEncoder(vocab_size=vocab_size),
        num_prototypes=len(prototype_keys),
        semantic_num_classes=len(prototype_keys),
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        use_power_branch=_infer_use_power_branch(checkpoint, None),
        first_path_power_mode=_infer_first_path_power_mode(checkpoint),
        first_path_power_use_internal_gate=_infer_first_path_power_use_internal_gate(
            checkpoint
        ),
        use_delay_spread_head=_infer_use_delay_spread_head(checkpoint),
        use_delay_specific_encoder=_infer_use_delay_specific_encoder(checkpoint),
        use_array_invariant_delay_encoder=_infer_use_array_invariant_delay_encoder(
            checkpoint
        ),
        use_los_angle_context_encoder=_infer_use_los_angle_context_encoder(checkpoint),
        use_first_path_angle_context_encoder=(
            _infer_use_first_path_angle_context_encoder(checkpoint)
        ),
        los_angle_context_token_norm_mode=token_norm_mode,
        use_shared_physics_token=_infer_use_shared_physics_token(checkpoint),
        shared_physics_token_residual_scale=(
            _infer_shared_physics_token_residual_scale(checkpoint)
        ),
        attribute_num_classes=checkpoint_attribute_shapes(checkpoint),
    ).to(device)
    _load_model_state_compatible(model, checkpoint["model_state"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the full multi-task model.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-sha256")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--warmup-samples", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--signal-description-correction",
        choices=("none", "bounds", "relational"),
        default="relational",
    )
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.limit <= 0 or args.warmup_samples < 0 or args.repeats <= 0:
        raise ValueError("limit/repeats must be positive and warmup-samples nonnegative.")
    if not torch.cuda.is_available():
        raise RuntimeError("The cost benchmark requires CUDA.")

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    samples = PreprocessedCSIDataset.from_pt(args.data_path).samples[: args.limit]
    if len(samples) != args.limit:
        raise ValueError(f"Requested {args.limit} samples, found {len(samples)}.")
    prototype_keys = deserialize_prototype_keys(checkpoint.get("prototype_keys"))
    if not prototype_keys:
        raise ValueError("Full-model checkpoint is missing prototype_keys.")
    tokenizer = build_tokenizer(samples, checkpoint)
    model = build_model(checkpoint, device)
    use_power_branch = _infer_use_power_branch(checkpoint, None)
    semantic_classifier_enabled = (
        float(checkpoint.get("args", {}).get("semantic_classifier_weight", 0.0)) > 0
    )
    prototype_features = model.encode_prototypes(normalize=True)
    power_gate_mode = _infer_first_path_power_gate_mode(checkpoint, None)
    if power_gate_mode == "predicted_los":
        power_gate_mode = "base"

    @torch.inference_mode()
    def predict(sample) -> dict[str, Any]:
        batch = move_batch(
            collate_fn([sample], tokenizer=tokenizer, max_caption_len=48), device
        )
        features = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
            config_features=batch.get("config_features"),
            antenna_coordinates=batch.get("antenna_coordinates"),
            antenna_mask=batch.get("antenna_mask"),
        )
        delay_context = model.encode_csi_delay_context(
            batch["tokens"],
            batch["token_mask"],
            subcarrier_spacing=batch.get("subcarrier_spacing"),
            config_features=batch.get("config_features"),
        )
        first_delay_context = model.encode_first_path_delay_context(
            batch["tokens"],
            batch["token_mask"],
            beam_positions=batch.get("beam_positions"),
            freq_bin=batch.get("freq_bin"),
            bw_bin=batch.get("bw_bin"),
            subcarrier_spacing=batch.get("subcarrier_spacing"),
            config_features=batch.get("config_features"),
        )
        los_angle_context = model.encode_los_angle_context(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
        )
        first_angle_context = model.encode_first_path_angle_context(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            subcarrier_spacing=batch.get("subcarrier_spacing"),
        )
        power_context = None
        if use_power_branch:
            power_context = model.encode_power_context(
                batch["tokens"],
                batch["token_mask"],
                delay_power_map=batch.get("delay_power_map"),
                delay_power_profile=batch.get("delay_power_profile"),
            )
        outputs = model.predict_physics_components(
            features,
            power_context=power_context,
            delay_context=delay_context,
            first_path_delay_context=first_delay_context,
            los_angle_context=los_angle_context,
            first_path_angle_context=first_angle_context,
        )
        normalized = outputs["final"].clone()
        if power_gate_mode == "base":
            idx = physics_index("first_path_power_dbw")
            normalized[:, idx] = outputs["base"][:, idx]
        raw = _physics_raw_predictions(normalized)[0]
        if semantic_classifier_enabled:
            semantic_label = int(model.predict_semantic(features).argmax(dim=1)[0])
        else:
            normalized_features = F.normalize(features, dim=-1)
            semantic_label = int((normalized_features @ prototype_features.T).argmax(dim=1)[0])
        reflection_idx = physics_index("reflection_count")
        reflection_count = (
            float(outputs["reflection_count_prediction"][0])
            * float(PHYSICS_TARGET_SCALES[reflection_idx])
            + float(PHYSICS_TARGET_OFFSETS[reflection_idx])
        )
        record = _signal_description_record(
            prototype_keys[semantic_label],
            raw,
            los_delay_ns=float(outputs["los_delay_context"][0]) * 3000.0,
            los_angle_sincos=outputs["los_angle_sincos"][0],
            reflection_count=reflection_count,
            reflection_path_count=(
                float(outputs["reflection_path_count_prediction"][0]) * 10.0
            ),
            signal_description_correction=args.signal_description_correction,
        )
        record = _apply_signal_description_correction(
            record, args.signal_description_correction
        )
        return {**record, "description": _render_signal_description(record)}

    with torch.inference_mode():
        for sample in samples[: args.warmup_samples]:
            predict(sample)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        rows: list[dict[str, Any]] = []
        predicted_records: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            for index, sample in enumerate(samples):
                result = timed_call(lambda sample=sample: predict(sample), device)
                rows.append(
                    {
                        "model": "full_multitask",
                        "seed": args.seed,
                        "repeat": repeat,
                        "sample_index": index,
                        "group_id": str(getattr(sample, "group_id", "")),
                        "config_key": str(getattr(sample, "config_key", "")),
                        "wall_ms": result.wall_ms,
                        "cuda_ms": result.cuda_ms,
                        "generated_tokens": 0,
                    }
                )
                if repeat == 0:
                    predicted_records.append(result.value)
                if (index + 1) % args.log_every == 0:
                    print(
                        f"benchmark_repeat={repeat + 1}/{args.repeats} "
                        f"completed={index + 1}/{args.limit}",
                        flush=True,
                    )

    target_records = [target_response(sample) for sample in samples]
    payload = {
        "predicted_signal_records": predicted_records,
        "target_signal_records": target_records,
        "predicted_signal_descriptions": [row["description"] for row in predicted_records],
        "target_signal_descriptions": [row["description"] for row in target_records],
        "comparisons": [
            {
                "index": index,
                "group_id": str(getattr(sample, "group_id", "")),
                "predicted_record": predicted_records[index],
                "target_record": target_records[index],
            }
            for index, sample in enumerate(samples)
        ],
    }
    total_parameters, trainable_parameters = parameter_counts(model)
    summary = summarize_cost(
        model="full_multitask",
        seed=args.seed,
        rows=rows,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        device=device,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        metadata={
            "repeats": args.repeats,
            "verbalizer": "deterministic_template",
            "inference_precision": "float32",
            "cpu_offload": False,
            "data_sha256": resolve_file_sha256(args.data_path, args.data_sha256),
            "checkpoint_sha256": sha256_file(args.checkpoint),
        },
    )
    output = Path(args.output_dir)
    write_benchmark_outputs(output, rows=rows, summary=summary, config=vars(args))
    write_environment(output, device)
    torch.save(payload, output / "signal_descriptions.pt")
    (output / "predictions.json").write_text(
        json.dumps(predicted_records, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print("benchmark_cost_summary=" + json.dumps(summary, sort_keys=True))
    print(f"saved_signal_descriptions={output / 'signal_descriptions.pt'}")


if __name__ == "__main__":
    main()
