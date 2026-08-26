from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset  # noqa: E402
from scripts.csi_prefix_language_model import (  # noqa: E402
    is_qwen35_model,
    language_model_hidden_size,
)
from scripts.evaluate_qwen_csi_prefix import (  # noqa: E402
    generation_token_suffix,
    load_qwen,
    resolve_compute_dtype,
)
from scripts.inference_benchmark_common import (  # noqa: E402
    parameter_counts,
    resolve_file_sha256,
    sha256_file,
    sha256_file_map,
    summarize_cost,
    timed_call,
    write_benchmark_outputs,
    write_environment,
)
from scripts.qwen_csi_prefix_common import (  # noqa: E402
    CSIPrefixMapper,
    collate_csi_samples,
    mapped_chat_text,
    move_csi_batch,
)
from scripts.qwen_csi_text_common import parse_generated_response, target_response  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-derived CSI soft-prefix language models."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mapping-checkpoint", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-sha256")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--warmup-samples", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--assistant-prefill")
    parser.add_argument("--response-prefix")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--warmup-log-every", type=int, default=10)
    args = parser.parse_args()
    if (
        args.limit <= 0
        or args.warmup_samples < 0
        or args.repeats <= 0
        or args.log_every <= 0
        or args.warmup_log_every <= 0
    ):
        raise ValueError(
            "limit, repeats, and log intervals must be positive; "
            "warmup-samples must be nonnegative."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The cost benchmark requires CUDA.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = resolve_compute_dtype(args.compute_dtype)
    qwen = load_qwen(args, dtype)
    device_map = getattr(qwen, "hf_device_map", {}) or getattr(
        getattr(qwen, "base_model", None), "hf_device_map", {}
    ) or {}
    offloaded_devices = sorted(
        {
            str(value)
            for value in device_map.values()
            if str(value).lower() in {"cpu", "disk"}
        }
    )
    if offloaded_devices:
        raise RuntimeError(
            "CPU/disk offload is not allowed in the comparable benchmark: "
            f"{offloaded_devices}"
        )
    device = next(qwen.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(
            "The first language-model device is not CUDA. Run on a GPU with enough memory "
            "and avoid CPU offload for comparable latency measurements."
        )
    checkpoint = torch.load(args.mapping_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    assistant_prefill = (
        args.assistant_prefill
        if args.assistant_prefill is not None
        else str(checkpoint_args.get("assistant_prefill", ""))
    )
    response_prefix = (
        args.response_prefix
        if args.response_prefix is not None
        else str(checkpoint_args.get("response_prefix", ""))
    )
    if response_prefix and not assistant_prefill.endswith(response_prefix):
        raise ValueError("--assistant-prefill must end with --response-prefix.")
    mapper = CSIPrefixMapper(**dict(checkpoint["mapper_config"])).to(device)
    expected_hidden_size = language_model_hidden_size(qwen)
    mapper_hidden_size = int(checkpoint["mapper_config"]["qwen_hidden_size"])
    if mapper_hidden_size != expected_hidden_size:
        raise ValueError(
            "CSI-prefix mapper/language-model hidden-size mismatch: "
            f"mapper={mapper_hidden_size} model={expected_hidden_size}"
        )
    mapper.csi_encoder.load_state_dict(checkpoint["csi_encoder"], strict=True)
    mapper.projector.load_state_dict(checkpoint["projector"], strict=True)
    mapper.eval()

    samples = PreprocessedCSIDataset.from_pt(args.data_path).samples[: args.limit]
    if len(samples) != args.limit:
        raise ValueError(f"Requested {args.limit} samples, found {len(samples)}.")

    @torch.inference_mode()
    def predict(sample) -> tuple[dict[str, Any], int, bool, bool]:
        batch = move_csi_batch(collate_csi_samples([sample]), device)
        prefix = mapper(batch).to(dtype=qwen.get_input_embeddings().weight.dtype)
        prompt_ids = tokenizer(
            mapped_chat_text(tokenizer, sample) + assistant_prefill,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        prompt_embeddings = qwen.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([prefix, prompt_embeddings], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )
        output_ids = qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids, _ = generation_token_suffix(
            output_ids,
            input_embedding_length=int(inputs_embeds.shape[1]),
        )
        generated_text = response_prefix + tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        record, description, parse_error = parse_generated_response(generated_text)
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_normal_stop = any(
            bool(generated_ids.eq(int(eos_id)).any()) for eos_id in (eos_ids or [])
        )
        hit_max_new_tokens = int(generated_ids.numel()) >= args.max_new_tokens
        return (
            {
                "generated_text": generated_text,
                "parsed_record": record,
                "parsed_description": description,
                "parse_error": parse_error,
            },
            int(generated_ids.numel()),
            eos_normal_stop,
            hit_max_new_tokens,
        )

    with torch.inference_mode():
        for warmup_index, sample in enumerate(
            samples[: args.warmup_samples],
            start=1,
        ):
            predict(sample)
            if (
                warmup_index % args.warmup_log_every == 0
                or warmup_index == args.warmup_samples
            ):
                print(
                    f"benchmark_warmup_completed={warmup_index}/"
                    f"{args.warmup_samples}",
                    flush=True,
                )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        rows: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        targets: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            for index, sample in enumerate(samples):
                result = timed_call(lambda sample=sample: predict(sample), device)
                (
                    prediction,
                    generated_tokens,
                    eos_normal_stop,
                    hit_max_new_tokens,
                ) = result.value
                rows.append(
                    {
                        "model": args.model_name,
                        "seed": args.seed,
                        "repeat": repeat,
                        "sample_index": index,
                        "group_id": str(getattr(sample, "group_id", "")),
                        "config_key": str(getattr(sample, "config_key", "")),
                        "wall_ms": result.wall_ms,
                        "cuda_ms": result.cuda_ms,
                        "generated_tokens": generated_tokens,
                        "eos_normal_stop": int(eos_normal_stop),
                        "hit_max_new_tokens": int(hit_max_new_tokens),
                    }
                )
                if repeat == 0:
                    predictions.append(
                        {
                            "index": index,
                            "group_id": str(getattr(sample, "group_id", "")),
                            "config_key": str(getattr(sample, "config_key", "")),
                            **prediction,
                        }
                    )
                    response = target_response(sample)
                    targets.append(
                        {
                            "index": index,
                            "group_id": str(getattr(sample, "group_id", "")),
                            "config_key": str(getattr(sample, "config_key", "")),
                            "target_response": response,
                            "target_text": response["description"],
                        }
                    )
                if (index + 1) % args.log_every == 0:
                    print(
                        f"benchmark_repeat={repeat + 1}/{args.repeats} "
                        f"completed={index + 1}/{args.limit}",
                        flush=True,
                    )

    total_parameters, _ = parameter_counts(qwen, mapper)
    mapper_args = checkpoint.get("args", {})
    csi_trainable = not bool(mapper_args.get("freeze_csi_encoder", False))
    mapper_trainable = sum(parameter.numel() for parameter in mapper.projector.parameters())
    if csi_trainable:
        mapper_trainable += sum(
            parameter.numel() for parameter in mapper.csi_encoder.parameters()
        )
    lora_parameters = sum(
        parameter.numel()
        for name, parameter in qwen.named_parameters()
        if "lora_" in name
    )
    trainable_parameters = mapper_trainable + lora_parameters
    model_path = Path(args.model_path)
    model_weight_paths = sorted(model_path.glob("*.safetensors"))
    model_metadata_paths = [
        path
        for path in (
            model_path / "config.json",
            model_path / "model.safetensors.index.json",
        )
        if path.is_file()
    ]
    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    adapter_weight_paths = (
        sorted(adapter_path.glob("adapter_model.*")) if adapter_path else []
    )
    summary = summarize_cost(
        model=args.model_name,
        seed=args.seed,
        rows=rows,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        device=device,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        metadata={
            "repeats": args.repeats,
            "quantization": "4bit-nf4" if args.load_in_4bit else str(dtype),
            "language_model_compute_dtype": str(dtype),
            "cpu_offload": False,
            "model_type": str(getattr(qwen.config, "model_type", "")),
            "multimodal_backbone": is_qwen35_model(qwen),
            "vision_inputs_used": False,
            "vision_encoder_frozen": is_qwen35_model(qwen),
            "hf_device_map": json.dumps(device_map, sort_keys=True),
            "data_sha256": resolve_file_sha256(args.data_path, args.data_sha256),
            "mapping_checkpoint_sha256": sha256_file(args.mapping_checkpoint),
            "adapter_sha256": json.dumps(
                sha256_file_map(adapter_weight_paths), sort_keys=True
            ),
            "language_model_sha256": json.dumps(
                sha256_file_map(model_metadata_paths + model_weight_paths),
                sort_keys=True,
            ),
            "parse_success_rate": sum(
                prediction["parse_error"] is None for prediction in predictions
            )
            / len(predictions),
            "eos_normal_stop_rate": sum(
                int(row["eos_normal_stop"]) for row in rows
            )
            / len(rows),
            "hit_max_new_tokens_rate": sum(
                int(row["hit_max_new_tokens"]) for row in rows
            )
            / len(rows),
        },
    )
    output = Path(args.output_dir)
    config = vars(args) | {
        "resolved_assistant_prefill": assistant_prefill,
        "resolved_response_prefix": response_prefix,
    }
    write_benchmark_outputs(output, rows=rows, summary=summary, config=config)
    write_environment(output, device)
    for filename, values in (("predictions.jsonl", predictions), ("targets.jsonl", targets)):
        with (output / filename).open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=True) + "\n")
    print("benchmark_cost_summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
