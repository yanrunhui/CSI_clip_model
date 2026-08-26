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
from scripts.evaluate_internlm3_csi_prefix import (  # noqa: E402
    generation_eos_ids,
    load_internlm3_for_evaluation,
    normalize_complete_json_code_fence,
)
from scripts.evaluate_qwen_csi_prefix import (  # noqa: E402
    generation_token_suffix,
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
from scripts.internlm3_csi_prefix_common import (  # noqa: E402
    language_model_hidden_size,
    load_internlm3_tokenizer,
)
from scripts.qwen_csi_prefix_common import (  # noqa: E402
    CSIPrefixMapper,
    collate_csi_samples,
    mapped_chat_text,
    move_csi_batch,
)
from scripts.qwen_csi_text_common import (  # noqa: E402
    parse_generated_response,
    target_response,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the InternLM3 CSI soft-prefix baseline."
    )
    parser.add_argument("--model-name", default="InternLM3-8B-Instruct")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mapping-checkpoint", required=True)
    parser.add_argument("--adapter-path", required=True)
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
    args = parser.parse_args()
    if args.limit <= 0 or args.warmup_samples < 0 or args.repeats <= 0:
        raise ValueError("limit/repeats must be positive and warmup nonnegative.")
    if not torch.cuda.is_available():
        raise RuntimeError("The InternLM3 cost benchmark requires CUDA.")

    tokenizer = load_internlm3_tokenizer(args.model_path)
    dtype = resolve_compute_dtype(args.compute_dtype)
    model = load_internlm3_for_evaluation(args, dtype)
    device_map = getattr(model, "hf_device_map", {}) or getattr(
        getattr(model, "base_model", None), "hf_device_map", {}
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
            "CPU/disk offload is not allowed in the benchmark: "
            f"{offloaded_devices}"
        )
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("InternLM3 must reside on CUDA for comparable timing.")

    checkpoint = torch.load(
        args.mapping_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
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
    expected_hidden_size = language_model_hidden_size(model)
    mapper_hidden_size = int(checkpoint["mapper_config"]["qwen_hidden_size"])
    if mapper_hidden_size != expected_hidden_size:
        raise ValueError(
            "CSI mapper/InternLM3 hidden-size mismatch: "
            f"mapper={mapper_hidden_size} model={expected_hidden_size}"
        )
    mapper.csi_encoder.load_state_dict(checkpoint["csi_encoder"], strict=True)
    mapper.projector.load_state_dict(checkpoint["projector"], strict=True)
    mapper.eval()

    samples = PreprocessedCSIDataset.from_pt(args.data_path).samples[: args.limit]
    if len(samples) != args.limit:
        raise ValueError(f"Requested {args.limit} samples, found {len(samples)}.")
    eos_ids = generation_eos_ids(model, tokenizer)
    eos_id_list = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])

    @torch.inference_mode()
    def predict(sample) -> tuple[dict[str, Any], int, bool, bool]:
        batch = move_csi_batch(collate_csi_samples([sample]), device)
        prefix = mapper(batch).to(dtype=model.get_input_embeddings().weight.dtype)
        prompt_ids = tokenizer(
            mapped_chat_text(tokenizer, sample) + assistant_prefill,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        prompt_embeddings = model.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([prefix, prompt_embeddings], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
        generated_ids, _ = generation_token_suffix(
            output_ids,
            input_embedding_length=int(inputs_embeds.shape[1]),
        )
        raw_text = response_prefix + tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        _, _, strict_parse_error = parse_generated_response(raw_text)
        normalized_text, normalization = normalize_complete_json_code_fence(raw_text)
        record, description, parse_error = parse_generated_response(normalized_text)
        eos_normal_stop = any(
            bool(generated_ids.eq(int(eos_id)).any()) for eos_id in eos_id_list
        )
        hit_max_new_tokens = int(generated_ids.numel()) >= args.max_new_tokens
        return (
            {
                "generated_text": raw_text,
                "normalized_generated_text": normalized_text,
                "parsed_record": record,
                "parsed_description": description,
                "parse_error": parse_error,
                "strict_parse_error": strict_parse_error,
                "format_normalization": normalization,
            },
            int(generated_ids.numel()),
            eos_normal_stop,
            hit_max_new_tokens,
        )

    with torch.inference_mode():
        for sample in samples[: args.warmup_samples]:
            predict(sample)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        rows: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        targets: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            for index, sample in enumerate(samples):
                result = timed_call(lambda sample=sample: predict(sample), device)
                prediction, generated_tokens, eos_stop, hit_limit = result.value
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
                        "eos_normal_stop": int(eos_stop),
                        "hit_max_new_tokens": int(hit_limit),
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
                        f"internlm3_benchmark_repeat={repeat + 1}/{args.repeats} "
                        f"completed={index + 1}/{args.limit}",
                        flush=True,
                    )

    total_parameters, _ = parameter_counts(model, mapper)
    mapper_args = checkpoint.get("args", {})
    mapper_trainable = sum(parameter.numel() for parameter in mapper.projector.parameters())
    if not bool(mapper_args.get("freeze_csi_encoder", False)):
        mapper_trainable += sum(
            parameter.numel() for parameter in mapper.csi_encoder.parameters()
        )
    lora_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )
    trainable_parameters = mapper_trainable + lora_parameters

    model_path = Path(args.model_path)
    model_paths = sorted(model_path.glob("*.safetensors"))
    model_paths.extend(
        path
        for path in (
            model_path / "config.json",
            model_path / "model.safetensors.index.json",
            model_path / "configuration_internlm3.py",
            model_path / "modeling_internlm3.py",
            model_path / "tokenization_internlm3.py",
        )
        if path.is_file()
    )
    adapter_path = Path(args.adapter_path)
    adapter_weights = sorted(adapter_path.glob("adapter_model.*"))
    parse_rate = sum(row["parse_error"] is None for row in predictions) / len(predictions)
    strict_rate = sum(
        row["strict_parse_error"] is None for row in predictions
    ) / len(predictions)
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
            "model_type": str(getattr(model.config, "model_type", "")),
            "trust_remote_code": True,
            "hf_device_map": json.dumps(device_map, sort_keys=True),
            "data_sha256": resolve_file_sha256(args.data_path, args.data_sha256),
            "mapping_checkpoint_sha256": sha256_file(args.mapping_checkpoint),
            "adapter_sha256": json.dumps(
                sha256_file_map(adapter_weights), sort_keys=True
            ),
            "language_model_sha256": json.dumps(
                sha256_file_map(model_paths), sort_keys=True
            ),
            "parse_success_rate": parse_rate,
            "strict_json_format_rate": strict_rate,
            "format_normalization_rate": 1.0 - strict_rate,
            "eos_normal_stop_rate": sum(int(row["eos_normal_stop"]) for row in rows)
            / len(rows),
            "hit_max_new_tokens_rate": sum(
                int(row["hit_max_new_tokens"]) for row in rows
            )
            / len(rows),
        },
    )
    output = Path(args.output_dir)
    write_benchmark_outputs(
        output,
        rows=rows,
        summary=summary,
        config=vars(args)
        | {
            "resolved_assistant_prefill": assistant_prefill,
            "resolved_response_prefix": response_prefix,
        },
    )
    write_environment(output, device)
    for filename, values in (("predictions.jsonl", predictions), ("targets.jsonl", targets)):
        with (output / filename).open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=True) + "\n")
    print("internlm3_benchmark_cost_summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
