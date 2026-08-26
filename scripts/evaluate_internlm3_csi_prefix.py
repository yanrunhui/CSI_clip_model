from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset  # noqa: E402
from scripts.evaluate_qwen_csi_prefix import (  # noqa: E402
    generation_token_suffix,
    prepare_resume_files,
    resolve_compute_dtype,
)
from scripts.internlm3_csi_prefix_common import (  # noqa: E402
    assert_soft_prefix_interface,
    load_internlm3_model,
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


def load_internlm3_for_evaluation(args: argparse.Namespace, dtype: torch.dtype):
    model = load_internlm3_model(args, dtype)
    assert_soft_prefix_interface(model)
    if args.adapter_path:
        from peft import PeftModel

        adapter_config = Path(args.adapter_path) / "adapter_config.json"
        if not adapter_config.is_file():
            raise FileNotFoundError(f"InternLM3 adapter is missing: {adapter_config}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model


def generation_eos_ids(model, tokenizer) -> int | list[int]:
    eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    return eos_ids


def normalize_complete_json_code_fence(text: str) -> tuple[str, str | None]:
    """Remove only a code fence that encloses the complete model response."""
    stripped = text.strip()
    lines = stripped.splitlines()
    if len(lines) < 3:
        return text, None
    opening = lines[0].strip().lower()
    closing = lines[-1].strip()
    if opening not in {"```", "```json"} or closing != "```":
        return text, None
    inner = "\n".join(lines[1:-1]).strip()
    if not inner.startswith("{") or not inner.endswith("}"):
        return text, None
    return inner, "complete_json_code_fence"


def retained_strict_success_count(path: Path, retained_count: int) -> int:
    if retained_count <= 0 or not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= retained_count:
                break
            row = json.loads(line)
            count += int(row.get("strict_parse_error") is None)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CSI descriptions with an InternLM3 soft-prefix baseline."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mapping-checkpoint", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--assistant-prefill", default=None)
    parser.add_argument("--response-prefix", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError("InternLM3 CSI-prefix generation currently requires batch size 1.")

    tokenizer = load_internlm3_tokenizer(args.model_path)
    dtype = resolve_compute_dtype(args.compute_dtype)
    print(f"language_model_compute_dtype={dtype}", flush=True)
    model = load_internlm3_for_evaluation(args, dtype)
    device = next(model.parameters()).device
    checkpoint = torch.load(
        args.mapping_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_args = checkpoint.get("args", {})
    args.assistant_prefill = (
        args.assistant_prefill
        if args.assistant_prefill is not None
        else str(checkpoint_args.get("assistant_prefill", ""))
    )
    args.response_prefix = (
        args.response_prefix
        if args.response_prefix is not None
        else str(checkpoint_args.get("response_prefix", ""))
    )
    if args.response_prefix and not args.assistant_prefill.endswith(args.response_prefix):
        raise ValueError("--assistant-prefill must end with --response-prefix.")

    mapper = CSIPrefixMapper(**dict(checkpoint["mapper_config"])).to(device)
    mapper.csi_encoder.load_state_dict(checkpoint["csi_encoder"], strict=True)
    mapper.projector.load_state_dict(checkpoint["projector"], strict=True)
    mapper.eval()

    dataset = PreprocessedCSIDataset.from_pt(args.data_path)
    if args.limit is not None:
        dataset.samples = dataset.samples[: args.limit]
    total_sample_count = len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_csi_samples,
    )
    output_path = Path(args.output_jsonl)
    target_path = Path(args.data_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        resume_count, parse_success = prepare_resume_files(output_path, target_path)
        strict_parse_success = retained_strict_success_count(output_path, resume_count)
    else:
        resume_count, parse_success = 0, 0
        strict_parse_success = 0
    if resume_count > total_sample_count:
        raise ValueError(
            f"Resume rows ({resume_count}) exceed samples ({total_sample_count})."
        )
    if resume_count:
        dataset.samples = dataset.samples[resume_count:]
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_csi_samples,
        )

    output_mode = "a" if args.resume else "w"
    eos_ids = generation_eos_ids(model, tokenizer)
    with output_path.open(output_mode, encoding="utf-8") as prediction_handle, target_path.open(
        output_mode, encoding="utf-8"
    ) as target_handle:
        for index, batch in enumerate(loader, start=resume_count):
            batch = move_csi_batch(batch, device)
            sample = batch["samples"][0]
            with torch.inference_mode():
                prefix = mapper(batch).to(dtype=model.get_input_embeddings().weight.dtype)
                prompt_ids = tokenizer(
                    mapped_chat_text(tokenizer, sample) + args.assistant_prefill,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(device)
                prompt_embeddings = model.get_input_embeddings()(prompt_ids)
                inputs_embeds = torch.cat([prefix, prompt_embeddings], dim=1)
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    dtype=torch.long,
                    device=device,
                )
                output_ids = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
            generated_ids, stripped_placeholder_count = generation_token_suffix(
                output_ids,
                input_embedding_length=int(inputs_embeds.shape[1]),
            )
            generated_text = args.response_prefix + tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            _, _, strict_parse_error = parse_generated_response(generated_text)
            normalized_text, format_normalization = normalize_complete_json_code_fence(
                generated_text
            )
            record, description, parse_error = parse_generated_response(normalized_text)
            strict_parse_success += int(strict_parse_error is None)
            parse_success += int(parse_error is None)
            prediction = {
                "index": index,
                "group_id": str(getattr(sample, "group_id", "")),
                "config_key": str(getattr(sample, "config_key", "")),
                "generated_text": generated_text,
                "normalized_generated_text": normalized_text,
                "parsed_record": record,
                "parsed_description": description,
                "parse_error": parse_error,
                "strict_parse_error": strict_parse_error,
                "format_normalization": format_normalization,
                "input_embedding_token_count": int(inputs_embeds.shape[1]),
                "raw_output_token_count": int(output_ids.shape[1]),
                "generated_token_count": int(generated_ids.numel()),
                "stripped_placeholder_token_count": stripped_placeholder_count,
                "input_was_truncated": False,
                "mapping_checkpoint": args.mapping_checkpoint,
                "adapter_path": args.adapter_path,
                "assistant_prefill": args.assistant_prefill,
                "response_prefix": args.response_prefix,
                "baseline": "InternLM3-8B-Instruct direct CSI soft-prefix decoder",
            }
            response = target_response(sample)
            target_row = {
                "index": index,
                "group_id": str(getattr(sample, "group_id", "")),
                "config_key": str(getattr(sample, "config_key", "")),
                "prompt": mapped_chat_text(tokenizer, sample),
                "target_response": response,
                "target_text": response["description"],
                "serialization": {
                    "representation": "csi_encoder_soft_prefix",
                    "prefix_length": mapper.prefix_length,
                    "language_model": "InternLM3-8B-Instruct",
                },
            }
            prediction_handle.write(json.dumps(prediction, ensure_ascii=True) + "\n")
            target_handle.write(json.dumps(target_row, ensure_ascii=True) + "\n")
            prediction_handle.flush()
            target_handle.flush()
            if (index + 1) % args.log_every == 0:
                print(
                    f"internlm3_csi_prefix_generated={index + 1} "
                    f"parse_success_rate={parse_success / (index + 1):.6f} "
                    f"strict_json_format_rate={strict_parse_success / (index + 1):.6f}",
                    flush=True,
                )

    summary = {
        "baseline": "InternLM3-8B-Instruct direct CSI soft-prefix decoder",
        "sample_count": total_sample_count,
        "parse_success_rate": (
            parse_success / total_sample_count if total_sample_count else None
        ),
        "strict_json_format_rate": (
            strict_parse_success / total_sample_count if total_sample_count else None
        ),
        "mapping_checkpoint": args.mapping_checkpoint,
        "adapter_path": args.adapter_path,
        "output_jsonl": str(output_path),
        "data_jsonl": str(target_path),
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("internlm3_csi_prefix_generation_summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
