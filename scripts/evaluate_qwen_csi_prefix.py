from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset  # noqa: E402
from scripts.csi_prefix_language_model import (  # noqa: E402
    assert_qwen35_auxiliary_modules_frozen,
    freeze_qwen35_auxiliary_modules,
    load_base_language_model,
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


def load_qwen(args, dtype: torch.dtype):
    qwen = load_base_language_model(args, dtype)
    if args.adapter_path:
        from peft import PeftModel

        qwen = PeftModel.from_pretrained(qwen, args.adapter_path)
    freeze_qwen35_auxiliary_modules(qwen)
    assert_qwen35_auxiliary_modules_frozen(qwen)
    qwen.eval()
    return qwen


def resolve_compute_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 was requested but the active CUDA device does not support it.")
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name != "auto":
        raise ValueError(f"Unsupported compute dtype: {name}")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def generation_token_suffix(
    output_ids: torch.Tensor,
    *,
    input_embedding_length: int,
) -> tuple[torch.Tensor, int]:
    """Remove placeholder ids that generate() may prepend for inputs_embeds.

    Some Transformers/model combinations represent an embedding-only input as
    token id 0 in the returned sequence. For Qwen tokenizers id 0 is ``!``, so
    decoding the complete sequence can look like a long exclamation-mark loop
    even though those ids only stand in for the CSI prefix and text prompt.
    """
    sequence = output_ids[0]
    stripped = 0
    if input_embedding_length > 0 and sequence.numel() >= input_embedding_length:
        possible_placeholder = sequence[:input_embedding_length]
        if bool(possible_placeholder.eq(0).all()):
            sequence = sequence[input_embedding_length:]
            stripped = input_embedding_length
    return sequence, stripped


def _read_jsonl_prefix(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"resume_discarding_invalid_jsonl_tail={path}:{line_number}",
                    flush=True,
                )
                break
            if not isinstance(row, dict):
                print(
                    f"resume_discarding_non_object_tail={path}:{line_number}",
                    flush=True,
                )
                break
            rows.append(row)
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".resume_tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(path)


def prepare_resume_files(output_path: Path, data_path: Path) -> tuple[int, int]:
    predictions = _read_jsonl_prefix(output_path)
    targets = _read_jsonl_prefix(data_path)
    common_count = 0
    for expected_index, (prediction, target) in enumerate(zip(predictions, targets)):
        if prediction.get("index") != expected_index or target.get("index") != expected_index:
            break
        if prediction.get("group_id") != target.get("group_id"):
            break
        common_count += 1

    retained_predictions = predictions[:common_count]
    retained_targets = targets[:common_count]
    _write_jsonl_rows(output_path, retained_predictions)
    _write_jsonl_rows(data_path, retained_targets)
    parse_success = sum(
        prediction.get("parse_error") is None
        for prediction in retained_predictions
    )
    print(
        "resume_reconciled "
        f"prediction_rows={len(predictions)} target_rows={len(targets)} "
        f"retained_rows={common_count} parse_success={parse_success}",
        flush=True,
    )
    return common_count, parse_success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate descriptions using a trained CSI-to-Qwen soft-prefix mapper."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mapping-checkpoint", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--assistant-prefill",
        default=None,
        help="Assistant-side control text embedded before generation.",
    )
    parser.add_argument(
        "--response-prefix",
        default=None,
        help="Visible response prefix prepended to decoded generated tokens.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted evaluation. Existing prediction and target "
            "JSONL files are reconciled to their last common complete row."
        ),
    )
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError("CSI-prefix generation currently requires --batch-size 1.")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = resolve_compute_dtype(args.compute_dtype)
    print(f"language_model_compute_dtype={dtype}", flush=True)
    qwen = load_qwen(args, dtype)
    device = next(qwen.parameters()).device
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
    if args.response_prefix and not args.assistant_prefill.endswith(
        args.response_prefix
    ):
        raise ValueError("--assistant-prefill must end with --response-prefix.")
    config = dict(checkpoint["mapper_config"])
    mapper = CSIPrefixMapper(**config).to(device)
    mapper.csi_encoder.load_state_dict(checkpoint["csi_encoder"])
    mapper.projector.load_state_dict(checkpoint["projector"])
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
    data_path = Path(args.data_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        resume_count, parse_success = prepare_resume_files(output_path, data_path)
    else:
        resume_count, parse_success = 0, 0
    if resume_count > total_sample_count:
        raise ValueError(
            f"Resume rows ({resume_count}) exceed evaluation samples ({total_sample_count})."
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
    with output_path.open(output_mode, encoding="utf-8") as prediction_handle, data_path.open(
        output_mode, encoding="utf-8"
    ) as data_handle:
        for index, batch in enumerate(loader, start=resume_count):
            batch = move_csi_batch(batch, device)
            sample = batch["samples"][0]
            with torch.inference_mode():
                prefix = mapper(batch).to(dtype=qwen.get_input_embeddings().weight.dtype)
                prompt_ids = tokenizer(
                    mapped_chat_text(tokenizer, sample) + args.assistant_prefill,
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
            generated_ids, stripped_placeholder_count = generation_token_suffix(
                output_ids,
                input_embedding_length=int(inputs_embeds.shape[1]),
            )
            generated_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            generated_text = args.response_prefix + generated_text
            record, description, parse_error = parse_generated_response(generated_text)
            parse_success += int(parse_error is None)
            prediction = {
                "index": index,
                "group_id": str(getattr(sample, "group_id", "")),
                "config_key": str(getattr(sample, "config_key", "")),
                "generated_text": generated_text,
                "parsed_record": record,
                "parsed_description": description,
                "parse_error": parse_error,
                "input_embedding_token_count": int(inputs_embeds.shape[1]),
                "raw_output_token_count": int(output_ids.shape[1]),
                "generated_token_count": int(generated_ids.numel()),
                "stripped_placeholder_token_count": stripped_placeholder_count,
                "input_was_truncated": False,
                "mapping_checkpoint": args.mapping_checkpoint,
                "adapter_path": args.adapter_path,
                "assistant_prefill": args.assistant_prefill,
                "response_prefix": args.response_prefix,
            }
            response = target_response(sample)
            data_row = {
                "index": index,
                "group_id": str(getattr(sample, "group_id", "")),
                "config_key": str(getattr(sample, "config_key", "")),
                "prompt": mapped_chat_text(tokenizer, sample),
                "target_response": response,
                "target_text": response["description"],
                "serialization": {
                    "representation": "csi_encoder_soft_prefix",
                    "prefix_length": mapper.prefix_length,
                },
            }
            prediction_handle.write(json.dumps(prediction, ensure_ascii=True) + "\n")
            data_handle.write(json.dumps(data_row, ensure_ascii=True) + "\n")
            prediction_handle.flush()
            data_handle.flush()
            if (index + 1) % args.log_every == 0:
                print(
                    f"csi_prefix_generated={index + 1} "
                    f"parse_success_rate={parse_success / (index + 1):.6f}",
                    flush=True,
                )
    summary = {
        "sample_count": total_sample_count,
        "parse_success_rate": (
            parse_success / total_sample_count if total_sample_count else None
        ),
        "mapping_checkpoint": args.mapping_checkpoint,
        "adapter_path": args.adapter_path,
        "output_jsonl": str(output_path),
        "data_jsonl": str(data_path),
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("csi_prefix_generation_summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
