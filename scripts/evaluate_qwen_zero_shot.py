from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen_csi_text_common import (  # noqa: E402
    apply_chat_template,
    parse_generated_response,
)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            yield row


def completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    result = set()
    for row in iter_jsonl(path):
        result.add(int(row["index"]))
    return result


def load_model(args, dtype: torch.dtype):
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as error:
        raise ImportError(
            "Install transformers, accelerate, peft, and bitsandbytes in the Qwen environment."
        ) from error

    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": False,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if args.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise ImportError("Loading a LoRA adapter requires peft.") from error
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model


def batched(rows: Iterator[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic Qwen CSI-to-description JSON generation."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Token limits must be positive.")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError("Install transformers in the Qwen environment.") from error

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    model = load_model(args, dtype)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_indices(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    source_rows = (
        row
        for row in iter_jsonl(Path(args.input_jsonl))
        if int(row["index"]) not in done
    )

    generated_count = 0
    parse_success_count = 0
    truncated_count = 0
    with output_path.open(mode, encoding="utf-8") as output_handle:
        for batch in batched(source_rows, args.batch_size):
            if args.limit is not None:
                remaining = args.limit - generated_count
                if remaining <= 0:
                    break
                batch = batch[:remaining]

            chat_texts = [
                apply_chat_template(
                    tokenizer,
                    str(row["prompt"]),
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            original_lengths = [
                len(tokenizer(text, add_special_tokens=False)["input_ids"])
                for text in chat_texts
            ]
            encoded = tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
                add_special_tokens=False,
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for row, text, original_length in zip(batch, texts, original_lengths):
                record, description, parse_error = parse_generated_response(text)
                was_truncated = original_length > args.max_input_tokens
                result = {
                    "index": int(row["index"]),
                    "group_id": str(row.get("group_id", "")),
                    "config_key": str(row.get("config_key", "")),
                    "generated_text": text,
                    "parsed_record": record,
                    "parsed_description": description,
                    "parse_error": parse_error,
                    "input_token_count_before_truncation": original_length,
                    "input_was_truncated": was_truncated,
                    "model_path": args.model_path,
                    "adapter_path": args.adapter_path,
                }
                output_handle.write(
                    json.dumps(result, ensure_ascii=True, allow_nan=False) + "\n"
                )
                output_handle.flush()
                generated_count += 1
                parse_success_count += int(parse_error is None)
                truncated_count += int(was_truncated)

            if generated_count % args.log_every == 0:
                print(
                    f"qwen_generated={generated_count} "
                    f"parse_success_rate={parse_success_count / generated_count:.6f} "
                    f"input_truncation_rate={truncated_count / generated_count:.6f}",
                    flush=True,
                )

    summary = {
        "generated_count": generated_count,
        "preexisting_count": len(done),
        "parse_success_rate": (
            parse_success_count / generated_count if generated_count else None
        ),
        "input_truncation_rate": (
            truncated_count / generated_count if generated_count else None
        ),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("qwen_generation_summary=" + json.dumps(summary, sort_keys=True))
    print(f"saved_qwen_predictions={output_path}")


if __name__ == "__main__":
    main()
