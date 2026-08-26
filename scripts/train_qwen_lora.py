from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen_csi_text_common import training_text  # noqa: E402


class IndexedJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.offsets = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"No JSONL examples found in {self.path}.")
        self._handle = None
        self._handle_pid = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _file_handle(self):
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            if self._handle is not None:
                self._handle.close()
            self._handle = self.path.open("rb")
            self._handle_pid = pid
        return self._handle

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        handle = self._file_handle()
        handle.seek(self.offsets[index])
        row = json.loads(handle.readline())
        response_text = json.dumps(
            row["target_response"],
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        prompt_text, answer_text = training_text(
            self.tokenizer,
            str(row["prompt"]),
            response_text,
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        answer_ids = self.tokenizer(
            answer_text,
            add_special_tokens=False,
        )["input_ids"]
        if len(answer_ids) >= self.max_length:
            raise ValueError(
                f"Target answer at index {index} uses {len(answer_ids)} tokens, "
                f"which exceeds max_length={self.max_length}."
            )
        prompt_budget = self.max_length - len(answer_ids)
        prompt_ids = prompt_ids[:prompt_budget]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


class CausalLmCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Qwen CSI-to-text LoRA/QLoRA adapter.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise ImportError(
            "Install transformers>=4.51, accelerate, peft, and bitsandbytes."
        ) from error

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
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
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = IndexedJsonlDataset(args.train_jsonl, tokenizer, args.max_length)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLmCollator(tokenizer.pad_token_id),
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "model_path": args.model_path,
        "train_jsonl": args.train_jsonl,
        "sample_count": len(dataset),
        "max_length": args.max_length,
        "load_in_4bit": args.load_in_4bit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "train_metrics": result.metrics,
    }
    (output_dir / "qwen_lora_training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("qwen_lora_training_metadata=" + json.dumps(metadata, sort_keys=True))
    print(f"saved_qwen_lora_adapter={output_dir}")


if __name__ == "__main__":
    main()
