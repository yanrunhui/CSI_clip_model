from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
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
    language_model_hidden_size,
    load_base_language_model,
    lora_target_modules,
    set_language_model_use_cache,
)
from scripts.qwen_csi_prefix_common import (  # noqa: E402
    MapperConfig,
    build_training_embeddings,
    collate_csi_samples,
    load_csi_encoder_checkpoint,
    move_csi_batch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_qwen(args, dtype: torch.dtype):
    qwen = load_base_language_model(args, dtype)
    for parameter in qwen.parameters():
        parameter.requires_grad = False

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.load_in_4bit and not args.skip_kbit_preparation:
            qwen = prepare_model_for_kbit_training(
                qwen,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        elif args.load_in_4bit:
            print(
                "kbit_preparation=skipped "
                "base_model_frozen=true input_gradients_enabled_later=true",
                flush=True,
            )
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_target_modules(qwen),
            bias="none",
        )
        qwen = get_peft_model(qwen, config)
    auxiliary_frozen = freeze_qwen35_auxiliary_modules(qwen)
    assert_qwen35_auxiliary_modules_frozen(qwen)
    print(
        "frozen_auxiliary_parameters=" + json.dumps(auxiliary_frozen, sort_keys=True),
        flush=True,
    )
    if args.gradient_checkpointing and args.use_lora:
        qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        qwen.enable_input_require_grads()
    set_language_model_use_cache(qwen, False)
    return qwen


def save_mapping_checkpoint(path: Path, mapper, args, step: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "csi_encoder": mapper.csi_encoder.state_dict(),
            "projector": mapper.projector.state_dict(),
            "mapper_config": metadata["mapper_config"],
            "args": vars(args),
            "step": step,
            "metadata": metadata,
        },
        path,
    )


CONTROLLED_COMPARISON_FIELDS = (
    "csi_checkpoint",
    "freeze_csi_encoder",
    "prefix_length",
    "projector_hidden_dim",
    "encoder_d_model",
    "encoder_d_clip",
    "token_norm_mode",
    "use_continuous_config_encoding",
    "batch_size",
    "gradient_accumulation_steps",
    "epochs",
    "max_steps",
    "max_length",
    "assistant_prefill",
    "response_prefix",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "grad_clip",
    "load_in_4bit",
    "compute_dtype",
    "skip_kbit_preparation",
    "use_lora",
    "gradient_checkpointing",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
)


def inherit_controlled_comparison_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.controlled_config_from is None:
        return None
    checkpoint = torch.load(
        args.controlled_config_from,
        map_location="cpu",
        weights_only=False,
    )
    reference_args = checkpoint.get("args")
    if not isinstance(reference_args, dict):
        raise ValueError(
            "Controlled-comparison checkpoint does not contain a saved args mapping: "
            f"{args.controlled_config_from}"
        )
    requested_max_steps = args.max_steps
    inherited: dict[str, Any] = {}
    missing = []
    for field in CONTROLLED_COMPARISON_FIELDS:
        if field not in reference_args:
            missing.append(field)
            continue
        value = reference_args[field]
        setattr(args, field, value)
        inherited[field] = value
    if requested_max_steps > 0:
        args.max_steps = requested_max_steps
        inherited["max_steps"] = requested_max_steps
    if missing:
        print(
            "controlled_config_missing_reference_fields=" + ",".join(missing),
            flush=True,
        )
    metadata = {
        "reference_mapping_checkpoint": args.controlled_config_from,
        "inherited_fields": inherited,
        "intentional_max_steps_override": (
            requested_max_steps
            if requested_max_steps > 0
            and requested_max_steps != reference_args.get("max_steps")
            else None
        ),
    }
    print("controlled_comparison_config=" + json.dumps(metadata, sort_keys=True))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an explicit CSIEncoder-to-Qwen soft-prefix mapping."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--csi-checkpoint")
    parser.add_argument(
        "--controlled-config-from",
        help=(
            "Inherit architecture, optimization, LoRA, quantization, and JSON "
            "format settings from an existing CSI-prefix mapping checkpoint. "
            "An explicitly positive --max-steps remains an intentional override."
        ),
    )
    parser.add_argument("--freeze-csi-encoder", action="store_true")
    parser.add_argument("--prefix-length", type=int, default=16)
    parser.add_argument("--projector-hidden-dim", type=int, default=1024)
    parser.add_argument("--encoder-d-model", type=int, default=384)
    parser.add_argument("--encoder-d-clip", type=int, default=256)
    parser.add_argument("--token-norm-mode", choices=("std", "rms", "none"), default="std")
    parser.add_argument("--use-continuous-config-encoding", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--assistant-prefill",
        default="",
        help="Masked assistant-side text inserted before the supervised response.",
    )
    parser.add_argument(
        "--response-prefix",
        default="",
        help=(
            "Prefix already supplied by --assistant-prefill and therefore removed "
            "from the supervised target, for example '{\"environment\":'."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Compute dtype for the causal LM and bitsandbytes 4-bit kernels.",
    )
    parser.add_argument(
        "--skip-kbit-preparation",
        action="store_true",
        help=(
            "Skip PEFT's full non-quantized-parameter FP32 conversion. Useful "
            "for large local Qwen3-derived models when that conversion fails; "
            "the base model remains frozen and LoRA/input gradients are still enabled."
        ),
    )
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    controlled_comparison = inherit_controlled_comparison_args(args)

    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive.")
    if args.response_prefix and not args.assistant_prefill.endswith(args.response_prefix):
        raise ValueError("--assistant-prefill must end with --response-prefix.")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PreprocessedCSIDataset.from_pt(args.train_data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_csi_samples,
        num_workers=args.num_workers,
        drop_last=True,
    )
    first_sample = dataset[0]
    dtype = resolve_compute_dtype(args.compute_dtype)
    print(f"language_model_compute_dtype={dtype}", flush=True)
    qwen = load_qwen(args, dtype)
    device = next(qwen.parameters()).device
    hidden_size = language_model_hidden_size(qwen)
    mapper_config = MapperConfig.from_args(
        args,
        qwen_hidden_size=hidden_size,
        d_token=int(first_sample.tokens.shape[1]),
    )
    mapper = mapper_config.build().to(device)
    csi_load_metadata = None
    if args.csi_checkpoint:
        csi_load_metadata = load_csi_encoder_checkpoint(mapper, args.csi_checkpoint)
    if args.freeze_csi_encoder:
        mapper.csi_encoder.eval()
        for parameter in mapper.csi_encoder.parameters():
            parameter.requires_grad = False

    trainable = [parameter for parameter in mapper.parameters() if parameter.requires_grad]
    trainable.extend(parameter for parameter in qwen.parameters() if parameter.requires_grad)
    if not trainable:
        raise ValueError("No trainable parameters remain.")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_steps = (
        args.max_steps if args.max_steps > 0 else max(args.epochs * updates_per_epoch, 1)
    )
    warmup_steps = int(round(total_steps * args.warmup_ratio))

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    metadata = {
        "architecture": "CSIEncoder -> MLP projector -> causal-LM soft prefix",
        "mapper_config": mapper_config.__dict__,
        "csi_checkpoint_load": csi_load_metadata,
        "train_samples": len(dataset),
        "language_model": args.model_path,
        "language_model_architectures": list(
            getattr(qwen.config, "architectures", None) or []
        ),
        "language_model_hidden_size": hidden_size,
        "qwen_model": args.model_path,
        "qwen_frozen": not args.use_lora,
        "use_lora": args.use_lora,
        "controlled_comparison": controlled_comparison,
    }
    print("csi_prefix_training_metadata=" + json.dumps(metadata, sort_keys=True))
    print(f"trainable_parameters={sum(p.numel() for p in trainable)}")
    print(f"total_optimizer_steps={total_steps}")

    qwen.train(args.use_lora)
    mapper.train()
    if args.freeze_csi_encoder:
        mapper.csi_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_micro_batches = 0
    start_time = time.perf_counter()
    stop = False
    autocast_context = (
        lambda: torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    use_grad_scaler = device.type == "cuda" and dtype == torch.float16
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    print(f"gradient_scaler_enabled={use_grad_scaler}", flush=True)
    for epoch in range(args.epochs if args.max_steps <= 0 else 10**9):
        for batch in loader:
            batch = move_csi_batch(batch, device)
            with autocast_context():
                prefix = mapper(batch)
                inputs_embeds, attention_mask, labels = build_training_embeddings(
                    qwen=qwen,
                    tokenizer=tokenizer,
                    prefix_embeddings=prefix,
                    samples=batch["samples"],
                    max_length=args.max_length,
                    assistant_prefill=args.assistant_prefill,
                    response_prefix=args.response_prefix,
                )
                output = qwen(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                if not bool(torch.isfinite(output.loss)):
                    raise FloatingPointError(
                        "Non-finite CSI-prefix training loss detected before "
                        f"optimizer step {global_step + 1}: {output.loss.item()}. "
                        "No final checkpoint will be saved. Prefer bfloat16 on "
                        "supported GPUs or reduce the learning rate."
                    )
                loss = output.loss / args.gradient_accumulation_steps
            grad_scaler.scale(loss).backward()
            running_loss += float(output.loss.detach().cpu())
            running_micro_batches += 1
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            if use_grad_scaler:
                grad_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.grad_clip,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    "Non-finite CSI-prefix gradient norm detected before "
                    f"optimizer step {global_step + 1}: {grad_norm.item()}."
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step == 1 or global_step % args.log_every == 0:
                elapsed = time.perf_counter() - start_time
                average_loss = running_loss / max(running_micro_batches, 1)
                print(
                    f"step={global_step}/{total_steps} epoch={epoch + 1} "
                    f"loss={average_loss:.6f} lr={scheduler.get_last_lr()[0]:.3e} "
                    f"elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )
                running_loss = 0.0
                running_micro_batches = 0
            if args.save_every > 0 and global_step % args.save_every == 0:
                save_mapping_checkpoint(
                    output_dir / f"mapping_step_{global_step}.pt",
                    mapper,
                    args,
                    global_step,
                    metadata,
                )
            if global_step >= total_steps:
                stop = True
                break
        if stop:
            break

    final_path = output_dir / "mapping_final.pt"
    save_mapping_checkpoint(final_path, mapper, args, global_step, metadata)
    adapter_path = None
    if args.use_lora:
        adapter_path = output_dir / "qwen_adapter"
        qwen.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
    metadata.update(
        {
            "completed_steps": global_step,
            "elapsed_seconds": time.perf_counter() - start_time,
            "mapping_checkpoint": str(final_path),
            "adapter_path": str(adapter_path) if adapter_path else None,
        }
    )
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved_csi_prefix_mapping={final_path}")
    if adapter_path:
        print(f"saved_qwen_adapter={adapter_path}")


if __name__ == "__main__":
    main()
