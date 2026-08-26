from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch


INTERNLM3_MODEL_TYPE = "internlm3"
REQUIRED_FILES = (
    "config.json",
    "configuration_internlm3.py",
    "modeling_internlm3.py",
    "tokenization_internlm3.py",
    "tokenizer.model",
    "tokenizer_config.json",
)


def read_internlm3_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"InternLM3 model directory not found: {path}")
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"InternLM3 model directory {path} is missing: {', '.join(missing)}"
        )
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if str(config.get("model_type", "")) != INTERNLM3_MODEL_TYPE:
        raise ValueError(
            f"Expected model_type={INTERNLM3_MODEL_TYPE!r}, got "
            f"{config.get('model_type')!r} in {path}."
        )
    index_path = path / "model.safetensors.index.json"
    single_weights = path / "model.safetensors"
    if not index_path.is_file() and not single_weights.is_file():
        raise FileNotFoundError(f"No safetensors weights found in {path}.")
    return config


def validate_internlm3_model_path(model_path: str | Path) -> dict[str, Any]:
    config = read_internlm3_config(model_path)
    metadata = {
        "model_path": str(model_path),
        "model_type": config["model_type"],
        "architectures": config.get("architectures", []),
        "hidden_size": int(config.get("hidden_size", 0)),
        "num_hidden_layers": int(config.get("num_hidden_layers", 0)),
        "torch_dtype": config.get("torch_dtype"),
        "remote_code_required": bool(config.get("auto_map")),
    }
    print(
        "internlm3_model_validation="
        + json.dumps(metadata, ensure_ascii=True, sort_keys=True),
        flush=True,
    )
    return metadata


def load_internlm3_tokenizer(model_path: str | Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_internlm3_model(args: Any, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    validate_internlm3_model_path(args.model_path)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    print(
        "internlm3_language_model_loader="
        + json.dumps(
            {
                "class": model.__class__.__name__,
                "model_type": getattr(model.config, "model_type", ""),
                "hidden_size": language_model_hidden_size(model),
                "trust_remote_code": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return model


def language_model_hidden_size(model: torch.nn.Module) -> int:
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("InternLM3 config does not expose hidden_size.")
    return int(hidden_size)


def internlm3_lora_target_modules(model: torch.nn.Module) -> list[str]:
    available_suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    expected = ("q_proj", "k_proj", "v_proj", "o_proj")
    missing = [name for name in expected if name not in available_suffixes]
    if missing:
        raise RuntimeError(
            "InternLM3 attention projections required for LoRA are missing: "
            + ", ".join(missing)
        )
    return list(expected)


def set_internlm3_use_cache(model: torch.nn.Module, enabled: bool) -> None:
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = enabled


def interface_metadata(model: torch.nn.Module) -> dict[str, Any]:
    forward_parameters = inspect.signature(model.forward).parameters
    generation_parameters = inspect.signature(
        model.prepare_inputs_for_generation
    ).parameters
    metadata = {
        "class": model.__class__.__name__,
        "model_type": str(getattr(model.config, "model_type", "")),
        "hidden_size": language_model_hidden_size(model),
        "forward_supports_inputs_embeds": "inputs_embeds" in forward_parameters,
        "forward_supports_labels": "labels" in forward_parameters,
        "generation_supports_inputs_embeds": (
            "inputs_embeds" in generation_parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in generation_parameters.values()
            )
        ),
    }
    return metadata


def assert_soft_prefix_interface(model: torch.nn.Module) -> dict[str, Any]:
    metadata = interface_metadata(model)
    required = (
        "forward_supports_inputs_embeds",
        "forward_supports_labels",
        "generation_supports_inputs_embeds",
    )
    if not all(bool(metadata[name]) for name in required):
        raise RuntimeError(f"InternLM3 soft-prefix interface check failed: {metadata}")
    print(
        "internlm3_csi_prefix_interface="
        + json.dumps(metadata, sort_keys=True),
        flush=True,
    )
    return metadata
