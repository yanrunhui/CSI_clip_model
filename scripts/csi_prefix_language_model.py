from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch


QWEN35_MODEL_TYPE = "qwen3_5"


def read_model_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing language-model config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Language-model config must be a JSON object: {path}")
    return config


def declared_model_type(model_path: str | Path) -> str:
    return str(read_model_config(model_path).get("model_type", ""))


def is_qwen35_model(model_or_path: Any) -> bool:
    if isinstance(model_or_path, (str, Path)):
        return declared_model_type(model_or_path) == QWEN35_MODEL_TYPE
    return str(getattr(model_or_path.config, "model_type", "")) == QWEN35_MODEL_TYPE


def language_model_hidden_size(model: torch.nn.Module) -> int:
    config = model.config
    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            f"Cannot resolve hidden size from {model.__class__.__name__} config."
        )
    return int(hidden_size)


def load_base_language_model(args: Any, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_type = declared_model_type(args.model_path)
    if model_type == QWEN35_MODEL_TYPE:
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as error:
            try:
                import transformers

                version = transformers.__version__
            except Exception:
                version = "unknown"
            raise RuntimeError(
                "This checkpoint requires Qwen3_5ForConditionalGeneration, but "
                f"Transformers {version} does not provide it. Install the "
                "Qwen3.5 environment requirements (Transformers >= 5.6.2)."
            ) from error
        model_class = Qwen3_5ForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": False,
    }
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif model_type == QWEN35_MODEL_TYPE:
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    model = model_class.from_pretrained(args.model_path, **kwargs)
    print(
        "language_model_loader="
        + json.dumps(
            {
                "class": model.__class__.__name__,
                "model_type": model_type,
                "multimodal": model_type == QWEN35_MODEL_TYPE,
                "vision_inputs_used": False,
                "hidden_size": language_model_hidden_size(model),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return model


def lora_target_modules(model: torch.nn.Module) -> list[str]:
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if is_qwen35_model(model):
        targets.extend(["in_proj_qkv", "out_proj"])
    return targets


def freeze_qwen35_auxiliary_modules(model: torch.nn.Module) -> dict[str, int]:
    frozen = {"visual": 0, "mtp": 0}
    if not is_qwen35_model(model):
        return frozen
    for name, parameter in model.named_parameters():
        category = None
        if ".visual." in name:
            category = "visual"
        elif ".mtp." in name or name.startswith("mtp."):
            category = "mtp"
        if category is not None:
            parameter.requires_grad = False
            frozen[category] += parameter.numel()
    return frozen


def assert_qwen35_auxiliary_modules_frozen(model: torch.nn.Module) -> None:
    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            ".visual." in name
            or ".mtp." in name
            or name.startswith("mtp.")
        )
    ]
    if trainable:
        raise RuntimeError(
            "Qwen3.5 visual/MTP parameters unexpectedly remain trainable: "
            + ", ".join(trainable[:10])
        )


def set_language_model_use_cache(model: torch.nn.Module, enabled: bool) -> None:
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = enabled
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = enabled


def qwen35_interface_metadata(model_path: str | Path) -> dict[str, Any]:
    config = read_model_config(model_path)
    if str(config.get("model_type", "")) != QWEN35_MODEL_TYPE:
        raise ValueError(f"Not a Qwen3.5 checkpoint: {model_path}")
    try:
        from transformers import Qwen3_5ForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "Qwen3_5ForConditionalGeneration is unavailable. Install "
            "Transformers >= 5.6.2 before running the smoke test."
        ) from error
    forward_parameters = inspect.signature(
        Qwen3_5ForConditionalGeneration.forward
    ).parameters
    generation_parameters = inspect.signature(
        Qwen3_5ForConditionalGeneration.prepare_inputs_for_generation
    ).parameters
    metadata = {
        "model_type": config["model_type"],
        "architectures": config.get("architectures", []),
        "hidden_size": int(config["text_config"]["hidden_size"]),
        "vision_encoder_present": "vision_config" in config,
        "forward_supports_inputs_embeds": "inputs_embeds" in forward_parameters,
        "forward_supports_labels": "labels" in forward_parameters,
        "generation_supports_inputs_embeds": (
            "inputs_embeds" in generation_parameters
        ),
    }
    if not all(
        metadata[key]
        for key in (
            "forward_supports_inputs_embeds",
            "forward_supports_labels",
            "generation_supports_inputs_embeds",
        )
    ):
        raise RuntimeError(f"Qwen3.5 soft-prefix interface check failed: {metadata}")
    return metadata
