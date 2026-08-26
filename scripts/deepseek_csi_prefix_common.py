from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_TYPES = {"qwen3"}


def validate_deepseek_model_path(model_path: str) -> dict[str, Any]:
    """Validate the local DeepSeek-R1-Qwen3 checkpoint before GPU loading."""
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"DeepSeek model directory not found: {path}")

    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"DeepSeek model directory {path} is missing: {', '.join(missing)}"
        )

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", ""))
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "The CSI-prefix DeepSeek baseline currently expects a Qwen3-based "
            f"checkpoint, but {path} declares model_type={model_type!r}."
        )

    index_path = path / "model.safetensors.index.json"
    single_weights = path / "model.safetensors"
    if not index_path.is_file() and not single_weights.is_file():
        raise FileNotFoundError(
            f"No safetensors weights or shard index found in {path}."
        )

    metadata = {
        "model_path": str(path),
        "model_type": model_type,
        "architectures": config.get("architectures", []),
        "hidden_size": int(config.get("hidden_size", 0)),
        "num_hidden_layers": int(config.get("num_hidden_layers", 0)),
        "torch_dtype": config.get("torch_dtype"),
    }
    print(
        "deepseek_model_validation="
        + json.dumps(metadata, ensure_ascii=True, sort_keys=True),
        flush=True,
    )
    return metadata

