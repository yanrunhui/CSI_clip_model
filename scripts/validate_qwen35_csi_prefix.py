from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.csi_prefix_language_model import qwen35_interface_metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate local Qwen3.5 files and CSI soft-prefix interfaces."
    )
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen3.5 model directory not found: {model_path}")
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    has_weights = (model_path / "model.safetensors").is_file() or (
        model_path / "model.safetensors.index.json"
    ).is_file()
    if not has_weights:
        missing.append("model.safetensors or model.safetensors.index.json")
    if missing:
        nested_configs = sorted(model_path.glob("*/config.json"))
        nested_hint = (
            f" Nested model config found at: {nested_configs[0].parent}"
            if nested_configs
            else ""
        )
        raise FileNotFoundError(
            f"Qwen3.5 model directory {model_path} is missing: "
            f"{', '.join(missing)}.{nested_hint}"
        )
    metadata = qwen35_interface_metadata(model_path)
    metadata["model_path"] = str(model_path)
    print("qwen35_csi_prefix_interface=" + json.dumps(metadata, sort_keys=True))
    print("qwen35_static_validation=passed")


if __name__ == "__main__":
    main()
