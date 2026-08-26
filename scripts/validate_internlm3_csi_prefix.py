from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.internlm3_csi_prefix_common import validate_internlm3_model_path


def _causal_lm_forward_arguments(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "InternLM3ForCausalLM":
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "forward":
                    return {argument.arg for argument in child.args.args + child.args.kwonlyargs}
    raise RuntimeError("InternLM3ForCausalLM.forward was not found in local model code.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statically validate a local InternLM3 CSI-prefix baseline."
    )
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    metadata = validate_internlm3_model_path(args.model_path)
    arguments = _causal_lm_forward_arguments(
        Path(args.model_path) / "modeling_internlm3.py"
    )
    metadata.update(
        {
            "forward_supports_inputs_embeds": "inputs_embeds" in arguments,
            "forward_supports_labels": "labels" in arguments,
        }
    )
    if not metadata["forward_supports_inputs_embeds"] or not metadata["forward_supports_labels"]:
        raise RuntimeError(f"InternLM3 soft-prefix static validation failed: {metadata}")
    print(
        "internlm3_static_interface=" + json.dumps(metadata, sort_keys=True),
        flush=True,
    )
    print("internlm3_static_validation=passed", flush=True)


if __name__ == "__main__":
    main()
