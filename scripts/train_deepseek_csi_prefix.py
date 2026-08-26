from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.deepseek_csi_prefix_common import validate_deepseek_model_path  # noqa: E402
from scripts.train_qwen_csi_prefix import main as train_csi_prefix  # noqa: E402


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--model-path", required=True)
    args, _ = bootstrap.parse_known_args()
    validate_deepseek_model_path(args.model_path)
    train_csi_prefix()


if __name__ == "__main__":
    main()

