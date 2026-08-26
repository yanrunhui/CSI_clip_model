from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset  # noqa: E402
from scripts.qwen_csi_text_common import (  # noqa: E402
    SYSTEM_PROMPT,
    sample_to_prompt,
    target_response,
    training_text,
)


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export preprocessed beamspace CSI and canonical target descriptions "
            "as JSONL for Qwen zero-shot or LoRA experiments."
        )
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument(
        "--number-format",
        choices=("fixed", "scientific"),
        default="fixed",
        help="Numeric representation used for serialized CSI values.",
    )
    parser.add_argument("--max-csi-values", type=int)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--stats-samples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=32768)
    args = parser.parse_args()

    if args.decimals < 0:
        raise ValueError("--decimals must be non-negative.")
    if args.stats_samples < 0:
        raise ValueError("--stats-samples must be non-negative.")

    samples = PreprocessedCSIDataset.from_pt(args.data_path).samples
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError(f"No samples loaded from {args.data_path}.")

    tokenizer = None
    if args.tokenizer_path:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Token statistics require transformers. Install the Qwen baseline dependencies."
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            trust_remote_code=False,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_lengths: list[int] = []
    training_lengths: list[int] = []
    original_value_counts: list[int] = []
    serialized_value_counts: list[int] = []

    with output_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            prompt, serialization = sample_to_prompt(
                sample,
                decimals=args.decimals,
                max_csi_values=args.max_csi_values,
                number_format=args.number_format,
            )
            response = target_response(sample)
            response_text = json.dumps(
                response,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            row = {
                "index": index,
                "group_id": str(getattr(sample, "group_id", "")),
                "config_key": str(getattr(sample, "config_key", "")),
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "target_response": response,
                "target_text": response["description"],
                "serialization": serialization,
            }
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")

            original_value_counts.append(serialization["original_value_count"])
            serialized_value_counts.append(serialization["serialized_value_count"])
            if tokenizer is not None and index < args.stats_samples:
                prompt_text, answer_text = training_text(tokenizer, prompt, response_text)
                prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
                prompt_lengths.append(len(prompt_ids))
                training_lengths.append(len(prompt_ids) + len(answer_ids))

    stats = {
        "data_path": args.data_path,
        "output": str(output_path),
        "sample_count": len(samples),
        "decimals": args.decimals,
        "number_format": args.number_format,
        "max_csi_values": args.max_csi_values,
        "original_value_count_min": min(original_value_counts),
        "original_value_count_max": max(original_value_counts),
        "serialized_value_count_min": min(serialized_value_counts),
        "serialized_value_count_max": max(serialized_value_counts),
        "tokenizer_path": args.tokenizer_path,
        "token_stats_count": len(prompt_lengths),
        "max_length": args.max_length,
    }
    if prompt_lengths:
        stats.update(
            {
                "prompt_tokens_min": min(prompt_lengths),
                "prompt_tokens_p50": percentile(prompt_lengths, 0.50),
                "prompt_tokens_p90": percentile(prompt_lengths, 0.90),
                "prompt_tokens_p95": percentile(prompt_lengths, 0.95),
                "prompt_tokens_p99": percentile(prompt_lengths, 0.99),
                "prompt_tokens_max": max(prompt_lengths),
                "training_tokens_p50": percentile(training_lengths, 0.50),
                "training_tokens_p95": percentile(training_lengths, 0.95),
                "training_tokens_max": max(training_lengths),
                "prompt_over_max_length_rate": sum(
                    length > args.max_length for length in prompt_lengths
                )
                / len(prompt_lengths),
                "training_over_max_length_rate": sum(
                    length > args.max_length for length in training_lengths
                )
                / len(training_lengths),
            }
        )

    stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("qwen_csi_text_data_stats=" + json.dumps(stats, sort_keys=True))
    print(f"saved_qwen_csi_text_jsonl={output_path}")
    print(f"saved_qwen_csi_text_stats={stats_path}")


if __name__ == "__main__":
    main()
