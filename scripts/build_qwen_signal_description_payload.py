from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen_csi_text_common import (  # noqa: E402
    RECORD_FIELDS,
    empty_prediction_record,
    json_safe_record,
    parse_generated_response,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Qwen JSONL generations into the existing text-evaluation payload."
    )
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    data_rows = read_jsonl(args.data_jsonl)
    prediction_rows = read_jsonl(args.predictions_jsonl)
    predictions_by_index = {int(row["index"]): row for row in prediction_rows}
    if len(predictions_by_index) != len(prediction_rows):
        raise ValueError("Duplicate prediction indices found.")

    predicted_records = []
    target_records = []
    predicted_texts = []
    target_texts = []
    comparisons = []
    parse_success_count = 0
    missing_count = 0
    truncated_count = 0

    for row in data_rows:
        index = int(row["index"])
        prediction = predictions_by_index.get(index)
        if prediction is None:
            if not args.allow_missing:
                raise ValueError(f"Missing Qwen prediction for index {index}.")
            generated_text = ""
            predicted_record = empty_prediction_record()
            predicted_text = ""
            parse_error = "missing_prediction"
            input_was_truncated = False
            missing_count += 1
        else:
            generated_text = str(prediction.get("generated_text", ""))
            predicted_record, predicted_text, parse_error = parse_generated_response(
                generated_text
            )
            input_was_truncated = bool(prediction.get("input_was_truncated", False))
            parse_success_count += int(parse_error is None)
            truncated_count += int(input_was_truncated)

        target_response = row.get("target_response")
        if not isinstance(target_response, dict):
            raise ValueError(f"Missing target_response for index {index}.")
        target_record = json_safe_record(target_response)
        target_text = str(
            target_response.get("description", row.get("target_text", ""))
        )

        predicted_records.append(predicted_record)
        target_records.append(target_record)
        predicted_texts.append(predicted_text)
        target_texts.append(target_text)
        comparisons.append(
            {
                "index": index,
                "group_id": str(row.get("group_id", "")),
                "config_key": str(row.get("config_key", "")),
                "predicted_signal_description": predicted_text,
                "target_signal_description": target_text,
                "predicted_record": predicted_record,
                "target_record": target_record,
                "qwen_generated_text": generated_text,
                "qwen_parse_error": parse_error,
                "qwen_input_was_truncated": input_was_truncated,
            }
        )

    count = len(data_rows)
    metadata = {
        "baseline": "qwen_serialized_preprocessed_beamspace_csi_to_text",
        "data_jsonl": args.data_jsonl,
        "predictions_jsonl": args.predictions_jsonl,
        "sample_count": count,
        "prediction_count": len(prediction_rows),
        "missing_prediction_count": missing_count,
        "json_parse_success_count": parse_success_count,
        "json_parse_success_rate": parse_success_count / count if count else None,
        "input_truncation_count": truncated_count,
        "input_truncation_rate": truncated_count / count if count else None,
        "record_fields": list(RECORD_FIELDS),
    }
    payload = {
        "predicted_signal_descriptions": predicted_texts,
        "target_signal_descriptions": target_texts,
        "predicted_signal_records": predicted_records,
        "target_signal_records": target_records,
        "comparisons": comparisons,
        "metadata": metadata,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("qwen_payload_metadata=" + json.dumps(metadata, sort_keys=True))
    print(f"saved_qwen_signal_description_payload={output_path}")
    print(f"saved_qwen_signal_description_metadata={metadata_path}")


if __name__ == "__main__":
    main()
