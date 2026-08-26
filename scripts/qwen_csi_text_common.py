from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from scripts.build_retrieval_signal_description_baseline import sample_record
from scripts.evaluate import _render_signal_description


SYSTEM_PROMPT = (
    "You analyze serialized preprocessed beamspace channel state information "
    "(CSI). Return exactly one JSON object and no markdown or explanation. "
    "Do not invent unsupported values. Use null when a value cannot be inferred."
)

NUMERIC_RECORD_FIELDS = (
    "path_count",
    "first_path_delay_ns",
    "first_path_angle_deg",
    "first_path_power_dbw",
    "k_factor_db",
    "delay_spread_ns",
    "angle_spread_deg",
    "los_delay_ns",
    "los_angle_deg",
    "reflection_count",
    "reflection_path_count",
)

RECORD_FIELDS = ("environment", "los_status", *NUMERIC_RECORD_FIELDS)

OUTPUT_FIELDS = (*RECORD_FIELDS, "description")


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in RECORD_FIELDS:
        value = record.get(field)
        if field in NUMERIC_RECORD_FIELDS:
            result[field] = finite_float(value)
        elif value is None:
            result[field] = None
        else:
            result[field] = str(value).strip().lower()
    if result.get("los_status") != "los":
        result["los_delay_ns"] = None
        result["los_angle_deg"] = None
    return result


def target_response(sample) -> dict[str, Any]:
    record = sample_record(sample)
    safe_record = json_safe_record(record)
    return {
        **safe_record,
        "description": _render_signal_description(record),
    }


def select_serialized_values(
    values: torch.Tensor,
    max_values: int | None,
) -> tuple[torch.Tensor, str]:
    values = values.flatten().float().cpu()
    if max_values is None or values.numel() <= max_values:
        return values, "all"
    if max_values <= 0:
        raise ValueError("max_values must be positive when provided.")
    indices = torch.linspace(0, values.numel() - 1, max_values).round().long()
    return values[indices], "uniform"


def sample_to_prompt(
    sample,
    *,
    decimals: int = 3,
    max_csi_values: int | None = None,
    number_format: str = "fixed",
) -> tuple[str, dict[str, Any]]:
    tokens = sample.tokens[: int(sample.n_tokens)]
    selected, selection = select_serialized_values(tokens, max_csi_values)
    if number_format == "fixed":
        value_format = f".{decimals}f"
    elif number_format == "scientific":
        value_format = f".{decimals}e"
    else:
        raise ValueError(
            f"Unsupported number_format={number_format!r}; expected fixed or scientific."
        )
    csi_text = ",".join(format(float(value), value_format) for value in selected)
    source_n_freq = int(getattr(sample, "source_n_freq", 0) or tokens.shape[-1])
    bandwidth_hz = finite_float(getattr(sample, "bandwidth_hz", None))
    spacing_hz = finite_float(getattr(sample, "subcarrier_spacing_hz", None))
    metadata = {
        "representation": "serialized_preprocessed_beamspace_csi",
        "tensor_shape": list(tokens.shape),
        "original_value_count": int(tokens.numel()),
        "serialized_value_count": int(selected.numel()),
        "value_selection": selection,
        "number_format": number_format,
        "decimal_places": decimals,
    }
    prompt = (
        "Infer the physical channel facts from the serialized preprocessed "
        "beamspace CSI below.\n"
        f"array_type={getattr(sample, 'array_type', '')}\n"
        f"array_rows={int(getattr(sample, 'array_rows', 0) or 0)}\n"
        f"array_cols={int(getattr(sample, 'array_cols', 0) or 0)}\n"
        f"antenna_spacing_wavelengths={getattr(sample, 'antenna_spacing_wavelengths', None)}\n"
        f"source_num_subcarriers={source_n_freq}\n"
        f"bandwidth_hz={bandwidth_hz}\n"
        f"subcarrier_spacing_hz={spacing_hz}\n"
        f"beamspace_tensor_shape={list(tokens.shape)}\n"
        f"serialized_value_count={selected.numel()}\n"
        f"value_selection={selection}\n"
        f"number_format={number_format}\n"
        "Return one JSON object with exactly these keys:\n"
        f"{json.dumps(OUTPUT_FIELDS, ensure_ascii=True, separators=(',', ':'))}\n"
        "environment and los_status must be inferred categorical strings or null. "
        "los_status, when known, must be exactly los or nlos. All count, delay, "
        "angle, power, factor, and spread fields must be JSON numbers or null. "
        "description must be a concise string supported by the inferred fields. "
        "No example output values are supplied; do not echo these instructions.\n"
        f"csi_values=[{csi_text}]"
    )
    return prompt, metadata


def apply_chat_template(
    tokenizer,
    prompt: str,
    *,
    add_generation_prompt: bool,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def training_text(tokenizer, prompt: str, response: str) -> tuple[str, str]:
    prompt_text = apply_chat_template(
        tokenizer,
        prompt,
        add_generation_prompt=True,
    )
    eos_token = tokenizer.eos_token or ""
    return prompt_text, response + eos_token


def extract_json_object_with_span(text: str) -> tuple[dict[str, Any], int, int]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, index, index + end
    raise ValueError("No valid JSON object found in model output.")


def extract_json_object(text: str) -> dict[str, Any]:
    value, _, _ = extract_json_object_with_span(text)
    return value


def empty_prediction_record() -> dict[str, Any]:
    return {field: None for field in RECORD_FIELDS}


def parse_generated_response(text: str) -> tuple[dict[str, Any], str, str | None]:
    try:
        response, start, end = extract_json_object_with_span(text)
    except ValueError as error:
        return empty_prediction_record(), str(text).strip(), str(error)

    nested_record = response.get("record")
    raw_record = nested_record if isinstance(nested_record, dict) else response
    record = json_safe_record(raw_record)
    description = response.get("description")
    if not isinstance(description, str) or not description.strip():
        description = str(text).strip()

    errors = []
    leading_text = text[:start].strip()
    trailing_text = text[end:].strip()
    if leading_text:
        errors.append("text_before_json")
    if trailing_text:
        errors.append("text_after_json")
    if nested_record is not None:
        errors.append("nested_record_schema")
    missing_fields = [field for field in OUTPUT_FIELDS if field not in response]
    extra_fields = [field for field in response if field not in OUTPUT_FIELDS]
    if missing_fields:
        errors.append("missing_fields=" + ",".join(missing_fields))
    if extra_fields:
        errors.append("extra_fields=" + ",".join(extra_fields))
    parse_error = "; ".join(errors) if errors else None
    return record, description.strip(), parse_error


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
