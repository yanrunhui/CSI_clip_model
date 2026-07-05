from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.caption import CaptionGenerator
from data.dataset import (
    PHYSICS_TARGET_NAMES,
    PHYSICS_TARGET_OFFSETS,
    PHYSICS_TARGET_SCALES,
    PreprocessedCSIDataset,
    apply_semantic_key_mode,
    collate_fn,
    semantic_key_mode_choices,
)
from data.semantic_key import (
    SemanticKey,
    default_attribute_fields,
    implied_attribute_value_filters,
    semantic_key_attribute_raw_value,
    semantic_key_attribute_value,
    semantic_key_field_choices,
)
from data.tokenizer import CaptionTokenizer
from models.encoder import CSIEncoder
from models.model import (
    CSIClip,
    CSI_DELAY_CONTEXT_DIM,
    DELAY_SPREAD_BIN_LABELS,
    DELAY_SPREAD_POSITION_BINS,
    FIRST_PATH_DELAY_BIN_LABELS,
    FIRST_PATH_DELAY_POSITION_BINS,
    REFLECTION_COUNT_BIN_LABELS,
)
from models.text_encoder import PhysicsTextEncoder
from scripts.pretrain import assert_checkpoint_prototype_compatibility, deserialize_prototype_keys
from training.losses import cosine_alignment_loss, paired_contrastive_loss

PHYSICAL_DESCRIPTION_FIELDS = (
    "delay_spread_ns",
    "k_factor_db",
    "azimuth_spread_deg",
)

PHYSICAL_DESCRIPTION_TOLERANCES = {
    "delay_spread_ns": 20.0,
    "k_factor_db": 3.0,
    "azimuth_spread_deg": 10.0,
}

STRONG_K_FACTOR_DIAGNOSTIC_BINS = (
    ("low", 3.0, 15.0),
    ("mid", 15.0, 30.0),
    ("high", 30.0, 45.0),
    ("very_high", 45.0, 70.0),
)

DELAY_SPREAD_DIAGNOSTIC_BINS = (
    ("0_25", 0.0, 25.0),
    ("25_50", 25.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("100_200", 100.0, 200.0),
    ("200_400", 200.0, 400.0),
    ("400_plus", 400.0, float("inf")),
)

FIRST_PATH_DELAY_DIAGNOSTIC_BINS = (
    *(
        (label, lower, float("inf") if idx == len(FIRST_PATH_DELAY_POSITION_BINS) - 1 else upper)
        for idx, (label, lower, upper) in enumerate(FIRST_PATH_DELAY_POSITION_BINS)
    ),
)

REFLECTION_COUNT_DIAGNOSTIC_BINS = (
    ("0_5", 0.0, 6.0),
    ("6_7", 6.0, 8.0),
    ("8_10", 8.0, 11.0),
    ("11_13", 11.0, 14.0),
    ("14_plus", 14.0, float("inf")),
)

def build_tokenizer(samples, checkpoint: dict | None) -> CaptionTokenizer:
    tokenizer = CaptionTokenizer()
    if checkpoint is not None and "tokenizer_word2id" in checkpoint:
        tokenizer.word2id = dict(checkpoint["tokenizer_word2id"])
        tokenizer.id2word = {idx: word for word, idx in tokenizer.word2id.items()}
        tokenizer.next_id = max(tokenizer.id2word) + 1
    else:
        tokenizer.build_vocab(_build_prototype_captions(samples))
        tokenizer.build_vocab(sample.prop_caption for sample in samples)
        tokenizer.build_vocab(sample.instance_caption for sample in samples)
    return tokenizer


def semantic_key_sort_key(key: SemanticKey) -> tuple[str, ...]:
    return (
        key.env_type,
        key.los_status,
        key.path_richness,
        key.ds_bin,
        key.as_az_bin,
        key.k_factor_bin,
        key.first_delay_bin,
        key.first_power_bin,
        key.first_angle_bin,
        key.reflection_bin,
        key.diffraction_bin,
    )


def _build_prototype_captions(samples) -> list[str]:
    generator = CaptionGenerator()
    unique_keys = sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
    return [generator.generate_canonical(key) for key in unique_keys]


def build_prototype_bank(
    samples,
    tokenizer: CaptionTokenizer,
    max_caption_len: int = 48,
    prototype_keys_override: list[SemanticKey] | None = None,
) -> tuple[list[SemanticKey], torch.Tensor, torch.Tensor, dict[SemanticKey, int]]:
    generator = CaptionGenerator()
    unique_keys = (
        list(prototype_keys_override)
        if prototype_keys_override is not None
        else sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
    )
    captions = [generator.generate_canonical(key) for key in unique_keys]
    tokenized = [tokenizer.encode(caption, max_len=max_caption_len) for caption in captions]
    token_ids = torch.stack([item.ids for item in tokenized], dim=0)
    token_mask = torch.stack([item.mask for item in tokenized], dim=0)
    label_map = {key: idx for idx, key in enumerate(unique_keys)}
    return unique_keys, token_ids, token_mask, label_map


def _physics_target_index(name: str) -> int:
    return PHYSICS_TARGET_NAMES.index(name)


def _physics_raw_predictions(physics_predictions: torch.Tensor) -> torch.Tensor:
    return physics_predictions * PHYSICS_TARGET_SCALES + PHYSICS_TARGET_OFFSETS


def _format_scalar(value: float, decimals: int = 1) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _structured_physical_record(
    los_status: str,
    raw_values: torch.Tensor,
) -> dict[str, float | str]:
    return {
        "los_status": str(los_status),
        "delay_spread_ns": float(raw_values[_physics_target_index("delay_spread_ns")]),
        "k_factor_db": float(raw_values[_physics_target_index("k_factor_db")]),
        "azimuth_spread_deg": float(raw_values[_physics_target_index("azimuth_spread_deg")]),
    }


def _render_physical_description(record: dict[str, float | str]) -> str:
    los_text = "LoS" if str(record["los_status"]) == "los" else "NLoS"
    return (
        f"This channel is likely {los_text}, with a delay spread of about "
        f"{_format_scalar(float(record['delay_spread_ns']))} ns, a K-factor of about "
        f"{_format_scalar(float(record['k_factor_db']))} dB, and an azimuth spread of about "
        f"{_format_scalar(float(record['azimuth_spread_deg']))} deg."
    )


def _finite_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _format_signal_value(value, unit: str = "", decimals: int = 1) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "unknown"
    formatted = f"{finite:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{formatted} {unit}".strip()


def _format_signal_count(value) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "unknown"
    return str(max(int(round(finite)), 0))


def _angle_deg_from_sincos(sin_value, cos_value) -> float:
    sin_value = _finite_float(sin_value)
    cos_value = _finite_float(cos_value)
    if sin_value is None or cos_value is None:
        return math.nan
    if math.hypot(sin_value, cos_value) < 1e-6:
        return math.nan
    return math.degrees(math.atan2(sin_value, cos_value))


def _signal_description_record(
    semantic_key: SemanticKey,
    raw_values: torch.Tensor,
    *,
    los_delay_ns: float = math.nan,
    los_angle_sincos: torch.Tensor | None = None,
    reflection_count: float | None = None,
) -> dict[str, float | str]:
    first_angle_sin_idx = _physics_target_index("first_path_aoa_az_sin")
    first_angle_cos_idx = _physics_target_index("first_path_aoa_az_cos")
    if los_angle_sincos is None:
        los_angle_deg = math.nan
    else:
        los_angle_deg = _angle_deg_from_sincos(
            los_angle_sincos[0],
            los_angle_sincos[1],
        )
    if reflection_count is None:
        reflection_count = float(raw_values[_physics_target_index("reflection_count")])
    return {
        "environment": str(semantic_key.env_type),
        "los_status": str(semantic_key.los_status),
        "path_count": float(raw_values[_physics_target_index("n_paths")]),
        "first_path_delay_ns": float(raw_values[_physics_target_index("first_path_delay_ns")]),
        "first_path_angle_deg": _angle_deg_from_sincos(
            raw_values[first_angle_sin_idx],
            raw_values[first_angle_cos_idx],
        ),
        "first_path_power_dbw": float(raw_values[_physics_target_index("first_path_power_dbw")]),
        "k_factor_db": float(raw_values[_physics_target_index("k_factor_db")]),
        "delay_spread_ns": float(raw_values[_physics_target_index("delay_spread_ns")]),
        "angle_spread_deg": float(raw_values[_physics_target_index("azimuth_spread_deg")]),
        "los_delay_ns": float(los_delay_ns),
        "los_angle_deg": float(los_angle_deg),
        "reflection_count": float(reflection_count),
    }


def _render_signal_description(record: dict[str, float | str]) -> str:
    los_text = "LoS" if str(record["los_status"]) == "los" else "NLoS"
    return (
        f"This signal is {record['environment']}, {los_text}, with "
        f"{_format_signal_count(record['path_count'])} paths. "
        f"Its first-path delay is {_format_signal_value(record['first_path_delay_ns'], 'ns')}, "
        f"first-path angle is {_format_signal_value(record['first_path_angle_deg'], 'deg')}, "
        f"and first-path power is {_format_signal_value(record['first_path_power_dbw'], 'dBW')}. "
        f"Its K-factor is {_format_signal_value(record['k_factor_db'], 'dB')}, "
        f"delay spread is {_format_signal_value(record['delay_spread_ns'], 'ns')}, "
        f"and azimuth angle spread is {_format_signal_value(record['angle_spread_deg'], 'deg')}. "
        f"The LoS delay is {_format_signal_value(record['los_delay_ns'], 'ns')}, "
        f"the LoS angle is {_format_signal_value(record['los_angle_deg'], 'deg')}, "
        f"and the reflection count is {_format_signal_count(record['reflection_count'])}."
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
        else:
            moved[key] = value
    return moved


def _infer_text_mode(checkpoint: dict | None, override: str | None) -> str:
    if override is not None:
        return override
    if checkpoint is not None:
        return str(checkpoint.get("args", {}).get("text_mode", "prototype"))
    return "prototype"


def _infer_min_class_size(checkpoint: dict | None, override: int | None) -> int:
    if override is not None:
        return override
    if checkpoint is not None:
        return int(checkpoint.get("args", {}).get("min_class_size", 1))
    return 1


def _infer_semantic_key_mode(checkpoint: dict | None, override: str | None) -> str:
    if override is not None:
        return override
    if checkpoint is not None:
        return str(checkpoint.get("args", {}).get("semantic_key_mode", "full"))
    return "full"


def _infer_token_norm_mode(checkpoint: dict | None, override: str | None) -> str:
    if override is not None:
        return override
    if checkpoint is not None:
        return str(checkpoint.get("args", {}).get("token_norm_mode", "std"))
    return "std"


def _infer_use_power_branch(checkpoint: dict | None, override: bool | None) -> bool:
    if override is not None:
        return override
    if checkpoint is not None:
        return bool(checkpoint.get("args", {}).get("use_power_branch", False))
    return False


def _infer_first_path_power_gate_mode(checkpoint: dict | None, override: str | None) -> str:
    if override is not None:
        return override
    if checkpoint is not None:
        return str(checkpoint.get("args", {}).get("first_path_power_gate_mode", "none"))
    return "none"


def _infer_first_path_power_mode(checkpoint: dict | None) -> str:
    if checkpoint is not None:
        return str(checkpoint.get("args", {}).get("first_path_power_mode", "residual"))
    return "residual"


def _infer_first_path_power_use_internal_gate(checkpoint: dict | None) -> bool:
    if checkpoint is not None:
        return bool(
            checkpoint.get("args", {}).get("first_path_power_use_internal_gate", True)
        )
    return True


def _infer_use_delay_spread_head(checkpoint: dict | None) -> bool:
    if checkpoint is not None:
        csi_delay_input_weight = checkpoint.get("model_state", {}).get(
            "csi_delay_spread_head.1.weight"
        )
        context_delay_weight = checkpoint.get("model_state", {}).get(
            "delay_spread_context_head.3.weight"
        )
        context_delay_input_weight = checkpoint.get("model_state", {}).get(
            "delay_spread_context_head.1.weight"
        )
        has_compatible_csi_delay_head = (
            isinstance(csi_delay_input_weight, torch.Tensor)
            and csi_delay_input_weight.ndim == 2
            and csi_delay_input_weight.shape[1] == 256
        )
        has_compatible_context_delay_head = (
            isinstance(context_delay_weight, torch.Tensor)
            and context_delay_weight.ndim == 2
            and context_delay_weight.shape[0] == 1
            and isinstance(context_delay_input_weight, torch.Tensor)
            and context_delay_input_weight.ndim == 2
            and context_delay_input_weight.shape[1] == 256 + CSI_DELAY_CONTEXT_DIM
        )
        return (
            (
                float(checkpoint.get("args", {}).get("delay_spread_weight", 0.0)) > 0.0
                or float(checkpoint.get("args", {}).get("delay_spread_raw_weight", 0.0)) > 0.0
            )
            and has_compatible_csi_delay_head
            and has_compatible_context_delay_head
        )
    return False


def _infer_use_delay_spread_bin_head(checkpoint: dict | None) -> bool:
    if checkpoint is not None:
        classifier_weight = checkpoint.get("model_state", {}).get(
            "delay_spread_bin_classifier.3.weight"
        )
        classifier_input_weight = checkpoint.get("model_state", {}).get(
            "delay_spread_bin_classifier.1.weight"
        )
        position_weight = checkpoint.get("model_state", {}).get(
            "delay_spread_bin_position_head.3.weight"
        )
        position_input_weight = checkpoint.get("model_state", {}).get(
            "delay_spread_bin_position_head.1.weight"
        )
        has_compatible_bin_head = (
            isinstance(classifier_weight, torch.Tensor)
            and classifier_weight.ndim == 2
            and classifier_weight.shape[0] == len(DELAY_SPREAD_BIN_LABELS)
            and isinstance(classifier_input_weight, torch.Tensor)
            and classifier_input_weight.ndim == 2
            and classifier_input_weight.shape[1] == 256 + CSI_DELAY_CONTEXT_DIM
            and isinstance(position_weight, torch.Tensor)
            and position_weight.ndim == 2
            and position_weight.shape[0] == 1
            and isinstance(position_input_weight, torch.Tensor)
            and position_input_weight.ndim == 2
            and position_input_weight.shape[1] == 256 + CSI_DELAY_CONTEXT_DIM
        )
        args = checkpoint.get("args", {})
        return (
            (
                float(args.get("delay_spread_bin_classifier_weight", 0.0)) > 0.0
                or float(args.get("delay_spread_bin_position_weight", 0.0)) > 0.0
            )
            and has_compatible_bin_head
        )
    return False


def _infer_use_first_path_delay_bin_head(checkpoint: dict | None) -> bool:
    if checkpoint is not None:
        classifier_weight = checkpoint.get("model_state", {}).get(
            "first_path_delay_bin_classifier.3.weight"
        )
        classifier_input_weight = checkpoint.get("model_state", {}).get(
            "first_path_delay_bin_classifier.1.weight"
        )
        position_weight = checkpoint.get("model_state", {}).get(
            "first_path_delay_bin_position_head.3.weight"
        )
        position_input_weight = checkpoint.get("model_state", {}).get(
            "first_path_delay_bin_position_head.1.weight"
        )
        has_compatible_bin_head = (
            isinstance(classifier_weight, torch.Tensor)
            and classifier_weight.ndim == 2
            and classifier_weight.shape[0] == len(FIRST_PATH_DELAY_BIN_LABELS)
            and isinstance(classifier_input_weight, torch.Tensor)
            and classifier_input_weight.ndim == 2
            and classifier_input_weight.shape[1] == 256 + CSI_DELAY_CONTEXT_DIM
            and isinstance(position_weight, torch.Tensor)
            and position_weight.ndim == 2
            and position_weight.shape[0] == 1
            and isinstance(position_input_weight, torch.Tensor)
            and position_input_weight.ndim == 2
            and position_input_weight.shape[1] == 256 + CSI_DELAY_CONTEXT_DIM
        )
        args = checkpoint.get("args", {})
        return (
            (
                float(args.get("first_path_delay_bin_classifier_weight", 0.0)) > 0.0
                or float(args.get("first_path_delay_bin_position_weight", 0.0)) > 0.0
            )
            and has_compatible_bin_head
        )
    return False


def _assert_checkpoint_first_path_delay_bin_labels(checkpoint: dict | None) -> None:
    if checkpoint is None:
        return
    checkpoint_labels = checkpoint.get("args", {}).get("first_path_delay_bin_label_order")
    if checkpoint_labels is None:
        return
    current_labels = tuple(FIRST_PATH_DELAY_BIN_LABELS)
    if tuple(checkpoint_labels) != current_labels:
        raise ValueError(
            "Checkpoint first-path-delay bin label order does not match current code: "
            f"checkpoint={tuple(checkpoint_labels)} current={current_labels}. "
            "Retrain the first-path-delay bin heads after changing bin boundaries."
        )


def _infer_use_delay_specific_encoder(checkpoint: dict | None) -> bool:
    if checkpoint is None:
        return False
    args = checkpoint.get("args", {})
    if "use_delay_specific_encoder" in args:
        return bool(args.get("use_delay_specific_encoder", False))
    model_state = checkpoint.get("model_state", {})
    return any(
        key.startswith("csi_delay_context_encoder.initial_conv.")
        or key.startswith("csi_delay_context_encoder.multi_scale_convs.")
        for key in model_state
    )


def _infer_use_los_angle_context_encoder(checkpoint: dict | None) -> bool:
    if checkpoint is None:
        return False
    args = checkpoint.get("args", {})
    if "use_los_angle_context_encoder" in args:
        return bool(args.get("use_los_angle_context_encoder", False))
    model_state = checkpoint.get("model_state", {})
    return any(key.startswith("los_angle_context_encoder.") for key in model_state)


def _infer_use_first_path_angle_context_encoder(checkpoint: dict | None) -> bool:
    if checkpoint is None:
        return False
    args = checkpoint.get("args", {})
    if "use_first_path_angle_context_encoder" in args:
        return bool(args.get("use_first_path_angle_context_encoder", False))
    model_state = checkpoint.get("model_state", {})
    return any(key.startswith("first_path_angle_context_encoder.") for key in model_state)


def _checkpoint_has_delay_family_heads(checkpoint: dict | None) -> bool:
    if checkpoint is None:
        return False
    model_state = checkpoint.get("model_state", {})
    return any(
        key.startswith("first_path_delay_context_head.")
        or key.startswith("los_delay_context_head.")
        for key in model_state
    )


def _infer_limit_samples(checkpoint: dict | None, override: int | None) -> int | None:
    if override is not None:
        return override
    if checkpoint is not None:
        limit_samples = checkpoint.get("args", {}).get("limit_samples")
        if limit_samples is not None:
            return int(limit_samples)
    return None


def _infer_limit_samples_by_attribute(checkpoint: dict | None, override: str | None) -> str | None:
    if override is not None:
        return override
    if checkpoint is not None:
        return checkpoint.get("args", {}).get("limit_samples_by_attribute")
    return None


def _infer_limit_samples_per_attribute_value(checkpoint: dict | None, override: int | None) -> int | None:
    if override is not None:
        return override
    if checkpoint is not None:
        value = checkpoint.get("args", {}).get("limit_samples_per_attribute_value")
        if value is not None:
            return int(value)
    return None


def _infer_max_delay_spread_ns(checkpoint: dict | None, override: float | None) -> float | None:
    if override is not None:
        return override
    if checkpoint is not None:
        value = checkpoint.get("args", {}).get("max_delay_spread_ns")
        if value is not None:
            return float(value)
    return None


def _infer_reflection_count_classifier_weight(checkpoint: dict | None) -> float:
    if checkpoint is None:
        return 0.0
    args = checkpoint.get("args", {})
    return float(
        args.get(
            "reflection_count_classifier_weight",
            args.get("interaction_count_classifier_weight", 0.0),
        )
    )


def _infer_reflection_count_regression_weight(checkpoint: dict | None) -> float:
    if checkpoint is None:
        return 0.0
    args = checkpoint.get("args", {})
    return float(
        args.get(
            "reflection_count_regression_weight",
            args.get("interaction_count_regression_weight", 0.0),
        )
    )


def _infer_reflection_count_nlos_weight(checkpoint: dict | None) -> float:
    if checkpoint is None:
        return 1.0
    return float(checkpoint.get("args", {}).get("reflection_count_nlos_weight", 1.0))


def _infer_interaction_count_soft_labels(checkpoint: dict | None) -> bool:
    if checkpoint is None:
        return False
    return bool(checkpoint.get("args", {}).get("interaction_count_soft_labels", False))


def _infer_delay_spread_raw_beta_ns(checkpoint: dict | None) -> float:
    if checkpoint is not None:
        return float(checkpoint.get("args", {}).get("delay_spread_raw_beta_ns", 20.0))
    return 20.0


def _sample_delay_spread_ns(sample) -> float:
    scale = 1.0
    if hasattr(sample, "delay_spread_ns"):
        value = getattr(sample, "delay_spread_ns")
    elif hasattr(sample, "delay_spread_s"):
        value = getattr(sample, "delay_spread_s")
        scale = 1e9
    elif hasattr(sample, "delay_spread"):
        value = getattr(sample, "delay_spread")
        scale = 1e9
    else:
        return math.nan
    try:
        value = float(value) * scale
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def filter_samples_by_max_delay_spread(samples, max_delay_spread_ns: float | None):
    if max_delay_spread_ns is None:
        return samples
    if max_delay_spread_ns <= 0.0:
        raise ValueError("max_delay_spread_ns must be positive.")
    filtered = []
    for sample in samples:
        delay_spread_ns = _sample_delay_spread_ns(sample)
        if math.isfinite(delay_spread_ns) and delay_spread_ns < max_delay_spread_ns:
            filtered.append(sample)
    if not filtered:
        raise ValueError(
            "No samples remain after applying "
            f"max_delay_spread_ns={max_delay_spread_ns}."
        )
    return filtered


def parse_attribute_value_filters(value) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    entries = [value] if isinstance(value, str) else value
    filters: dict[str, tuple[str, ...]] = {}
    if isinstance(entries, dict):
        entries = [f"{field}={','.join(values) if isinstance(values, (list, tuple)) else values}" for field, values in entries.items()]
    for entry in entries:
        if "=" not in str(entry):
            raise ValueError(
                "--filter-attribute-values entries must use FIELD=VALUE[,VALUE...], "
                f"got {entry!r}."
            )
        field, raw_values = str(entry).split("=", 1)
        field = field.strip()
        if field not in semantic_key_field_choices():
            raise ValueError(
                f"Unknown filter attribute field: {field!r}. "
                f"Choose from: {', '.join(semantic_key_field_choices())}"
            )
        values = tuple(part.strip() for part in raw_values.split(",") if part.strip())
        if not values:
            raise ValueError(f"No values provided for filter attribute field {field!r}.")
        filters[field] = values
    return filters


def _infer_filter_attribute_values(
    checkpoint: dict | None,
    override: dict[str, tuple[str, ...]] | None,
) -> dict[str, tuple[str, ...]]:
    if override is not None:
        return override
    if checkpoint is not None:
        return parse_attribute_value_filters(
            checkpoint.get("args", {}).get("filter_attribute_values")
        )
    return {}


def format_attribute_value_filters(filters: dict[str, tuple[str, ...]]) -> str:
    if not filters:
        return "none"
    return ";".join(
        f"{field}={','.join(values)}"
        for field, values in sorted(filters.items())
    )


def parse_attribute_remap(value) -> dict[str, dict[str, tuple[str, ...]]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("attribute_remap must be a mapping of FIELD -> LABEL -> source labels.")
    remap: dict[str, dict[str, tuple[str, ...]]] = {}
    for field, label_map in value.items():
        if field not in semantic_key_field_choices():
            raise ValueError(
                f"Unknown attribute_remap field: {field!r}. "
                f"Choose from: {', '.join(semantic_key_field_choices())}"
            )
        if not isinstance(label_map, dict):
            raise ValueError(f"attribute_remap.{field} must map output labels to source labels.")
        remap[str(field)] = {}
        for mapped_value, source_values in label_map.items():
            if isinstance(source_values, str):
                values = (source_values,)
            else:
                values = tuple(str(source_value) for source_value in source_values)
            if not values:
                raise ValueError(f"attribute_remap.{field}.{mapped_value} must not be empty.")
            remap[str(field)][str(mapped_value)] = values
    return remap


def _infer_attribute_remap(checkpoint: dict | None) -> dict[str, dict[str, tuple[str, ...]]]:
    if checkpoint is None:
        return {}
    return parse_attribute_remap(checkpoint.get("args", {}).get("attribute_remap"))


def format_attribute_remap(remap: dict[str, dict[str, tuple[str, ...]]]) -> str:
    if not remap:
        return "none"
    return ";".join(
        f"{field}="
        + ",".join(f"{mapped}:{'|'.join(values)}" for mapped, values in mapping.items())
        for field, mapping in sorted(remap.items())
    )


def _infer_attribute_fields(checkpoint: dict | None, override: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if override is not None:
        return override
    if checkpoint is not None:
        fields = checkpoint.get("args", {}).get("attribute_classifier_fields")
        if fields is not None:
            return tuple(str(field) for field in fields)
    return default_attribute_fields()


def parse_attribute_binary_thresholds(values: list[str] | None) -> dict[str, float]:
    thresholds = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                "--attribute-binary-threshold entries must use FIELD=THRESHOLD, "
                f"got {value!r}."
            )
        field, threshold = value.split("=", 1)
        thresholds[field.strip()] = float(threshold)
    return thresholds


def build_attribute_label_maps(
    samples,
    fields: tuple[str, ...],
    attribute_remap: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> dict[str, dict[str, int]]:
    label_maps = {}
    for field in fields:
        values = sorted(
            {
                semantic_key_attribute_value(sample.semantic_key, field, attribute_remap)
                for sample in samples
            }
        )
        label_maps[field] = {value: idx for idx, value in enumerate(values)}
    return label_maps


def _load_model_state_compatible(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    compatible_state = {
        name: value
        for name, value in state_dict.items()
        if name in model_state and model_state[name].shape == value.shape
    }
    skipped = sorted(set(state_dict) - set(compatible_state))
    model.load_state_dict(compatible_state, strict=False)
    if skipped:
        print(f"skipped_incompatible_checkpoint_keys={','.join(skipped)}")


def filter_samples_by_min_class_size(samples, min_class_size: int):
    if min_class_size <= 1:
        return samples
    key_counts = Counter(sample.semantic_key for sample in samples)
    filtered = [sample for sample in samples if key_counts[sample.semantic_key] >= min_class_size]
    if not filtered:
        raise ValueError(
            f"No samples remain after filtering semantic classes with min_class_size={min_class_size}."
        )
    return filtered


def filter_samples_by_attribute_values(samples, filters: dict[str, tuple[str, ...]]):
    filtered = samples
    for field, values in filters.items():
        allowed_values = set(values)
        next_filtered = [
            sample
            for sample in filtered
            if semantic_key_attribute_raw_value(sample.semantic_key, field) in allowed_values
        ]
        if not next_filtered:
            raise ValueError(
                f"No samples remain after filtering {field} to values {','.join(values)}."
            )
        filtered = next_filtered
    return filtered


def limit_samples_for_debug(samples, limit_samples: int | None):
    if limit_samples is None:
        return samples
    if limit_samples <= 0:
        raise ValueError("--limit-samples must be a positive integer.")
    limited = samples[:limit_samples]
    if not limited:
        raise ValueError(f"No samples remain after applying limit_samples={limit_samples}.")
    return limited


def limit_samples_by_attribute_value_for_debug(
    samples,
    attribute_field: str | None,
    samples_per_value: int | None,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]] | None = None,
):
    if attribute_field is None and samples_per_value is None:
        return samples
    if attribute_field is None or samples_per_value is None:
        raise ValueError(
            "--limit-samples-by-attribute and --limit-samples-per-attribute-value must be used together."
        )
    if samples_per_value <= 0:
        raise ValueError("--limit-samples-per-attribute-value must be a positive integer.")
    grouped = {}
    for sample in samples:
        grouped.setdefault(
            semantic_key_attribute_value(sample.semantic_key, attribute_field, attribute_remap),
            [],
        ).append(sample)
    limited = []
    for value in sorted(grouped):
        limited.extend(grouped[value][:samples_per_value])
    if not limited:
        raise ValueError(
            "No samples remain after applying "
            f"limit_samples_by_attribute={attribute_field!r}."
        )
    return limited


def align_samples_to_checkpoint_prototypes(samples, checkpoint: dict | None):
    if checkpoint is None:
        return samples, None
    checkpoint_keys = deserialize_prototype_keys(checkpoint.get("prototype_keys"))
    if checkpoint_keys is None:
        return samples, None
    allowed_keys = set(checkpoint_keys)
    aligned = [sample for sample in samples if sample.semantic_key in allowed_keys]
    if not aligned:
        raise ValueError(
            "No evaluation samples match checkpoint prototype_keys after filtering. "
            "Use an eval split drawn from the same semantic-key space as the checkpoint."
        )
    if len(aligned) != len(samples):
        before_counts = Counter(sample.semantic_key for sample in samples)
        after_counts = Counter(sample.semantic_key for sample in aligned)
        dropped_keys = sorted(
            set(before_counts) - set(after_counts),
            key=semantic_key_sort_key,
        )
        print(
            "aligned evaluation samples to checkpoint prototypes: "
            f"samples {len(samples)} -> {len(aligned)}, "
            f"semantic_prototypes {len(before_counts)} -> {len(after_counts)}, "
            f"checkpoint_prototypes={len(checkpoint_keys)}, "
            f"dropped_extra_eval_prototypes={len(dropped_keys)}"
        )
    return aligned, checkpoint_keys


@torch.no_grad()
def evaluate(
    data_path: str,
    checkpoint_path: str | None,
    batch_size: int,
    device: torch.device,
    text_mode_override: str | None = None,
    min_class_size_override: int | None = None,
    semantic_key_mode_override: str | None = None,
    token_norm_mode_override: str | None = None,
    use_power_branch_override: bool | None = None,
    first_path_power_gate_mode_override: str | None = None,
    attribute_fields_override: tuple[str, ...] | None = None,
    filter_attribute_values_override: dict[str, tuple[str, ...]] | None = None,
    limit_samples_override: int | None = None,
    limit_samples_by_attribute_override: str | None = None,
    limit_samples_per_attribute_value_override: int | None = None,
    max_delay_spread_ns_override: float | None = None,
    attribute_binary_thresholds: dict[str, float] | None = None,
    physical_caption_examples: int = 3,
    save_signal_descriptions_path: str | None = None,
) -> None:
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) if checkpoint_path else None
    text_mode = _infer_text_mode(checkpoint, text_mode_override)
    min_class_size = _infer_min_class_size(checkpoint, min_class_size_override)
    semantic_key_mode = _infer_semantic_key_mode(checkpoint, semantic_key_mode_override)
    token_norm_mode = _infer_token_norm_mode(checkpoint, token_norm_mode_override)
    use_power_branch = _infer_use_power_branch(checkpoint, use_power_branch_override)
    first_path_power_gate_mode = _infer_first_path_power_gate_mode(
        checkpoint,
        first_path_power_gate_mode_override,
    )
    if first_path_power_gate_mode == "predicted_los":
        print(
            "warning: first_path_power_gate_mode=predicted_los is disabled; "
            "using base final first-path power.",
            file=sys.stderr,
        )
        first_path_power_gate_mode = "base"
    if first_path_power_gate_mode not in {"none", "base"}:
        raise ValueError(
            "first_path_power_gate_mode must be one of: none, base."
        )
    first_path_power_mode = _infer_first_path_power_mode(checkpoint)
    if first_path_power_mode not in {"residual", "absolute"}:
        raise ValueError(
            "first_path_power_mode must be one of: residual, absolute."
        )
    first_path_power_use_internal_gate = _infer_first_path_power_use_internal_gate(
        checkpoint
    )
    use_delay_spread_head = _infer_use_delay_spread_head(checkpoint)
    use_delay_spread_bin_head = _infer_use_delay_spread_bin_head(checkpoint)
    use_first_path_delay_bin_head = _infer_use_first_path_delay_bin_head(checkpoint)
    use_delay_specific_encoder = _infer_use_delay_specific_encoder(checkpoint)
    use_los_angle_context_encoder = _infer_use_los_angle_context_encoder(checkpoint)
    use_first_path_angle_context_encoder = (
        _infer_use_first_path_angle_context_encoder(checkpoint)
    )
    delay_family_heads_enabled = _checkpoint_has_delay_family_heads(checkpoint)
    attribute_fields = _infer_attribute_fields(checkpoint, attribute_fields_override)
    attribute_remap = _infer_attribute_remap(checkpoint)
    filter_attribute_values = _infer_filter_attribute_values(
        checkpoint,
        filter_attribute_values_override,
    )
    filter_attribute_values = {
        **implied_attribute_value_filters(attribute_fields, attribute_remap),
        **filter_attribute_values,
    }
    limit_samples = _infer_limit_samples(checkpoint, limit_samples_override)
    limit_samples_by_attribute = _infer_limit_samples_by_attribute(
        checkpoint,
        limit_samples_by_attribute_override,
    )
    limit_samples_per_attribute_value = _infer_limit_samples_per_attribute_value(
        checkpoint,
        limit_samples_per_attribute_value_override,
    )
    delay_spread_raw_beta_ns = _infer_delay_spread_raw_beta_ns(checkpoint)
    reflection_count_classifier_weight = _infer_reflection_count_classifier_weight(
        checkpoint
    )
    reflection_count_regression_weight = _infer_reflection_count_regression_weight(
        checkpoint
    )
    reflection_count_nlos_weight = _infer_reflection_count_nlos_weight(checkpoint)
    interaction_count_soft_labels = _infer_interaction_count_soft_labels(checkpoint)
    max_delay_spread_ns = _infer_max_delay_spread_ns(
        checkpoint,
        max_delay_spread_ns_override,
    )
    mode_samples = apply_semantic_key_mode(dataset.samples, semantic_key_mode)
    if semantic_key_mode != "full":
        before_counts = Counter(sample.semantic_key for sample in dataset.samples)
        after_counts = Counter(sample.semantic_key for sample in mode_samples)
        print(
            f"semantic_key_mode={semantic_key_mode}: "
            f"semantic_prototypes {len(before_counts)} -> {len(after_counts)}"
        )
    samples = filter_samples_by_min_class_size(mode_samples, min_class_size=min_class_size)
    if len(samples) != len(mode_samples):
        before_counts = Counter(sample.semantic_key for sample in mode_samples)
        after_counts = Counter(sample.semantic_key for sample in samples)
        print(
            f"filtered classes with min_class_size={min_class_size}: "
            f"samples {len(mode_samples)} -> {len(samples)}, "
            f"semantic_prototypes {len(before_counts)} -> {len(after_counts)}"
        )
    value_filtered_samples = filter_samples_by_attribute_values(samples, filter_attribute_values)
    if len(value_filtered_samples) != len(samples):
        before_counts = Counter(sample.semantic_key for sample in samples)
        after_counts = Counter(sample.semantic_key for sample in value_filtered_samples)
        value_count_text = ";".join(
            ",".join(
                f"{value}:{count}"
                for value, count in sorted(
                    Counter(
                        semantic_key_attribute_raw_value(sample.semantic_key, field)
                        for sample in value_filtered_samples
                    ).items()
                )
            )
            for field in sorted(filter_attribute_values)
        )
        print(
            f"filtered samples by attribute values: "
            f"filters={format_attribute_value_filters(filter_attribute_values)} "
            f"samples {len(samples)} -> {len(value_filtered_samples)}, "
            f"semantic_prototypes {len(before_counts)} -> {len(after_counts)}, "
            f"value_counts={value_count_text}"
        )
    samples = value_filtered_samples
    delay_filtered_samples = filter_samples_by_max_delay_spread(
        samples,
        max_delay_spread_ns,
    )
    if len(delay_filtered_samples) != len(samples):
        before_counts = Counter(sample.semantic_key for sample in samples)
        after_counts = Counter(sample.semantic_key for sample in delay_filtered_samples)
        print(
            f"filtered samples by max_delay_spread_ns={max_delay_spread_ns}: "
            f"samples {len(samples)} -> {len(delay_filtered_samples)}, "
            f"semantic_prototypes {len(before_counts)} -> {len(after_counts)}"
        )
    samples = delay_filtered_samples
    balanced_limited_samples = limit_samples_by_attribute_value_for_debug(
        samples,
        limit_samples_by_attribute,
        limit_samples_per_attribute_value,
        attribute_remap=attribute_remap,
    )
    if len(balanced_limited_samples) != len(samples):
        before_counts = Counter(sample.semantic_key for sample in samples)
        after_counts = Counter(sample.semantic_key for sample in balanced_limited_samples)
        value_counts = Counter(
            semantic_key_attribute_value(
                sample.semantic_key,
                limit_samples_by_attribute,
                attribute_remap,
            )
            for sample in balanced_limited_samples
        )
        value_count_text = ",".join(
            f"{value}:{count}"
            for value, count in sorted(value_counts.items())
        )
        print(
            f"limited samples by attribute for debug: "
            f"attribute={limit_samples_by_attribute} "
            f"samples {len(samples)} -> {len(balanced_limited_samples)}, "
            f"semantic_prototypes {len(before_counts)} -> {len(after_counts)}, "
            f"value_counts={value_count_text}"
        )
    samples = balanced_limited_samples
    limited_samples = limit_samples_for_debug(samples, limit_samples)
    if len(limited_samples) != len(samples):
        before_counts = Counter(sample.semantic_key for sample in samples)
        after_counts = Counter(sample.semantic_key for sample in limited_samples)
        print(
            f"limited samples for debug: "
            f"samples {len(samples)} -> {len(limited_samples)}, "
            f"semantic_prototypes {len(before_counts)} -> {len(after_counts)}"
        )
    samples = limited_samples
    samples, checkpoint_prototype_keys = align_samples_to_checkpoint_prototypes(
        samples,
        checkpoint,
    )
    tokenizer = build_tokenizer(samples, checkpoint)
    prototype_keys, prototype_token_ids, prototype_token_mask, prototype_label_map = build_prototype_bank(
        samples,
        tokenizer,
        prototype_keys_override=checkpoint_prototype_keys,
    )
    attribute_label_maps = build_attribute_label_maps(
        samples,
        attribute_fields,
        attribute_remap=attribute_remap,
    )
    loader = DataLoader(
        PreprocessedCSIDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
    model = CSIClip(
        CSIEncoder(
            d_token=8,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
        ),
        PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300)),
        num_prototypes=len(prototype_keys),
        semantic_num_classes=len(prototype_keys),
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        use_power_branch=use_power_branch,
        first_path_power_mode=first_path_power_mode,
        first_path_power_use_internal_gate=first_path_power_use_internal_gate,
        use_delay_spread_head=use_delay_spread_head,
        use_delay_specific_encoder=use_delay_specific_encoder,
        use_los_angle_context_encoder=use_los_angle_context_encoder,
        use_first_path_angle_context_encoder=use_first_path_angle_context_encoder,
        los_angle_context_token_norm_mode=token_norm_mode,
        attribute_num_classes={
            field: len(label_map)
            for field, label_map in attribute_label_maps.items()
        },
    ).to(device)
    if checkpoint is not None:
        _assert_checkpoint_first_path_delay_bin_labels(checkpoint)
        assert_checkpoint_prototype_compatibility(
            checkpoint,
            prototype_keys,
            expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
            context="evaluation checkpoint",
        )
        _load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()
    all_csi_features = []
    all_semantic_logits = []
    all_attribute_logits = {field: [] for field in attribute_label_maps}
    all_attribute_labels = {field: [] for field in attribute_label_maps}
    all_instance_text_features = []
    all_physics_predictions = []
    all_base_physics_predictions = []
    all_enhanced_first_path_power_predictions = []
    all_csi_delay_spread_predictions = []
    all_enhanced_delay_spread_predictions = []
    all_profile_delay_spread_predictions = []
    all_profile_direct_delay_spread_predictions = []
    all_delay_spread_context_predictions = []
    all_first_path_delay_context_predictions = []
    all_first_path_delay_bin_fused_raw_predictions = []
    all_first_path_delay_bin_soft_fused_raw_predictions = []
    all_los_delay_context_predictions = []
    all_los_angle_predictions = []
    all_delay_spread_bin_logits = []
    all_delay_spread_bin_positions = []
    all_first_path_delay_bin_logits = []
    all_first_path_delay_bin_positions = []
    all_strong_k_bin_logits = []
    all_strong_k_positions = []
    all_reflection_count_logits = []
    all_reflection_count_predictions = []
    all_physics_targets = []
    all_physics_raw_targets = []
    all_physics_masks = []
    all_los_delay_raw_targets = []
    all_los_delay_masks = []
    all_los_angle_targets = []
    all_los_angle_masks = []
    all_labels = []
    all_text_labels = []
    all_semantic_keys = []
    semantic_classifier_enabled = (
        checkpoint is not None
        and float(checkpoint.get("args", {}).get("semantic_classifier_weight", 0.0)) > 0
    )

    for batch in loader:
        batch = move_batch(batch, device)
        csi_features_raw = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
        )
        csi_features = F.normalize(csi_features_raw, dim=-1)
        all_csi_features.append(csi_features.cpu())
        if semantic_classifier_enabled:
            all_semantic_logits.append(model.predict_semantic(csi_features_raw).cpu())
        if attribute_label_maps:
            attribute_logits = model.predict_attributes(csi_features_raw)
            for field, label_map in attribute_label_maps.items():
                all_attribute_logits[field].append(attribute_logits[field].cpu())
                all_attribute_labels[field].extend(
                    label_map[semantic_key_attribute_value(key, field, attribute_remap)]
                    for key in batch["semantic_keys"]
                )
        power_context = None
        delay_context = None
        first_path_delay_context = None
        los_angle_context = None
        first_path_angle_context = None
        if hasattr(model, "encode_csi_delay_context"):
            delay_context = model.encode_csi_delay_context(
                batch["tokens"],
                batch["token_mask"],
                subcarrier_spacing=batch.get("subcarrier_spacing"),
            )
        if hasattr(model, "encode_first_path_delay_context"):
            first_path_delay_context = model.encode_first_path_delay_context(
                batch["tokens"],
                batch["token_mask"],
                beam_positions=batch.get("beam_positions"),
                freq_bin=batch.get("freq_bin"),
                bw_bin=batch.get("bw_bin"),
                subcarrier_spacing=batch.get("subcarrier_spacing"),
            )
        if hasattr(model, "encode_los_angle_context"):
            los_angle_context = model.encode_los_angle_context(
                batch["tokens"],
                batch["beam_positions"],
                batch["token_mask"],
                batch["freq_bin"],
                batch["bw_bin"],
                batch["subcarrier_spacing"],
            )
        if hasattr(model, "encode_first_path_angle_context"):
            first_path_angle_context = model.encode_first_path_angle_context(
                batch["tokens"],
                batch["beam_positions"],
                batch["token_mask"],
                subcarrier_spacing=batch.get("subcarrier_spacing"),
            )
        if use_power_branch:
            power_context = model.encode_power_context(
                batch["tokens"],
                batch["token_mask"],
                delay_power_map=batch.get("delay_power_map"),
                delay_power_profile=batch.get("delay_power_profile"),
            )
        physics_outputs = model.predict_physics_components(
            csi_features_raw,
            power_context=power_context,
            delay_context=delay_context,
            first_path_delay_context=first_path_delay_context,
            los_angle_context=los_angle_context,
            first_path_angle_context=first_path_angle_context,
        )
        physics_predictions = physics_outputs["final"]
        all_base_physics_predictions.append(physics_outputs["base"].cpu())
        all_enhanced_first_path_power_predictions.append(
            physics_outputs["enhanced_first_path_power"].cpu()
        )
        all_csi_delay_spread_predictions.append(
            physics_outputs["csi_delay_spread"].cpu()
        )
        all_enhanced_delay_spread_predictions.append(
            physics_outputs["enhanced_delay_spread"].cpu()
        )
        all_profile_delay_spread_predictions.append(
            physics_outputs["profile_delay_spread"].cpu()
        )
        all_profile_direct_delay_spread_predictions.append(
            physics_outputs["profile_direct_delay_spread"].cpu()
        )
        all_delay_spread_context_predictions.append(
            physics_outputs["delay_spread_context"].cpu()
        )
        all_first_path_delay_context_predictions.append(
            physics_outputs["first_path_delay_context"].cpu()
        )
        all_first_path_delay_bin_fused_raw_predictions.append(
            physics_outputs["first_path_delay_bin_fused_raw"].cpu()
        )
        all_first_path_delay_bin_soft_fused_raw_predictions.append(
            physics_outputs["first_path_delay_bin_soft_fused_raw"].cpu()
        )
        all_los_delay_context_predictions.append(
            physics_outputs["los_delay_context"].cpu()
        )
        all_los_angle_predictions.append(
            physics_outputs["los_angle_sincos"].cpu()
        )
        all_delay_spread_bin_logits.append(
            physics_outputs["delay_spread_bin_logits"].cpu()
        )
        all_delay_spread_bin_positions.append(
            physics_outputs["delay_spread_bin_position"].cpu()
        )
        all_first_path_delay_bin_logits.append(
            physics_outputs["first_path_delay_bin_logits"].cpu()
        )
        all_first_path_delay_bin_positions.append(
            physics_outputs["first_path_delay_bin_position"].cpu()
        )
        all_strong_k_bin_logits.append(physics_outputs["k_factor_strong_bin_logits"].cpu())
        all_strong_k_positions.append(physics_outputs["k_factor_strong_position"].cpu())
        all_reflection_count_logits.append(physics_outputs["reflection_count_logits"].cpu())
        all_reflection_count_predictions.append(
            physics_outputs["reflection_count_prediction"].cpu()
        )
        all_physics_predictions.append(physics_predictions.cpu())
        all_physics_targets.append(batch["physics_targets"].cpu())
        all_physics_raw_targets.append(batch["physics_raw_targets"].cpu())
        all_physics_masks.append(batch["physics_target_mask"].cpu())
        all_los_delay_raw_targets.append(batch["los_delay_raw_target"].cpu())
        all_los_delay_masks.append(batch["los_delay_target_mask"].cpu())
        all_los_angle_targets.append(batch["los_angle_target"].cpu())
        all_los_angle_masks.append(batch["los_angle_target_mask"].cpu())
        all_semantic_keys.extend(batch["semantic_keys"])
        if text_mode in ("instance", "multipositive"):
            instance_text_features = model.encode_text(
                batch["t_instance_ids"],
                batch["t_instance_mask"],
                normalize=True,
            )
            all_instance_text_features.append(instance_text_features.cpu())
            all_text_labels.extend(prototype_label_map[key] for key in batch["semantic_keys"])
        all_labels.extend(prototype_label_map[key] for key in batch["semantic_keys"])

    prototype_text_features = model.encode_text(
        prototype_token_ids.to(device),
        prototype_token_mask.to(device),
        normalize=True,
    ).cpu()
    prototype_features = model.encode_prototypes(normalize=True).cpu()
    csi_features = torch.cat(all_csi_features, dim=0)
    semantic_logits = torch.cat(all_semantic_logits, dim=0) if all_semantic_logits else None
    attribute_logits = {
        field: torch.cat(chunks, dim=0)
        for field, chunks in all_attribute_logits.items()
        if chunks
    }
    attribute_labels = {
        field: torch.tensor(values, dtype=torch.long)
        for field, values in all_attribute_labels.items()
        if values
    }
    physics_predictions = torch.cat(all_physics_predictions, dim=0)
    base_physics_predictions = torch.cat(all_base_physics_predictions, dim=0)
    enhanced_first_path_power_predictions = torch.cat(all_enhanced_first_path_power_predictions, dim=0)
    csi_delay_spread_predictions = torch.cat(all_csi_delay_spread_predictions, dim=0)
    enhanced_delay_spread_predictions = torch.cat(all_enhanced_delay_spread_predictions, dim=0)
    if not use_delay_spread_head:
        csi_delay_spread_predictions = torch.full_like(
            enhanced_delay_spread_predictions,
            float("nan"),
        )
    profile_delay_spread_predictions = torch.cat(all_profile_delay_spread_predictions, dim=0)
    profile_direct_delay_spread_predictions = torch.cat(
        all_profile_direct_delay_spread_predictions,
        dim=0,
    )
    delay_spread_context_predictions = torch.cat(
        all_delay_spread_context_predictions,
        dim=0,
    )
    first_path_delay_context_predictions = torch.cat(
        all_first_path_delay_context_predictions,
        dim=0,
    )
    first_path_delay_bin_fused_raw_predictions = torch.cat(
        all_first_path_delay_bin_fused_raw_predictions,
        dim=0,
    )
    first_path_delay_bin_soft_fused_raw_predictions = torch.cat(
        all_first_path_delay_bin_soft_fused_raw_predictions,
        dim=0,
    )
    los_delay_context_predictions = torch.cat(
        all_los_delay_context_predictions,
        dim=0,
    )
    los_angle_predictions = torch.cat(all_los_angle_predictions, dim=0)
    delay_spread_bin_logits = torch.cat(all_delay_spread_bin_logits, dim=0)
    delay_spread_bin_positions = torch.cat(all_delay_spread_bin_positions, dim=0)
    first_path_delay_bin_logits = torch.cat(all_first_path_delay_bin_logits, dim=0)
    first_path_delay_bin_positions = torch.cat(all_first_path_delay_bin_positions, dim=0)
    strong_k_bin_logits = torch.cat(all_strong_k_bin_logits, dim=0)
    strong_k_positions = torch.cat(all_strong_k_positions, dim=0)
    reflection_count_logits = torch.cat(all_reflection_count_logits, dim=0)
    reflection_count_predictions = torch.cat(all_reflection_count_predictions, dim=0)
    physics_targets = torch.cat(all_physics_targets, dim=0)
    physics_raw_targets = torch.cat(all_physics_raw_targets, dim=0)
    physics_masks = torch.cat(all_physics_masks, dim=0)
    los_delay_raw_targets = torch.cat(all_los_delay_raw_targets, dim=0)
    los_delay_masks = torch.cat(all_los_delay_masks, dim=0)
    los_angle_targets = torch.cat(all_los_angle_targets, dim=0)
    los_angle_masks = torch.cat(all_los_angle_masks, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    logit_scale = float(model.logit_scale.exp().detach().cpu().item())
    prototype_logits = logit_scale * csi_features @ prototype_features.T
    first_path_power_idx = _physics_target_index("first_path_power_dbw")
    if first_path_power_gate_mode == "base":
        physics_predictions = physics_predictions.clone()
        physics_predictions[:, first_path_power_idx] = base_physics_predictions[
            :, first_path_power_idx
        ]

    if text_mode == "prototype":
        text_features = prototype_text_features
        text_labels = labels
        logits = logit_scale * csi_features @ text_features.T
        text_metric_prefix = "csi_to_text_proto"
        text_prototype_targets = prototype_features
    elif text_mode in ("instance", "multipositive"):
        text_features = torch.cat(all_instance_text_features, dim=0)
        text_labels = torch.arange(text_features.shape[0], dtype=torch.long)
        instance_semantic_labels = torch.tensor(all_text_labels, dtype=torch.long)
        logits = logit_scale * csi_features @ text_features.T
        text_metric_prefix = "csi_to_instance_text"
        text_prototype_targets = prototype_features[labels]
    else:
        raise ValueError(f"Unsupported text_mode={text_mode!r}")

    instance_loss = F.cross_entropy(logits, text_labels)
    csi_prototype_loss = F.cross_entropy(prototype_logits, labels)
    if text_mode == "prototype":
        text_prototype_loss = paired_contrastive_loss(
            text_features,
            text_prototype_targets,
            torch.tensor(logit_scale, dtype=text_features.dtype),
        )
    else:
        text_prototype_loss = cosine_alignment_loss(
            text_features,
            text_prototype_targets,
        )
    eval_loss = instance_loss + csi_prototype_loss + text_prototype_loss
    print(f"eval_learnable_prototype_loss={float(eval_loss):.4f}")
    print(f"eval_csi_to_text_loss={float(instance_loss):.4f}")
    print(f"eval_csi_to_prototype_loss={float(csi_prototype_loss):.4f}")
    print(f"eval_text_to_prototype_loss={float(text_prototype_loss):.4f}")
    print(f"logit_scale={logit_scale:.4f}")
    print(f"text_mode={text_mode}")
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"token_norm_mode={token_norm_mode}")
    print(f"use_power_branch={use_power_branch}")
    print(f"first_path_power_gate_mode={first_path_power_gate_mode}")
    print(f"first_path_power_mode={first_path_power_mode}")
    print(f"first_path_power_use_internal_gate={first_path_power_use_internal_gate}")
    print(f"use_delay_spread_head={use_delay_spread_head}")
    print(f"use_delay_specific_encoder={use_delay_specific_encoder}")
    print(f"use_los_angle_context_encoder={use_los_angle_context_encoder}")
    print(f"use_first_path_angle_context_encoder={use_first_path_angle_context_encoder}")
    print(f"use_delay_family_heads={delay_family_heads_enabled}")
    print(f"use_delay_spread_bin_head={use_delay_spread_bin_head}")
    print(f"use_first_path_delay_bin_head={use_first_path_delay_bin_head}")
    print(f"min_class_size={min_class_size}")
    print(f"attribute_remap={format_attribute_remap(attribute_remap)}")
    print(f"filter_attribute_values={format_attribute_value_filters(filter_attribute_values)}")
    print(f"limit_samples={limit_samples}")
    print(f"limit_samples_by_attribute={limit_samples_by_attribute}")
    print(f"limit_samples_per_attribute_value={limit_samples_per_attribute_value}")
    print(f"max_delay_spread_ns={max_delay_spread_ns}")
    print(f"delay_spread_raw_beta_ns={delay_spread_raw_beta_ns:.4f}")
    print(f"reflection_count_classifier_weight={reflection_count_classifier_weight:.4f}")
    print(f"reflection_count_regression_weight={reflection_count_regression_weight:.4f}")
    print(f"reflection_count_nlos_weight={reflection_count_nlos_weight:.4f}")
    print(f"interaction_count_soft_labels={interaction_count_soft_labels}")
    print(f"semantic_prototypes={len(prototype_keys)}")
    if semantic_classifier_enabled and semantic_logits is not None:
        _print_semantic_classifier_metrics(semantic_logits, labels, prototype_keys)
    if checkpoint is not None and float(checkpoint.get("args", {}).get("attribute_classifier_weight", 0.0)) > 0:
        print(f"attribute_classifier_fields={','.join(attribute_fields)}")
        print(
            "attribute_classifier_class_weight="
            f"{checkpoint.get('args', {}).get('attribute_classifier_class_weight', 'none')}"
        )
        print(
            "attribute_classifier_logit_adjustment="
            f"{float(checkpoint.get('args', {}).get('attribute_classifier_logit_adjustment', 0.0)):.4f}"
        )
    _print_retrieval_metrics(text_metric_prefix, logits, text_labels)
    if checkpoint is not None and float(checkpoint.get("args", {}).get("attribute_classifier_weight", 0.0)) > 0:
        _print_attribute_classifier_metrics(
            attribute_logits,
            attribute_labels,
            attribute_label_maps,
            attribute_binary_thresholds or {},
        )
    _print_physics_regression_metrics(
        physics_predictions=physics_predictions,
        physics_raw_targets=physics_raw_targets,
        physics_masks=physics_masks,
    )
    _print_interaction_count_head_diagnostics(
        reflection_logits=reflection_count_logits,
        reflection_predictions=reflection_count_predictions,
        physics_raw_targets=physics_raw_targets,
        physics_masks=physics_masks,
        semantic_keys=all_semantic_keys,
    )
    _print_first_path_angle_diagnostics(
        physics_predictions=physics_predictions,
        physics_targets=physics_targets,
        physics_masks=physics_masks,
        semantic_keys=all_semantic_keys,
    )
    _print_k_factor_diagnostics(
        physics_predictions=physics_predictions,
        physics_raw_targets=physics_raw_targets,
        physics_masks=physics_masks,
        semantic_keys=all_semantic_keys,
    )
    _print_first_path_power_diagnostics(
        base_physics_predictions=base_physics_predictions,
        physics_predictions=physics_predictions,
        enhanced_first_path_power_predictions=enhanced_first_path_power_predictions,
        physics_raw_targets=physics_raw_targets,
        physics_masks=physics_masks,
        semantic_keys=all_semantic_keys,
        prototype_logits=prototype_logits,
        prototype_keys=prototype_keys,
    )
    if delay_family_heads_enabled:
        _print_delay_family_diagnostics(
            first_path_delay_predictions=first_path_delay_context_predictions,
            first_path_delay_bin_logits=(
                first_path_delay_bin_logits if use_first_path_delay_bin_head else None
            ),
            first_path_delay_bin_positions=(
                first_path_delay_bin_positions if use_first_path_delay_bin_head else None
            ),
            first_path_delay_bin_fused_raw=(
                first_path_delay_bin_fused_raw_predictions
                if use_first_path_delay_bin_head
                else None
            ),
            first_path_delay_bin_soft_fused_raw=(
                first_path_delay_bin_soft_fused_raw_predictions
                if use_first_path_delay_bin_head
                else None
            ),
            los_delay_predictions=los_delay_context_predictions,
            los_angle_predictions=los_angle_predictions,
            physics_raw_targets=physics_raw_targets,
            physics_masks=physics_masks,
            los_delay_raw_targets=los_delay_raw_targets,
            los_delay_masks=los_delay_masks,
            los_angle_targets=los_angle_targets,
            los_angle_masks=los_angle_masks,
            semantic_keys=all_semantic_keys,
        )
    _print_delay_spread_diagnostics(
        base_physics_predictions=base_physics_predictions,
        physics_predictions=physics_predictions,
        csi_delay_spread_predictions=csi_delay_spread_predictions,
        enhanced_delay_spread_predictions=enhanced_delay_spread_predictions,
        profile_delay_spread_predictions=profile_delay_spread_predictions,
        profile_direct_delay_spread_predictions=profile_direct_delay_spread_predictions,
        delay_spread_context_predictions=delay_spread_context_predictions if use_delay_spread_head else None,
        delay_spread_bin_logits=delay_spread_bin_logits if use_delay_spread_bin_head else None,
        delay_spread_bin_positions=delay_spread_bin_positions if use_delay_spread_bin_head else None,
        physics_raw_targets=physics_raw_targets,
        physics_masks=physics_masks,
        raw_beta_ns=delay_spread_raw_beta_ns,
    )
    _print_structured_physical_description_metrics(
        physics_predictions=physics_predictions,
        physics_raw_targets=physics_raw_targets,
        physics_masks=physics_masks,
        semantic_keys=all_semantic_keys,
        prototype_logits=prototype_logits,
        prototype_keys=prototype_keys,
        example_count=physical_caption_examples,
    )
    signal_description_payload = _build_signal_description_payload(
        samples=samples,
        physics_predictions=physics_predictions,
        physics_raw_targets=physics_raw_targets,
        semantic_keys=all_semantic_keys,
        prototype_logits=prototype_logits,
        prototype_keys=prototype_keys,
        semantic_logits=semantic_logits if semantic_classifier_enabled else None,
        los_delay_predictions=los_delay_context_predictions,
        los_delay_raw_targets=los_delay_raw_targets,
        los_delay_masks=los_delay_masks,
        los_angle_predictions=los_angle_predictions,
        los_angle_targets=los_angle_targets,
        los_angle_masks=los_angle_masks,
        reflection_count_predictions=reflection_count_predictions,
    )
    _print_signal_description_examples(
        signal_description_payload,
        example_count=physical_caption_examples,
    )
    if save_signal_descriptions_path is not None:
        _save_signal_description_payload(
            save_signal_descriptions_path,
            signal_description_payload,
        )
    if text_mode in ("instance", "multipositive"):
        _print_semantic_retrieval_metrics(
            "csi_to_instance_text_semantic",
            logits,
            query_labels=labels,
            item_labels=instance_semantic_labels,
        )
        _print_gated_exact_retrieval_metrics(
            "csi_to_instance_text_oracle_semantic_exact",
            logits,
            target_indices=text_labels,
            item_labels=instance_semantic_labels,
            gate_labels=labels,
        )
        _print_gated_exact_retrieval_metrics(
            "csi_to_instance_text_predicted_semantic_exact",
            logits,
            target_indices=text_labels,
            item_labels=instance_semantic_labels,
            gate_labels=prototype_logits.argmax(dim=1),
        )
        _print_physics_neighbor_retrieval_metrics(
            "csi_to_instance_text_physics_neighbor",
            logits,
            query_targets=physics_targets,
            query_masks=physics_masks,
            item_targets=physics_targets,
            item_masks=physics_masks,
            distance_threshold=float(
                checkpoint.get("args", {}).get("multipositive_distance_threshold", 0.25)
                if checkpoint is not None
                else 0.25
            ),
        )
    _print_retrieval_metrics("csi_to_learnable_prototype", prototype_logits, labels)

    _print_retrieval_metrics(
        "text_proto_to_learnable_prototype",
        logit_scale * prototype_text_features @ prototype_features.T,
        torch.arange(prototype_features.shape[0], dtype=torch.long),
    )


def _print_retrieval_metrics(prefix: str, logits: torch.Tensor, labels: torch.Tensor) -> None:
    ranking = logits.argsort(dim=1, descending=True)
    target_ranks = (ranking == labels.unsqueeze(1)).float().argmax(dim=1) + 1
    for k in (1, 5, 10):
        hits = (target_ranks <= min(k, logits.shape[1])).float().mean().item()
        print(f"{prefix}_R@{k}={hits:.4f}")
    print(f"{prefix}_MRR={float((1.0 / target_ranks.float()).mean()):.4f}")
    print(f"{prefix}_mean_rank={float(target_ranks.float().mean()):.2f}")


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) == 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def _binary_threshold_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold_override: float | None = None,
) -> dict[str, float]:
    scores = logits[:, 1] - logits[:, 0]
    if threshold_override is not None:
        thresholds = torch.tensor([threshold_override], dtype=scores.dtype)
    else:
        sorted_scores = torch.sort(scores).values
        if sorted_scores.numel() == 1:
            thresholds = sorted_scores
        else:
            midpoints = (sorted_scores[:-1] + sorted_scores[1:]) * 0.5
            thresholds = torch.cat(
                [
                    sorted_scores[:1] - 1.0,
                    midpoints,
                    sorted_scores[-1:] + 1.0,
                ]
            )

    best = {
        "macro_top1": -1.0,
        "top1": 0.0,
        "threshold": 0.0,
        "class0_acc": 0.0,
        "class1_acc": 0.0,
    }
    for threshold in thresholds:
        predictions = (scores >= threshold).long()
        class_accs = []
        for class_idx in (0, 1):
            mask = labels == class_idx
            if bool(mask.any()):
                class_accs.append((predictions[mask] == labels[mask]).float().mean())
            else:
                class_accs.append(torch.zeros(()))
        macro_top1 = float(torch.stack(class_accs).mean())
        top1 = float((predictions == labels).float().mean())
        if threshold_override is not None or macro_top1 > best["macro_top1"]:
            best = {
                "macro_top1": macro_top1,
                "top1": top1,
                "threshold": float(threshold),
                "class0_acc": float(class_accs[0]),
                "class1_acc": float(class_accs[1]),
            }
    best["margin_mean"] = float(scores.mean())
    best["margin_std"] = float(scores.std())
    return best


def _print_attribute_classifier_metrics(
    attribute_logits: dict[str, torch.Tensor],
    attribute_labels: dict[str, torch.Tensor],
    attribute_label_maps: dict[str, dict[str, int]],
    attribute_binary_thresholds: dict[str, float],
) -> None:
    for field, logits in attribute_logits.items():
        labels = attribute_labels[field]
        predictions = logits.argmax(dim=1)
        label_map = attribute_label_maps[field]
        id_to_value = {idx: value for value, idx in label_map.items()}
        num_classes = len(label_map)
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for true_label, pred_label in zip(labels.tolist(), predictions.tolist()):
            confusion[int(true_label), int(pred_label)] += 1

        class_sizes = confusion.sum(dim=1)
        class_correct = confusion.diag()
        nonempty = class_sizes > 0
        class_accuracy = torch.zeros(num_classes, dtype=torch.float32)
        class_accuracy[nonempty] = class_correct[nonempty].float() / class_sizes[nonempty].float()
        top1 = (predictions == labels).float().mean()
        macro_acc = class_accuracy[nonempty].mean() if bool(nonempty.any()) else torch.zeros(())
        majority_label = int(class_sizes.argmax().item())
        majority_acc = float(class_sizes[majority_label]) / max(int(class_sizes.sum().item()), 1)

        print(f"attribute_classifier_{field}_loss={float(F.cross_entropy(logits, labels)):.4f}")
        print(f"attribute_classifier_{field}_top1={float(top1):.4f}")
        print(f"attribute_classifier_{field}_macro_top1={float(macro_acc):.4f}")
        print(f"attribute_classifier_{field}_majority_baseline_R@1={majority_acc:.4f}")
        print(f"attribute_classifier_{field}_majority_value={id_to_value[majority_label]}")
        print(
            f"attribute_classifier_{field}_class_size_accuracy_pearson="
            f"{_safe_pearson(class_sizes[nonempty].float(), class_accuracy[nonempty]):.4f}"
        )
        if num_classes == 2:
            threshold_metrics = _binary_threshold_metrics(logits, labels)
            print(
                f"attribute_classifier_{field}_binary_margin_mean="
                f"{threshold_metrics['margin_mean']:.4f}"
            )
            print(
                f"attribute_classifier_{field}_binary_margin_std="
                f"{threshold_metrics['margin_std']:.4f}"
            )
            print(
                f"attribute_classifier_{field}_calibrated_threshold="
                f"{threshold_metrics['threshold']:.4f}"
            )
            print(
                f"attribute_classifier_{field}_calibrated_top1="
                f"{threshold_metrics['top1']:.4f}"
            )
            print(
                f"attribute_classifier_{field}_calibrated_macro_top1="
                f"{threshold_metrics['macro_top1']:.4f}"
            )
            print(
                f"attribute_classifier_{field}_calibrated_value_{id_to_value[0]}_acc="
                f"{threshold_metrics['class0_acc']:.4f}"
            )
            print(
                f"attribute_classifier_{field}_calibrated_value_{id_to_value[1]}_acc="
                f"{threshold_metrics['class1_acc']:.4f}"
            )
            print(
                f"attribute_classifier_{field}_binary_score_definition="
                f"logit_{id_to_value[1]}-logit_{id_to_value[0]}"
            )
            if field in attribute_binary_thresholds:
                fixed_threshold_metrics = _binary_threshold_metrics(
                    logits,
                    labels,
                    threshold_override=attribute_binary_thresholds[field],
                )
                print(
                    f"attribute_classifier_{field}_fixed_threshold="
                    f"{fixed_threshold_metrics['threshold']:.4f}"
                )
                print(
                    f"attribute_classifier_{field}_fixed_threshold_rule="
                    f"score>=threshold predicts {id_to_value[1]}, else {id_to_value[0]}"
                )
                print(
                    f"attribute_classifier_{field}_fixed_threshold_top1="
                    f"{fixed_threshold_metrics['top1']:.4f}"
                )
                print(
                    f"attribute_classifier_{field}_fixed_threshold_macro_top1="
                    f"{fixed_threshold_metrics['macro_top1']:.4f}"
                )
                print(
                    f"attribute_classifier_{field}_fixed_threshold_value_{id_to_value[0]}_acc="
                    f"{fixed_threshold_metrics['class0_acc']:.4f}"
                )
                print(
                    f"attribute_classifier_{field}_fixed_threshold_value_{id_to_value[1]}_acc="
                    f"{fixed_threshold_metrics['class1_acc']:.4f}"
                )
        for class_idx in range(num_classes):
            print(
                f"attribute_classifier_{field}_value_{id_to_value[class_idx]}="
                f"size:{int(class_sizes[class_idx])} "
                f"acc:{float(class_accuracy[class_idx]):.4f}"
            )

        offdiag = confusion.clone()
        offdiag.fill_diagonal_(0)
        flat_counts = offdiag.flatten()
        top_confusions = torch.argsort(flat_counts, descending=True)
        printed = 0
        for flat_idx_tensor in top_confusions:
            count = int(flat_counts[int(flat_idx_tensor)].item())
            if count <= 0 or printed >= min(5, num_classes * num_classes):
                break
            true_label = int(flat_idx_tensor.item() // num_classes)
            pred_label = int(flat_idx_tensor.item() % num_classes)
            printed += 1
            print(
                f"attribute_classifier_{field}_confusion_pair_rank_{printed}="
                f"true:{id_to_value[true_label]} pred:{id_to_value[pred_label]} count:{count}"
            )


def _print_semantic_classifier_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prototype_keys: list[SemanticKey],
) -> None:
    predictions = logits.argmax(dim=1)
    num_classes = len(prototype_keys)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true_label, pred_label in zip(labels.tolist(), predictions.tolist()):
        confusion[int(true_label), int(pred_label)] += 1

    class_sizes = confusion.sum(dim=1)
    class_correct = confusion.diag()
    nonempty = class_sizes > 0
    class_accuracy = torch.zeros(num_classes, dtype=torch.float32)
    class_accuracy[nonempty] = class_correct[nonempty].float() / class_sizes[nonempty].float()
    top1 = (predictions == labels).float().mean()
    macro_acc = class_accuracy[nonempty].mean() if bool(nonempty.any()) else torch.zeros(())
    majority_label = int(class_sizes.argmax().item())
    majority_acc = float(class_sizes[majority_label]) / max(int(class_sizes.sum().item()), 1)
    prediction_sizes = torch.bincount(predictions, minlength=num_classes)

    print(f"semantic_classifier_loss={float(F.cross_entropy(logits, labels)):.4f}")
    print(f"semantic_classifier_top1={float(top1):.4f}")
    print(f"semantic_classifier_macro_top1={float(macro_acc):.4f}")
    print(f"semantic_classifier_majority_baseline_R@1={majority_acc:.4f}")
    print(f"semantic_classifier_majority_key={prototype_keys[majority_label]}")
    print(
        f"semantic_classifier_class_size_accuracy_pearson="
        f"{_safe_pearson(class_sizes[nonempty].float(), class_accuracy[nonempty]):.4f}"
    )
    print(
        "semantic_classifier_prediction_distribution="
        + ";".join(
            f"{prototype_keys[class_idx]}:{int(prediction_sizes[class_idx])}"
            for class_idx in range(num_classes)
        )
    )
    for class_idx in range(num_classes):
        print(
            f"semantic_classifier_class_{class_idx}="
            f"key:{prototype_keys[class_idx]} "
            f"size:{int(class_sizes[class_idx])} "
            f"pred:{int(prediction_sizes[class_idx])} "
            f"acc:{float(class_accuracy[class_idx]):.4f}"
        )

    offdiag = confusion.clone()
    offdiag.fill_diagonal_(0)
    flat_counts = offdiag.flatten()
    top_confusions = torch.argsort(flat_counts, descending=True)
    printed = 0
    for flat_idx_tensor in top_confusions:
        count = int(flat_counts[int(flat_idx_tensor)].item())
        if count <= 0 or printed >= min(10, num_classes * num_classes):
            break
        true_label = int(flat_idx_tensor.item() // num_classes)
        pred_label = int(flat_idx_tensor.item() % num_classes)
        printed += 1
        print(
            f"semantic_classifier_confusion_pair_rank_{printed}="
            f"true:{prototype_keys[true_label]} pred:{prototype_keys[pred_label]} count:{count}"
        )


def _print_semantic_retrieval_metrics(
    prefix: str,
    logits: torch.Tensor,
    query_labels: torch.Tensor,
    item_labels: torch.Tensor,
) -> None:
    ranking = logits.argsort(dim=1, descending=True)
    ranked_labels = item_labels[ranking]
    matches = ranked_labels == query_labels.unsqueeze(1)
    target_ranks = matches.float().argmax(dim=1) + 1
    positive_counts = (item_labels.unsqueeze(0) == query_labels.unsqueeze(1)).sum(dim=1).float()
    for k in (1, 5, 10):
        hits = matches[:, : min(k, logits.shape[1])].any(dim=1).float().mean().item()
        print(f"{prefix}_R@{k}={hits:.4f}")
    print(f"{prefix}_MRR={float((1.0 / target_ranks.float()).mean()):.4f}")
    print(f"{prefix}_mean_rank={float(target_ranks.float().mean()):.2f}")
    print(f"{prefix}_positive_count_mean={float(positive_counts.mean()):.2f}")


def _print_gated_exact_retrieval_metrics(
    prefix: str,
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    item_labels: torch.Tensor,
    gate_labels: torch.Tensor,
) -> None:
    ranks = []
    fallback_rank = logits.shape[1] + 1
    for row_idx in range(logits.shape[0]):
        target_idx = int(target_indices[row_idx])
        candidate_mask = item_labels == gate_labels[row_idx]
        if not bool(candidate_mask[target_idx]):
            ranks.append(fallback_rank)
            continue
        target_score = logits[row_idx, target_idx]
        candidate_scores = logits[row_idx, candidate_mask]
        ranks.append(int((candidate_scores > target_score).sum().item()) + 1)

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    for k in (1, 5, 10):
        hits = (ranks_tensor <= k).float().mean().item()
        print(f"{prefix}_R@{k}={hits:.4f}")
    print(f"{prefix}_MRR={float((1.0 / ranks_tensor).mean()):.4f}")
    print(f"{prefix}_mean_rank={float(ranks_tensor.mean()):.2f}")


def _print_physics_neighbor_retrieval_metrics(
    prefix: str,
    logits: torch.Tensor,
    query_targets: torch.Tensor,
    query_masks: torch.Tensor,
    item_targets: torch.Tensor,
    item_masks: torch.Tensor,
    distance_threshold: float,
) -> None:
    ranking = logits.argsort(dim=1, descending=True)
    ranks = []
    positive_counts = []
    for row_idx in range(logits.shape[0]):
        common_mask = query_masks[row_idx].unsqueeze(0) & item_masks
        common_count = common_mask.sum(dim=1)
        diffs = (query_targets[row_idx].unsqueeze(0) - item_targets).abs()
        distances = (diffs * common_mask.to(dtype=diffs.dtype)).sum(dim=1)
        distances = distances / common_count.clamp(min=1).to(dtype=diffs.dtype)
        positives = (common_count >= 3) & (distances <= distance_threshold)
        positives[row_idx] = True
        positive_counts.append(int(positives.sum().item()))
        ranked_positive = positives[ranking[row_idx]]
        ranks.append(int(ranked_positive.float().argmax().item()) + 1)

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    positive_counts_tensor = torch.tensor(positive_counts, dtype=torch.float32)
    for k in (1, 5, 10):
        hits = (ranks_tensor <= k).float().mean().item()
        print(f"{prefix}_R@{k}={hits:.4f}")
    print(f"{prefix}_MRR={float((1.0 / ranks_tensor).mean()):.4f}")
    print(f"{prefix}_mean_rank={float(ranks_tensor.mean()):.2f}")
    print(f"{prefix}_positive_count_mean={float(positive_counts_tensor.mean()):.2f}")


def _print_physics_regression_metrics(
    physics_predictions: torch.Tensor,
    physics_raw_targets: torch.Tensor,
    physics_masks: torch.Tensor,
) -> None:
    raw_predictions = _physics_raw_predictions(physics_predictions)
    errors = (raw_predictions - physics_raw_targets).abs()
    mae_values = []
    angle_sin_idx = PHYSICS_TARGET_NAMES.index("first_path_aoa_az_sin")
    angle_cos_idx = PHYSICS_TARGET_NAMES.index("first_path_aoa_az_cos")
    for idx, name in enumerate(PHYSICS_TARGET_NAMES):
        if name in {
            "first_path_aoa_az_sin",
            "first_path_aoa_az_cos",
            "first_path_power_dbw",
        }:
            continue
        mask = physics_masks[:, idx]
        if not bool(mask.any()):
            continue
        mae = errors[:, idx][mask].mean()
        mae_values.append(mae)
        print(f"physics_regression_{name}_MAE={float(mae):.4f}")
    angle_mask = physics_masks[:, angle_sin_idx] & physics_masks[:, angle_cos_idx]
    if bool(angle_mask.any()):
        pred_angle = torch.atan2(
            raw_predictions[:, angle_sin_idx],
            raw_predictions[:, angle_cos_idx],
        )
        target_angle = torch.atan2(
            physics_raw_targets[:, angle_sin_idx],
            physics_raw_targets[:, angle_cos_idx],
        )
        delta = pred_angle - target_angle
        circular_errors = torch.rad2deg(torch.atan2(torch.sin(delta), torch.cos(delta)).abs())
        circular_mae = circular_errors[angle_mask].mean()
        mae_values.append(circular_mae)
        print(f"physics_regression_first_path_aoa_az_deg_MAE={float(circular_mae):.4f}")
    if mae_values:
        total_mae = torch.stack(mae_values).mean()
    else:
        total_mae = torch.zeros(())
    print(f"physics_regression_MAE_mean={float(total_mae):.4f}")


def _interaction_count_bin_targets(
    raw_count: torch.Tensor,
    bins: tuple[tuple[str, float, float], ...],
) -> torch.Tensor:
    targets = torch.full_like(raw_count, fill_value=-1, dtype=torch.long)
    finite = torch.isfinite(raw_count)
    rounded = raw_count.round()
    for class_idx, (_, lower, upper) in enumerate(bins):
        upper_mask = rounded <= upper if math.isinf(upper) else rounded < upper
        mask = finite & (rounded >= lower) & upper_mask
        targets = torch.where(mask, torch.full_like(targets, class_idx), targets)
    return targets


def _print_single_interaction_count_head(
    prefix: str,
    target_name: str,
    bin_labels: tuple[str, ...],
    bins: tuple[tuple[str, float, float], ...],
    logits: torch.Tensor,
    predictions: torch.Tensor,
    physics_raw_targets: torch.Tensor,
    physics_masks: torch.Tensor,
    semantic_keys: list[SemanticKey],
) -> None:
    target_idx = PHYSICS_TARGET_NAMES.index(target_name)
    raw_targets = physics_raw_targets[:, target_idx]
    target_labels = _interaction_count_bin_targets(raw_targets, bins)
    valid_mask = (target_labels >= 0) & physics_masks[:, target_idx].bool()
    print(f"{prefix}_count={int(valid_mask.sum().item())}")
    print(
        f"{prefix}_bin_label_order="
        + ",".join(bin_labels)
    )
    if not bool(valid_mask.any()):
        print(f"{prefix}_head_accuracy=nan")
        print(f"{prefix}_head_adjacent_accuracy=nan")
        print(f"{prefix}_head_far_miss_fraction=nan")
        print(f"{prefix}_head_MAE=nan")
        print(f"{prefix}_head_RMSE=nan")
        print(f"{prefix}_head_signed_mean=nan")
        print(f"{prefix}_head_pearson=nan")
        print(f"{prefix}_target_mean=nan")
        print(f"{prefix}_target_std=nan")
        print(f"{prefix}_target_min=nan")
        print(f"{prefix}_target_p50=nan")
        print(f"{prefix}_target_p90=nan")
        print(f"{prefix}_target_p99=nan")
        print(f"{prefix}_target_max=nan")
        print(f"{prefix}_mean_baseline_MAE=nan")
        print(f"{prefix}_median_baseline_MAE=nan")
        print(f"{prefix}_head_confusion=nan")
        for group in ("los", "nlos"):
            print(f"{prefix}_{group}_count=0")
            print(f"{prefix}_{group}_head_MAE=nan")
            print(f"{prefix}_{group}_head_RMSE=nan")
            print(f"{prefix}_{group}_head_signed_mean=nan")
        return

    valid_targets = target_labels[valid_mask]
    valid_raw_targets = raw_targets[valid_mask].to(dtype=predictions.dtype)
    target_quantiles = torch.quantile(
        valid_raw_targets.float(),
        torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32),
    )
    target_mean = valid_raw_targets.mean()
    target_median = valid_raw_targets.median()
    predicted_labels = logits.argmax(dim=1)
    valid_predictions = predicted_labels[valid_mask]
    confusion = torch.zeros(len(bin_labels), len(bin_labels), dtype=torch.long)
    for target_label, predicted_label in zip(
        valid_targets.tolist(),
        valid_predictions.tolist(),
    ):
        confusion[int(target_label), int(predicted_label)] += 1
    adjacent_hits = (valid_predictions - valid_targets).abs() <= 1
    far_misses = (valid_predictions - valid_targets).abs() > 1
    print(
        f"{prefix}_head_accuracy="
        f"{float((valid_predictions == valid_targets).float().mean()):.4f}"
    )
    print(
        f"{prefix}_head_adjacent_accuracy="
        f"{float(adjacent_hits.float().mean()):.4f}"
    )
    print(
        f"{prefix}_head_far_miss_fraction="
        f"{float(far_misses.float().mean()):.4f}"
    )
    scale = PHYSICS_TARGET_SCALES[target_idx].to(dtype=predictions.dtype)
    offset = PHYSICS_TARGET_OFFSETS[target_idx].to(dtype=predictions.dtype)
    raw_predictions = predictions * scale + offset
    valid_raw_predictions = raw_predictions[valid_mask]
    errors = valid_raw_predictions - valid_raw_targets
    print(f"{prefix}_head_MAE={float(errors.abs().mean()):.4f}")
    print(f"{prefix}_head_RMSE={float(torch.sqrt(errors.square().mean())):.4f}")
    print(f"{prefix}_head_signed_mean={float(errors.mean()):.4f}")
    print(
        f"{prefix}_head_pearson="
        f"{_safe_pearson(valid_raw_predictions.float(), valid_raw_targets.float()):.4f}"
    )
    print(f"{prefix}_target_mean={float(target_mean):.4f}")
    print(f"{prefix}_target_std={float(valid_raw_targets.float().std(correction=0)):.4f}")
    print(f"{prefix}_target_min={float(valid_raw_targets.min()):.4f}")
    print(f"{prefix}_target_p50={float(target_quantiles[0]):.4f}")
    print(f"{prefix}_target_p90={float(target_quantiles[1]):.4f}")
    print(f"{prefix}_target_p99={float(target_quantiles[2]):.4f}")
    print(f"{prefix}_target_max={float(valid_raw_targets.max()):.4f}")
    print(
        f"{prefix}_mean_baseline_MAE="
        f"{float((valid_raw_targets - target_mean).abs().mean()):.4f}"
    )
    print(
        f"{prefix}_median_baseline_MAE="
        f"{float((valid_raw_targets - target_median).abs().mean()):.4f}"
    )
    print(
        f"{prefix}_target_histogram="
        + ",".join(
            f"{label}:{int((valid_targets == idx).sum().item())}"
            for idx, label in enumerate(bin_labels)
        )
    )
    print(
        f"{prefix}_head_confusion="
        + ";".join(
            f"{bin_labels[row]}:"
            + ",".join(
                f"{bin_labels[col]}:{int(confusion[row, col].item())}"
                for col in range(len(bin_labels))
            )
            for row in range(len(bin_labels))
        )
    )
    print(
        f"{prefix}_prediction_histogram="
        + ",".join(
            f"{label}:{int((valid_predictions == idx).sum().item())}"
            for idx, label in enumerate(bin_labels)
        )
    )

    los_mask = torch.tensor(
        [key.los_status == "los" for key in semantic_keys],
        dtype=torch.bool,
        device=valid_mask.device,
    )

    def print_group(prefix_suffix: str, group_mask: torch.Tensor) -> None:
        mask = valid_mask & group_mask
        count = int(mask.sum().item())
        print(f"{prefix}_{prefix_suffix}_count={count}")
        if count == 0:
            print(f"{prefix}_{prefix_suffix}_head_MAE=nan")
            print(f"{prefix}_{prefix_suffix}_head_RMSE=nan")
            print(f"{prefix}_{prefix_suffix}_head_signed_mean=nan")
            return
        group_errors = raw_predictions[mask] - raw_targets[mask].to(
            dtype=predictions.dtype
        )
        print(
            f"{prefix}_{prefix_suffix}_head_MAE="
            f"{float(group_errors.abs().mean()):.4f}"
        )
        print(
            f"{prefix}_{prefix_suffix}_head_RMSE="
            f"{float(torch.sqrt(group_errors.square().mean())):.4f}"
        )
        print(
            f"{prefix}_{prefix_suffix}_head_signed_mean="
            f"{float(group_errors.mean()):.4f}"
        )

    print_group("los", los_mask)
    print_group("nlos", ~los_mask)


def _print_interaction_count_head_diagnostics(
    reflection_logits: torch.Tensor,
    reflection_predictions: torch.Tensor,
    physics_raw_targets: torch.Tensor,
    physics_masks: torch.Tensor,
    semantic_keys: list[SemanticKey],
) -> None:
    _print_single_interaction_count_head(
        "reflection_count",
        "reflection_count",
        REFLECTION_COUNT_BIN_LABELS,
        REFLECTION_COUNT_DIAGNOSTIC_BINS,
        reflection_logits,
        reflection_predictions,
        physics_raw_targets,
        physics_masks,
        semantic_keys,
    )


def _print_angle_diagnostics(
    prefix: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    count = int(mask.sum().item())
    print(f"{prefix}_count={count}")
    if count == 0:
        print(f"{prefix}_MAE=nan")
        print(f"{prefix}_accuracy@10deg=nan")
        print(f"{prefix}_accuracy@30deg=nan")
        print(f"{prefix}_signed_mean=nan")
        return
    valid_predictions = F.normalize(predictions[mask], dim=-1, eps=1e-6)
    valid_targets = F.normalize(targets[mask], dim=-1, eps=1e-6)
    predicted_angle = torch.atan2(valid_predictions[:, 0], valid_predictions[:, 1])
    target_angle = torch.atan2(valid_targets[:, 0], valid_targets[:, 1])
    signed_angle_error = torch.atan2(
        torch.sin(predicted_angle - target_angle),
        torch.cos(predicted_angle - target_angle),
    )
    angle_error_deg = signed_angle_error.abs() * (180.0 / math.pi)
    signed_error_deg = signed_angle_error * (180.0 / math.pi)
    print(f"{prefix}_MAE={float(angle_error_deg.mean()):.4f}")
    for threshold in (10.0, 30.0):
        print(
            f"{prefix}_accuracy@{_format_scalar(threshold)}deg="
            f"{float((angle_error_deg <= threshold).float().mean()):.4f}"
        )
    print(f"{prefix}_signed_mean={float(signed_error_deg.mean()):.4f}")


def _metric_suffix(value: str) -> str:
    suffix = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower())
    return suffix.strip("_") or "unknown"


def _print_first_path_angle_nlos_bucket_diagnostics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    semantic_keys: list[SemanticKey],
    field: str,
) -> None:
    values = sorted(
        {
            semantic_key_attribute_raw_value(key, field)
            for key in semantic_keys
            if key.los_status != "los"
        }
    )
    for value in values:
        bucket_mask = torch.tensor(
            [
                key.los_status != "los"
                and semantic_key_attribute_raw_value(key, field) == value
                for key in semantic_keys
            ],
            dtype=torch.bool,
            device=valid_mask.device,
        )
        _print_angle_diagnostics(
            f"first_path_angle_nlos_{field}_{_metric_suffix(value)}",
            predictions=predictions,
            targets=targets,
            mask=valid_mask & bucket_mask,
        )


def _print_first_path_angle_diagnostics(
    physics_predictions: torch.Tensor,
    physics_targets: torch.Tensor,
    physics_masks: torch.Tensor,
    semantic_keys: list[SemanticKey],
) -> None:
    sin_idx = PHYSICS_TARGET_NAMES.index("first_path_aoa_az_sin")
    cos_idx = PHYSICS_TARGET_NAMES.index("first_path_aoa_az_cos")
    predictions = physics_predictions[:, [sin_idx, cos_idx]]
    targets = physics_targets[:, [sin_idx, cos_idx]]
    mask = (
        physics_masks[:, sin_idx].bool()
        & physics_masks[:, cos_idx].bool()
        & torch.isfinite(targets).all(dim=1)
    )
    los_mask = torch.tensor(
        [key.los_status == "los" for key in semantic_keys],
        dtype=torch.bool,
        device=mask.device,
    )
    _print_angle_diagnostics(
        "first_path_angle",
        predictions=predictions,
        targets=targets,
        mask=mask,
    )
    _print_angle_diagnostics(
        "first_path_angle_los",
        predictions=predictions,
        targets=targets,
        mask=mask & los_mask,
    )
    _print_angle_diagnostics(
        "first_path_angle_nlos",
        predictions=predictions,
        targets=targets,
        mask=mask & ~los_mask,
    )


def _strong_k_targets(
    physics_raw_targets: torch.Tensor,
    physics_masks: torch.Tensor,
    semantic_keys: list[SemanticKey],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_idx = _physics_target_index("k_factor_db")
    raw_k = physics_raw_targets[:, k_idx]
    valid_mask = physics_masks[:, k_idx] & torch.tensor(
        [key.k_factor_bin == "strong" for key in semantic_keys],
        dtype=torch.bool,
        device=raw_k.device,
    )
    labels = torch.full_like(raw_k, fill_value=-1, dtype=torch.long)
    positions = torch.zeros_like(raw_k)
    for bin_idx, (_, lower, upper) in enumerate(STRONG_K_FACTOR_DIAGNOSTIC_BINS):
        upper_mask = (
            raw_k <= upper
            if bin_idx == len(STRONG_K_FACTOR_DIAGNOSTIC_BINS) - 1
            else raw_k < upper
        )
        mask = valid_mask & (raw_k >= lower) & upper_mask
        labels = torch.where(mask, torch.full_like(labels, bin_idx), labels)
        positions = torch.where(
            mask,
            ((raw_k - lower) / max(upper - lower, 1e-6)).clamp(0.0, 1.0),
            positions,
        )
    return labels, positions, labels >= 0


def _decode_strong_k(bin_labels: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    lowers = torch.tensor(
        [lower for _, lower, _ in STRONG_K_FACTOR_DIAGNOSTIC_BINS],
        dtype=positions.dtype,
        device=positions.device,
    )
    widths = torch.tensor(
        [upper - lower for _, lower, upper in STRONG_K_FACTOR_DIAGNOSTIC_BINS],
        dtype=positions.dtype,
        device=positions.device,
    )
    return lowers[bin_labels] + positions.clamp(0.0, 1.0) * widths[bin_labels]


def _print_k_factor_diagnostics(
    physics_predictions: torch.Tensor,
    physics_raw_targets: torch.Tensor,
    physics_masks: torch.Tensor,
    semantic_keys: list[SemanticKey],
) -> None:
    k_idx = _physics_target_index("k_factor_db")
    mask = physics_masks[:, k_idx]
    if not bool(mask.any()):
        print("strong_k_MAE=nan")
        return

    raw_predictions = _physics_raw_predictions(physics_predictions)
    predictions = raw_predictions[:, k_idx][mask]
    targets = physics_raw_targets[:, k_idx][mask]
    valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()

    strong_mask = torch.tensor(
        [semantic_keys[idx].k_factor_bin == "strong" for idx in valid_indices],
        dtype=torch.bool,
        device=predictions.device,
    )
    if bool(strong_mask.any()):
        print(f"strong_k_MAE={float((predictions[strong_mask] - targets[strong_mask]).abs().mean()):.4f}")
    else:
        print("strong_k_MAE=nan")


def _print_k_factor_group_diagnostics(
    prefix: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    count = int(mask.sum().item())
    print(f"{prefix}_count={count}")
    if count == 0:
        print(f"{prefix}_MAE=nan")
        print(f"{prefix}_accuracy@3=nan")
        print(f"{prefix}_signed_mean=nan")
        print(f"{prefix}_pearson=nan")
        print(f"{prefix}_target_range=nan,nan")
        print(f"{prefix}_pred_range=nan,nan")
        return
    group_predictions = predictions[mask]
    group_targets = targets[mask]
    group_errors = group_predictions - group_targets
    group_abs_errors = group_errors.abs()
    print(f"{prefix}_MAE={float(group_abs_errors.mean()):.4f}")
    print(f"{prefix}_accuracy@3={float((group_abs_errors <= 3.0).float().mean()):.4f}")
    print(f"{prefix}_signed_mean={float(group_errors.mean()):.4f}")
    print(f"{prefix}_pearson={_safe_pearson(group_predictions, group_targets):.4f}")
    print(
        f"{prefix}_target_range="
        f"{float(group_targets.min()):.4f},{float(group_targets.max()):.4f}"
    )
    print(
        f"{prefix}_pred_range="
        f"{float(group_predictions.min()):.4f},{float(group_predictions.max()):.4f}"
    )


def _print_structured_physical_description_metrics(
    physics_predictions: torch.Tensor,
    physics_raw_targets: torch.Tensor,
    physics_masks: torch.Tensor,
    semantic_keys: list[SemanticKey],
    prototype_logits: torch.Tensor,
    prototype_keys: list[SemanticKey],
    example_count: int,
) -> None:
    raw_predictions = _physics_raw_predictions(physics_predictions)
    predicted_labels = prototype_logits.argmax(dim=1)
    los_true = torch.tensor(
        [1 if key.los_status == "los" else 0 for key in semantic_keys],
        dtype=torch.long,
    )
    los_pred = torch.tensor(
        [
            1 if prototype_keys[int(label)].los_status == "los" else 0
            for label in predicted_labels.tolist()
        ],
        dtype=torch.long,
    )
    los_accuracy = (los_true == los_pred).float().mean()
    print(f"physical_description_los_status_accuracy={float(los_accuracy):.4f}")

    tolerance_hits = []
    tolerance_masks = []
    for field in PHYSICAL_DESCRIPTION_FIELDS:
        idx = _physics_target_index(field)
        mask = physics_masks[:, idx]
        if not bool(mask.any()):
            print(f"physical_description_{field}_MAE=nan")
            print(
                f"physical_description_{field}_accuracy@{_format_scalar(PHYSICAL_DESCRIPTION_TOLERANCES[field])}=nan"
            )
            continue
        errors = (raw_predictions[:, idx] - physics_raw_targets[:, idx]).abs()
        mae = errors[mask].mean()
        tolerance = PHYSICAL_DESCRIPTION_TOLERANCES[field]
        hits = errors <= tolerance
        accuracy = hits[mask].float().mean()
        print(f"physical_description_{field}_MAE={float(mae):.4f}")
        print(
            f"physical_description_{field}_accuracy@{_format_scalar(tolerance)}={float(accuracy):.4f}"
        )
        tolerance_hits.append(hits)
        tolerance_masks.append(mask)

    if tolerance_hits:
        valid_sentence_mask = torch.ones_like(tolerance_masks[0], dtype=torch.bool)
        within_tolerance = torch.ones_like(tolerance_hits[0], dtype=torch.bool)
        for mask in tolerance_masks:
            valid_sentence_mask &= mask
        for hits in tolerance_hits:
            within_tolerance &= hits
        sentence_correct = (los_true == los_pred) & within_tolerance
        if bool(valid_sentence_mask.any()):
            sentence_accuracy = sentence_correct[valid_sentence_mask].float().mean()
            print(
                f"physical_description_sentence_level_accuracy={float(sentence_accuracy):.4f}"
            )
            print(
                f"physical_description_sentence_level_valid_samples={int(valid_sentence_mask.sum().item())}"
            )
        else:
            print("physical_description_sentence_level_accuracy=nan")
            print("physical_description_sentence_level_valid_samples=0")

    for idx in range(min(example_count, raw_predictions.shape[0])):
        predicted_record = _structured_physical_record(
            los_status=prototype_keys[int(predicted_labels[idx])].los_status,
            raw_values=raw_predictions[idx],
        )
        target_record = _structured_physical_record(
            los_status=semantic_keys[idx].los_status,
            raw_values=physics_raw_targets[idx],
        )
        print(f"physical_description_example_{idx + 1}_pred_struct={predicted_record}")
        print(f"physical_description_example_{idx + 1}_true_struct={target_record}")
        print(
            f"physical_description_example_{idx + 1}_pred_text="
            f"{_render_physical_description(predicted_record)}"
        )
        print(
            f"physical_description_example_{idx + 1}_true_text="
            f"{_render_physical_description(target_record)}"
        )


def _build_signal_description_payload(
    *,
    samples,
    physics_predictions: torch.Tensor,
    physics_raw_targets: torch.Tensor,
    semantic_keys: list[SemanticKey],
    prototype_logits: torch.Tensor,
    prototype_keys: list[SemanticKey],
    semantic_logits: torch.Tensor | None,
    los_delay_predictions: torch.Tensor,
    los_delay_raw_targets: torch.Tensor,
    los_delay_masks: torch.Tensor,
    los_angle_predictions: torch.Tensor,
    los_angle_targets: torch.Tensor,
    los_angle_masks: torch.Tensor,
    reflection_count_predictions: torch.Tensor,
) -> dict:
    raw_predictions = _physics_raw_predictions(physics_predictions).cpu()
    physics_raw_targets = physics_raw_targets.cpu()
    predicted_labels = (
        semantic_logits.argmax(dim=1).cpu()
        if semantic_logits is not None
        else prototype_logits.argmax(dim=1).cpu()
    )
    los_delay_raw_predictions = (los_delay_predictions.cpu() * 3000.0)
    los_angle_predictions = los_angle_predictions.cpu()
    los_angle_targets = los_angle_targets.cpu()
    los_delay_raw_targets = los_delay_raw_targets.cpu()
    los_delay_masks = los_delay_masks.cpu().bool()
    los_angle_masks = los_angle_masks.cpu().bool()
    reflection_idx = _physics_target_index("reflection_count")
    reflection_raw_predictions = (
        reflection_count_predictions.cpu()
        * PHYSICS_TARGET_SCALES[reflection_idx]
        + PHYSICS_TARGET_OFFSETS[reflection_idx]
    )

    predicted_records = []
    target_records = []
    predicted_texts = []
    target_texts = []
    comparisons = []
    for idx in range(raw_predictions.shape[0]):
        predicted_key = prototype_keys[int(predicted_labels[idx])]
        target_key = semantic_keys[idx]
        predicted_record = _signal_description_record(
            predicted_key,
            raw_predictions[idx],
            los_delay_ns=float(los_delay_raw_predictions[idx]),
            los_angle_sincos=los_angle_predictions[idx],
            reflection_count=float(reflection_raw_predictions[idx]),
        )
        target_record = _signal_description_record(
            target_key,
            physics_raw_targets[idx],
            los_delay_ns=(
                float(los_delay_raw_targets[idx])
                if bool(los_delay_masks[idx])
                else math.nan
            ),
            los_angle_sincos=(
                los_angle_targets[idx]
                if bool(los_angle_masks[idx])
                else None
            ),
            reflection_count=float(physics_raw_targets[idx, reflection_idx]),
        )
        predicted_text = _render_signal_description(predicted_record)
        target_text = _render_signal_description(target_record)
        predicted_records.append(predicted_record)
        target_records.append(target_record)
        predicted_texts.append(predicted_text)
        target_texts.append(target_text)
        sample = samples[idx]
        comparisons.append(
            {
                "index": idx,
                "group_id": getattr(sample, "group_id", ""),
                "config_key": getattr(sample, "config_key", ""),
                "predicted_signal_description": predicted_text,
                "target_signal_description": target_text,
                "predicted_record": predicted_record,
                "target_record": target_record,
            }
        )

    return {
        "predicted_signal_descriptions": predicted_texts,
        "target_signal_descriptions": target_texts,
        "predicted_signal_records": predicted_records,
        "target_signal_records": target_records,
        "comparisons": comparisons,
        "predicted_semantic_labels": predicted_labels,
        "physics_raw_predictions": raw_predictions,
        "physics_raw_targets": physics_raw_targets,
        "los_delay_raw_predictions": los_delay_raw_predictions,
        "los_delay_raw_targets": los_delay_raw_targets,
        "los_angle_predictions": los_angle_predictions,
        "los_angle_targets": los_angle_targets,
        "reflection_count_raw_predictions": reflection_raw_predictions,
    }


def _print_signal_description_examples(payload: dict, example_count: int) -> None:
    predicted_texts = payload["predicted_signal_descriptions"]
    target_texts = payload["target_signal_descriptions"]
    for idx in range(min(example_count, len(predicted_texts))):
        print(
            f"signal_description_example_{idx + 1}_pred_text="
            f"{predicted_texts[idx]}"
        )
        print(
            f"signal_description_example_{idx + 1}_true_text="
            f"{target_texts[idx]}"
        )


def _save_signal_description_payload(path: str, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"saved_signal_descriptions={output_path}")


def _print_first_path_power_diagnostics(
    base_physics_predictions: torch.Tensor,
    physics_predictions: torch.Tensor,
    enhanced_first_path_power_predictions: torch.Tensor,
    physics_raw_targets: torch.Tensor,
    physics_masks: torch.Tensor,
    semantic_keys: list[SemanticKey],
    prototype_logits: torch.Tensor,
    prototype_keys: list[SemanticKey],
) -> None:
    first_path_power_idx = _physics_target_index("first_path_power_dbw")
    first_path_power_mask = physics_masks[:, first_path_power_idx]
    if not bool(first_path_power_mask.any()):
        print("base_first_power_MAE=nan")
        print("enhanced_first_power_MAE=nan")
        print("final_first_power_MAE=nan")
        print("los_base_first_power_MAE=nan")
        print("los_enhanced_first_power_MAE=nan")
        print("los_final_first_power_MAE=nan")
        print("nlos_base_first_power_MAE=nan")
        print("nlos_enhanced_first_power_MAE=nan")
        print("nlos_final_first_power_MAE=nan")
        print("oracle_los_base_nlos_enhanced_first_power_MAE=nan")
        print("oracle_los_enhanced_nlos_base_first_power_MAE=nan")
        print("predicted_los_status_first_power_accuracy=nan")
        print("predicted_los_base_nlos_enhanced_first_power_MAE=nan")
        return

    base_physics_raw_predictions = _physics_raw_predictions(base_physics_predictions)
    final_physics_raw_predictions = _physics_raw_predictions(physics_predictions)
    enhanced_raw_predictions = (
        enhanced_first_path_power_predictions * PHYSICS_TARGET_SCALES[first_path_power_idx]
        + PHYSICS_TARGET_OFFSETS[first_path_power_idx]
    )

    masked_base_raw = base_physics_raw_predictions[first_path_power_mask, first_path_power_idx]
    masked_final_raw = final_physics_raw_predictions[first_path_power_mask, first_path_power_idx]
    masked_enhanced_raw = enhanced_raw_predictions[first_path_power_mask]
    masked_target_raw = physics_raw_targets[first_path_power_mask, first_path_power_idx]

    print(f"base_first_power_MAE={float((masked_base_raw - masked_target_raw).abs().mean()):.4f}")
    print(f"enhanced_first_power_MAE={float((masked_enhanced_raw - masked_target_raw).abs().mean()):.4f}")
    print(f"final_first_power_MAE={float((masked_final_raw - masked_target_raw).abs().mean()):.4f}")

    def print_group_mae(prefix: str, group_mask: torch.Tensor) -> None:
        mask = first_path_power_mask & group_mask
        if not bool(mask.any()):
            print(f"{prefix}_base_first_power_MAE=nan")
            print(f"{prefix}_enhanced_first_power_MAE=nan")
            print(f"{prefix}_final_first_power_MAE=nan")
            return
        group_base_raw = base_physics_raw_predictions[mask, first_path_power_idx]
        group_enhanced_raw = enhanced_raw_predictions[mask]
        group_final_raw = final_physics_raw_predictions[mask, first_path_power_idx]
        group_target_raw = physics_raw_targets[mask, first_path_power_idx]
        print(
            f"{prefix}_base_first_power_MAE="
            f"{float((group_base_raw - group_target_raw).abs().mean()):.4f}"
        )
        print(
            f"{prefix}_enhanced_first_power_MAE="
            f"{float((group_enhanced_raw - group_target_raw).abs().mean()):.4f}"
        )
        print(
            f"{prefix}_final_first_power_MAE="
            f"{float((group_final_raw - group_target_raw).abs().mean()):.4f}"
        )

    los_mask = torch.tensor(
        [key.los_status == "los" for key in semantic_keys],
        dtype=torch.bool,
        device=first_path_power_mask.device,
    )
    print_group_mae("los", los_mask)
    print_group_mae("nlos", ~los_mask)

    def print_fused_mae(prefix: str, use_base_mask: torch.Tensor) -> None:
        masked_use_base = use_base_mask[first_path_power_mask]
        fused_raw = torch.where(masked_use_base, masked_base_raw, masked_enhanced_raw)
        print(
            f"{prefix}_first_power_MAE="
            f"{float((fused_raw - masked_target_raw).abs().mean()):.4f}"
        )

    print_fused_mae("oracle_los_base_nlos_enhanced", los_mask)
    print_fused_mae("oracle_los_enhanced_nlos_base", ~los_mask)

    predicted_labels = prototype_logits.argmax(dim=1)
    predicted_los_mask = torch.tensor(
        [
            prototype_keys[int(label)].los_status == "los"
            for label in predicted_labels.tolist()
        ],
        dtype=torch.bool,
        device=first_path_power_mask.device,
    )
    predicted_los_accuracy = (
        predicted_los_mask[first_path_power_mask]
        == los_mask[first_path_power_mask]
    ).float().mean()
    print(
        "predicted_los_status_first_power_accuracy="
        f"{float(predicted_los_accuracy):.4f}"
    )
    print_fused_mae("predicted_los_base_nlos_enhanced", predicted_los_mask)


def _print_delay_family_diagnostics(
    first_path_delay_predictions: torch.Tensor,
    first_path_delay_bin_logits: torch.Tensor | None,
    first_path_delay_bin_positions: torch.Tensor | None,
    first_path_delay_bin_fused_raw: torch.Tensor | None,
    first_path_delay_bin_soft_fused_raw: torch.Tensor | None,
    los_delay_predictions: torch.Tensor,
    los_angle_predictions: torch.Tensor,
    physics_raw_targets: torch.Tensor,
    physics_masks: torch.Tensor,
    los_delay_raw_targets: torch.Tensor,
    los_delay_masks: torch.Tensor,
    los_angle_targets: torch.Tensor,
    los_angle_masks: torch.Tensor,
    semantic_keys: list[SemanticKey],
) -> None:
    del first_path_delay_bin_positions
    first_delay_idx = _physics_target_index("first_path_delay_ns")
    first_delay_raw = (
        first_path_delay_predictions * PHYSICS_TARGET_SCALES[first_delay_idx]
        + PHYSICS_TARGET_OFFSETS[first_delay_idx]
    )
    first_delay_mask = physics_masks[:, first_delay_idx]
    los_sample_mask = torch.tensor(
        [key.los_status == "los" for key in semantic_keys],
        dtype=torch.bool,
    )
    if bool(first_delay_mask.any()):
        first_delay_target = physics_raw_targets[first_delay_mask, first_delay_idx]
        first_delay_pred = first_delay_raw[first_delay_mask]
        first_delay_errors = first_delay_pred - first_delay_target
        print(f"first_path_delay_context_count={int(first_delay_mask.sum().item())}")
        print(f"first_path_delay_context_MAE={float(first_delay_errors.abs().mean()):.4f}")
        print(f"first_path_delay_context_signed_mean={float(first_delay_errors.mean()):.4f}")
        _print_first_path_delay_group_metrics(
            prefix="first_path_delay_context_los",
            predictions=first_delay_raw,
            targets=physics_raw_targets[:, first_delay_idx],
            mask=first_delay_mask & los_sample_mask,
        )
        _print_first_path_delay_group_metrics(
            prefix="first_path_delay_context_nlos",
            predictions=first_delay_raw,
            targets=physics_raw_targets[:, first_delay_idx],
            mask=first_delay_mask & ~los_sample_mask,
        )
        if first_path_delay_bin_logits is not None:
            _print_first_path_delay_bin_head_diagnostics(
                target_raw=first_delay_target,
                bin_logits=first_path_delay_bin_logits[first_delay_mask],
            )
        if first_path_delay_bin_fused_raw is not None:
            fused_pred = first_path_delay_bin_fused_raw.to(dtype=first_delay_raw.dtype)
            fused_target = physics_raw_targets[:, first_delay_idx]
            fused_errors = fused_pred[first_delay_mask] - first_delay_target
            print(f"first_path_delay_bin_fused_MAE={float(fused_errors.abs().mean()):.4f}")
            print(f"first_path_delay_bin_fused_signed_mean={float(fused_errors.mean()):.4f}")
        if first_path_delay_bin_soft_fused_raw is not None:
            soft_fused_pred = first_path_delay_bin_soft_fused_raw.to(dtype=first_delay_raw.dtype)
            fused_target = physics_raw_targets[:, first_delay_idx]
            soft_fused_errors = soft_fused_pred[first_delay_mask] - first_delay_target
            print(f"first_path_delay_bin_soft_fused_MAE={float(soft_fused_errors.abs().mean()):.4f}")
            print(f"first_path_delay_bin_soft_fused_signed_mean={float(soft_fused_errors.mean()):.4f}")
    else:
        print("first_path_delay_context_count=0")
        print("first_path_delay_context_MAE=nan")
        print("first_path_delay_context_signed_mean=nan")
        print("first_path_delay_context_los_count=0")
        print("first_path_delay_context_los_MAE=nan")
        print("first_path_delay_context_los_signed_mean=nan")
        print("first_path_delay_context_nlos_count=0")
        print("first_path_delay_context_nlos_MAE=nan")
        print("first_path_delay_context_nlos_signed_mean=nan")
        if first_path_delay_bin_fused_raw is not None:
            print("first_path_delay_bin_fused_MAE=nan")
            print("first_path_delay_bin_fused_signed_mean=nan")
        if first_path_delay_bin_soft_fused_raw is not None:
            print("first_path_delay_bin_soft_fused_MAE=nan")
            print("first_path_delay_bin_soft_fused_signed_mean=nan")
    los_delay_mask = los_delay_masks & los_sample_mask
    _print_los_delay_diagnostics(
        predictions=los_delay_predictions * 3000.0,
        targets=los_delay_raw_targets,
        mask=los_delay_mask,
    )
    los_angle_mask = (
        los_angle_masks
        & los_sample_mask
        & torch.isfinite(los_angle_targets).all(dim=1)
    )
    _print_los_angle_diagnostics(
        predictions=los_angle_predictions,
        targets=los_angle_targets,
        mask=los_angle_mask,
    )


def _print_los_angle_diagnostics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    _print_angle_diagnostics(
        "los_angle",
        predictions=predictions,
        targets=targets,
        mask=mask,
    )


def _print_first_path_delay_group_metrics(
    prefix: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    count = int(mask.sum().item())
    print(f"{prefix}_count={count}")
    if count == 0:
        print(f"{prefix}_MAE=nan")
        print(f"{prefix}_signed_mean=nan")
        return
    errors = predictions[mask] - targets[mask]
    print(f"{prefix}_MAE={float(errors.abs().mean()):.4f}")
    print(f"{prefix}_signed_mean={float(errors.mean()):.4f}")


def _first_path_delay_bin_targets(raw_first_path_delay_ns: torch.Tensor) -> torch.Tensor:
    targets = torch.full_like(raw_first_path_delay_ns, fill_value=-1, dtype=torch.long)
    for class_idx, (_, lower, upper) in enumerate(FIRST_PATH_DELAY_DIAGNOSTIC_BINS):
        upper_mask = (
            raw_first_path_delay_ns <= upper
            if class_idx == len(FIRST_PATH_DELAY_DIAGNOSTIC_BINS) - 1
            else raw_first_path_delay_ns < upper
        )
        mask = (
            torch.isfinite(raw_first_path_delay_ns)
            & (raw_first_path_delay_ns >= lower)
            & upper_mask
        )
        targets = torch.where(mask, torch.full_like(targets, class_idx), targets)
    return targets


def _print_first_path_delay_bin_head_diagnostics(
    target_raw: torch.Tensor,
    bin_logits: torch.Tensor,
) -> None:
    target_labels = _first_path_delay_bin_targets(target_raw)
    valid_mask = target_labels >= 0
    if not bool(valid_mask.any()):
        print("first_path_delay_bin_head_count=0")
        print("first_path_delay_bin_head_accuracy=nan")
        return

    predicted_labels = bin_logits.argmax(dim=1)
    print(f"first_path_delay_bin_head_count={int(valid_mask.sum().item())}")
    print(
        "first_path_delay_bin_head_label_order="
        + ",".join(FIRST_PATH_DELAY_BIN_LABELS)
    )
    print(
        f"first_path_delay_bin_head_accuracy="
        f"{float((predicted_labels[valid_mask] == target_labels[valid_mask]).float().mean()):.4f}"
    )


def _print_los_delay_diagnostics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    print(f"los_delay_context_count={int(mask.sum().item())}")
    if not bool(mask.any()):
        print("los_delay_context_MAE=nan")
        print("los_delay_context_signed_mean=nan")
        print("los_delay_context_RMSE=nan")
        print("los_delay_context_pearson=nan")
        print("los_delay_context_accuracy@50ns=nan")
        return

    pred = predictions[mask].float()
    target = targets[mask].float()
    errors = pred - target
    abs_errors = errors.abs()
    print(f"los_delay_context_MAE={float(abs_errors.mean()):.4f}")
    print(f"los_delay_context_signed_mean={float(errors.mean()):.4f}")
    print(f"los_delay_context_RMSE={float(torch.sqrt(errors.square().mean())):.4f}")
    print(f"los_delay_context_pearson={_safe_pearson(pred, target):.4f}")
    print(f"los_delay_context_accuracy@50ns={float((abs_errors <= 50.0).float().mean()):.4f}")


def _print_delay_spread_diagnostics(
    base_physics_predictions: torch.Tensor,
    physics_predictions: torch.Tensor,
    csi_delay_spread_predictions: torch.Tensor,
    enhanced_delay_spread_predictions: torch.Tensor,
    profile_delay_spread_predictions: torch.Tensor,
    profile_direct_delay_spread_predictions: torch.Tensor,
    delay_spread_context_predictions: torch.Tensor | None,
    delay_spread_bin_logits: torch.Tensor | None,
    delay_spread_bin_positions: torch.Tensor | None,
    physics_raw_targets: torch.Tensor,
    physics_masks: torch.Tensor,
    raw_beta_ns: float,
) -> None:
    del base_physics_predictions
    del csi_delay_spread_predictions
    del enhanced_delay_spread_predictions
    del profile_delay_spread_predictions
    del profile_direct_delay_spread_predictions
    delay_spread_idx = _physics_target_index("delay_spread_ns")
    delay_spread_mask = physics_masks[:, delay_spread_idx]
    if not bool(delay_spread_mask.any()):
        print("delay_context_spread_MAE=nan")
        print("delay_context_spread_normalized_MAE=nan")
        print("delay_context_spread_normalized_smooth_l1=nan")
        print("delay_context_spread_raw_huber=nan")
        print("final_delay_spread_MAE=nan")
        print("final_delay_spread_normalized_MAE=nan")
        print("final_delay_spread_normalized_smooth_l1=nan")
        print("final_delay_spread_raw_huber=nan")
        return

    final_physics_raw_predictions = _physics_raw_predictions(physics_predictions)
    target_scale = PHYSICS_TARGET_SCALES[delay_spread_idx].to(
        device=physics_raw_targets.device,
        dtype=physics_raw_targets.dtype,
    )
    target_offset = PHYSICS_TARGET_OFFSETS[delay_spread_idx].to(
        device=physics_raw_targets.device,
        dtype=physics_raw_targets.dtype,
    )
    delay_context_raw_predictions = None
    if delay_spread_context_predictions is not None:
        delay_context_raw_predictions = (
            delay_spread_context_predictions * target_scale
            + target_offset
        )

    masked_final_raw = final_physics_raw_predictions[delay_spread_mask, delay_spread_idx]
    masked_target_raw = physics_raw_targets[delay_spread_mask, delay_spread_idx]
    masked_final_normalized = physics_predictions[delay_spread_mask, delay_spread_idx]
    masked_target_normalized = (masked_target_raw - target_offset) / target_scale
    masked_delay_context_raw = (
        delay_context_raw_predictions[delay_spread_mask]
        if delay_context_raw_predictions is not None
        else None
    )
    masked_delay_context_normalized = (
        delay_spread_context_predictions[delay_spread_mask]
        if delay_spread_context_predictions is not None
        else None
    )

    def _smooth_l1_mean(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(predictions, targets, reduction="mean")

    def _raw_huber_mean(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        beta = max(float(raw_beta_ns), 1e-6)
        abs_error = (predictions - targets).abs()
        return torch.where(
            abs_error < beta,
            0.5 * abs_error.square() / beta,
            abs_error - 0.5 * beta,
        ).mean()

    if masked_delay_context_raw is not None and masked_delay_context_normalized is not None:
        print(f"delay_context_spread_MAE={float((masked_delay_context_raw - masked_target_raw).abs().mean()):.4f}")
        print(
            "delay_context_spread_normalized_MAE="
            f"{float((masked_delay_context_normalized - masked_target_normalized).abs().mean()):.4f}"
        )
        print(
            "delay_context_spread_normalized_smooth_l1="
            f"{float(_smooth_l1_mean(masked_delay_context_normalized, masked_target_normalized)):.4f}"
        )
        print(
            "delay_context_spread_raw_huber="
            f"{float(_raw_huber_mean(masked_delay_context_raw, masked_target_raw)):.4f}"
        )
    else:
        print("delay_context_spread_MAE=nan")
        print("delay_context_spread_normalized_MAE=nan")
        print("delay_context_spread_normalized_smooth_l1=nan")
        print("delay_context_spread_raw_huber=nan")
    print(f"final_delay_spread_MAE={float((masked_final_raw - masked_target_raw).abs().mean()):.4f}")
    print(
        "final_delay_spread_normalized_MAE="
        f"{float((masked_final_normalized - masked_target_normalized).abs().mean()):.4f}"
    )
    print(
        "final_delay_spread_normalized_smooth_l1="
        f"{float(_smooth_l1_mean(masked_final_normalized, masked_target_normalized)):.4f}"
    )
    print(
        "final_delay_spread_raw_huber="
        f"{float(_raw_huber_mean(masked_final_raw, masked_target_raw)):.4f}"
    )
    if delay_spread_bin_logits is not None:
        masked_bin_logits = delay_spread_bin_logits[delay_spread_mask]
        masked_bin_positions = (
            delay_spread_bin_positions[delay_spread_mask]
            if delay_spread_bin_positions is not None
            else None
        )
        _print_delay_spread_bin_head_diagnostics(
            target_raw=masked_target_raw,
            bin_logits=masked_bin_logits,
            bin_positions=masked_bin_positions,
        )


def _delay_spread_bin_targets(raw_delay_spread_ns: torch.Tensor) -> torch.Tensor:
    targets = torch.full_like(raw_delay_spread_ns, fill_value=-1, dtype=torch.long)
    for class_idx, (_, lower, upper) in enumerate(DELAY_SPREAD_DIAGNOSTIC_BINS):
        upper_mask = (
            raw_delay_spread_ns <= upper
            if class_idx == len(DELAY_SPREAD_DIAGNOSTIC_BINS) - 1
            else raw_delay_spread_ns < upper
        )
        mask = torch.isfinite(raw_delay_spread_ns) & (raw_delay_spread_ns >= lower) & upper_mask
        targets = torch.where(mask, torch.full_like(targets, class_idx), targets)
    return targets


def _delay_spread_finite_bin_bounds(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    label_to_bounds = {
        label: (lower, upper)
        for label, lower, upper in DELAY_SPREAD_POSITION_BINS
    }
    finite_mask = torch.tensor(
        [label in label_to_bounds for label in DELAY_SPREAD_BIN_LABELS],
        device=device,
        dtype=torch.bool,
    )
    lower = torch.zeros(len(DELAY_SPREAD_BIN_LABELS), device=device, dtype=dtype)
    upper = torch.zeros(len(DELAY_SPREAD_BIN_LABELS), device=device, dtype=dtype)
    for idx, label in enumerate(DELAY_SPREAD_BIN_LABELS):
        if label not in label_to_bounds:
            continue
        bin_lower, bin_upper = label_to_bounds[label]
        lower[idx] = float(bin_lower)
        upper[idx] = float(bin_upper)
    return lower, upper, finite_mask


def _fuse_delay_spread_from_bin_position(
    bin_logits: torch.Tensor,
    bin_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower, upper, finite_mask = _delay_spread_finite_bin_bounds(
        device=bin_logits.device,
        dtype=bin_logits.dtype,
    )
    predicted_labels = bin_logits.argmax(dim=1)
    covered_mask = finite_mask[predicted_labels]
    position = bin_positions.to(dtype=bin_logits.dtype).clamp(0.0, 1.0)
    fused = lower[predicted_labels] + position * (
        upper[predicted_labels] - lower[predicted_labels]
    )
    return fused, covered_mask


def _fuse_delay_spread_soft_from_bin_position(
    bin_logits: torch.Tensor,
    bin_positions: torch.Tensor,
) -> torch.Tensor:
    lower, upper, finite_mask = _delay_spread_finite_bin_bounds(
        device=bin_logits.device,
        dtype=bin_logits.dtype,
    )
    position = bin_positions.to(dtype=bin_logits.dtype).clamp(0.0, 1.0).unsqueeze(1)
    candidates = lower.unsqueeze(0) + position * (upper - lower).unsqueeze(0)
    probabilities = torch.softmax(bin_logits, dim=1)
    probabilities = probabilities * finite_mask.to(dtype=probabilities.dtype).unsqueeze(0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return (probabilities * candidates).sum(dim=1)


def _print_delay_spread_bin_head_diagnostics(
    target_raw: torch.Tensor,
    bin_logits: torch.Tensor,
    bin_positions: torch.Tensor | None = None,
) -> None:
    target_labels = _delay_spread_bin_targets(target_raw)
    valid_mask = target_labels >= 0
    if not bool(valid_mask.any()):
        print("delay_spread_bin_head_count=0")
        print("delay_spread_bin_head_accuracy=nan")
        return

    predicted_labels = bin_logits.argmax(dim=1)
    valid_targets = target_labels[valid_mask]
    valid_predictions = predicted_labels[valid_mask]
    confusion = torch.zeros(
        len(DELAY_SPREAD_BIN_LABELS),
        len(DELAY_SPREAD_BIN_LABELS),
        dtype=torch.long,
    )
    for true_label, pred_label in zip(valid_targets.tolist(), valid_predictions.tolist()):
        confusion[int(true_label), int(pred_label)] += 1
    adjacent_hits = (valid_predictions - valid_targets).abs() <= 1
    far_misses = (valid_predictions - valid_targets).abs() > 1
    print(f"delay_spread_bin_head_count={int(valid_mask.sum().item())}")
    print(
        "delay_spread_bin_head_label_order="
        + ",".join(DELAY_SPREAD_BIN_LABELS)
    )
    print(
        f"delay_spread_bin_head_accuracy="
        f"{float((predicted_labels[valid_mask] == target_labels[valid_mask]).float().mean()):.4f}"
    )
    print(
        f"delay_spread_bin_head_adjacent_accuracy="
        f"{float(adjacent_hits.float().mean()):.4f}"
    )
    print(
        f"delay_spread_bin_head_far_miss_fraction="
        f"{float(far_misses.float().mean()):.4f}"
    )
    print(
        "delay_spread_bin_head_target_histogram="
        + ",".join(
            f"{label}:{int((target_labels[valid_mask] == idx).sum().item())}"
            for idx, label in enumerate(DELAY_SPREAD_BIN_LABELS)
        )
    )
    print(
        "delay_spread_bin_head_confusion="
        + ";".join(
            f"{DELAY_SPREAD_BIN_LABELS[row]}:"
            + ",".join(
                f"{DELAY_SPREAD_BIN_LABELS[col]}:{int(confusion[row, col].item())}"
                for col in range(len(DELAY_SPREAD_BIN_LABELS))
            )
            for row in range(len(DELAY_SPREAD_BIN_LABELS))
        )
    )
    print(
        "delay_spread_bin_head_prediction_histogram="
        + ",".join(
            f"{label}:{int((predicted_labels[valid_mask] == idx).sum().item())}"
            for idx, label in enumerate(DELAY_SPREAD_BIN_LABELS)
        )
    )
    if bin_positions is None:
        print("delay_spread_bin_fused_MAE=nan")
        print("delay_spread_bin_soft_fused_MAE=nan")
        print("delay_spread_bin_fused_covered_count=0")
        return

    hard_fused_raw, covered_mask = _fuse_delay_spread_from_bin_position(
        bin_logits,
        bin_positions,
    )
    soft_fused_raw = _fuse_delay_spread_soft_from_bin_position(
        bin_logits,
        bin_positions,
    )
    hard_valid_mask = valid_mask & covered_mask
    print(f"delay_spread_bin_fused_covered_count={int(hard_valid_mask.sum().item())}")
    if bool(hard_valid_mask.any()):
        hard_errors = hard_fused_raw[hard_valid_mask] - target_raw[hard_valid_mask].to(
            dtype=hard_fused_raw.dtype
        )
        print(f"delay_spread_bin_fused_MAE={float(hard_errors.abs().mean()):.4f}")
        print(f"delay_spread_bin_fused_signed_mean={float(hard_errors.mean()):.4f}")
    else:
        print("delay_spread_bin_fused_MAE=nan")
        print("delay_spread_bin_fused_signed_mean=nan")
    soft_errors = soft_fused_raw[valid_mask] - target_raw[valid_mask].to(
        dtype=soft_fused_raw.dtype
    )
    print(f"delay_spread_bin_soft_fused_MAE={float(soft_errors.abs().mean()):.4f}")
    print(f"delay_spread_bin_soft_fused_signed_mean={float(soft_errors.mean()):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-mode", choices=["prototype", "instance", "multipositive"])
    parser.add_argument(
        "--semantic-key-mode",
        choices=semantic_key_mode_choices(),
        help="Semantic key granularity for evaluation. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--token-norm-mode",
        choices=["std", "rms", "none"],
        help="CSI token normalization mode. Defaults to checkpoint args.",
    )
    power_branch_group = parser.add_mutually_exclusive_group()
    power_branch_group.add_argument(
        "--enable-power-branch",
        action="store_true",
        help="Enable the power branch regardless of checkpoint args.",
    )
    power_branch_group.add_argument(
        "--disable-power-branch",
        action="store_true",
        help="Disable the power branch regardless of checkpoint args.",
    )
    parser.add_argument(
        "--first-path-power-gate-mode",
        choices=("none", "base"),
        help="Override first-path-power final fusion mode. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        help="Drop semantic classes with fewer than this many samples before evaluation. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--filter-attribute-values",
        action="append",
        help=(
            "Keep only samples whose SemanticKey field matches listed values. "
            "Use FIELD=VALUE[,VALUE...], e.g. k_factor_bin=weak,strong. Defaults to checkpoint args."
        ),
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="Keep only the first N samples after semantic remapping and class-size filtering. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--limit-samples-by-attribute",
        choices=semantic_key_field_choices(),
        help="Keep up to N samples per value of this SemanticKey field. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--limit-samples-per-attribute-value",
        type=int,
        help="Number of samples to keep per value when --limit-samples-by-attribute is set. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--max-delay-spread-ns",
        type=float,
        help="Drop samples whose delay_spread_ns is greater than or equal to this value. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--attribute-classifier-fields",
        nargs="+",
        choices=semantic_key_field_choices(),
        help="SemanticKey fields for attribute classifier evaluation. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--attribute-binary-threshold",
        action="append",
        help=(
            "Apply a fixed binary threshold for an attribute as FIELD=THRESHOLD. "
            "The score is logit[class_1]-logit[class_0], using the printed label order."
        ),
    )
    parser.add_argument(
        "--physical-caption-examples",
        type=int,
        default=3,
        help="How many structured physical caption prediction examples to print.",
    )
    parser.add_argument(
        "--save-signal-descriptions",
        help=(
            "Optional .pt path for predicted-vs-target signal description texts "
            "and their structured records."
        ),
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(
        args.data_path,
        args.checkpoint,
        args.batch_size,
        device,
        text_mode_override=args.text_mode,
        min_class_size_override=args.min_class_size,
        semantic_key_mode_override=args.semantic_key_mode,
        token_norm_mode_override=args.token_norm_mode,
        use_power_branch_override=(
            True if args.enable_power_branch else False if args.disable_power_branch else None
        ),
        first_path_power_gate_mode_override=args.first_path_power_gate_mode,
        attribute_fields_override=tuple(args.attribute_classifier_fields) if args.attribute_classifier_fields else None,
        filter_attribute_values_override=(
            parse_attribute_value_filters(args.filter_attribute_values)
            if args.filter_attribute_values is not None
            else None
        ),
        limit_samples_override=args.limit_samples,
        limit_samples_by_attribute_override=args.limit_samples_by_attribute,
        limit_samples_per_attribute_value_override=args.limit_samples_per_attribute_value,
        max_delay_spread_ns_override=args.max_delay_spread_ns,
        attribute_binary_thresholds=parse_attribute_binary_thresholds(args.attribute_binary_threshold),
        physical_caption_examples=args.physical_caption_examples,
        save_signal_descriptions_path=args.save_signal_descriptions,
    )


if __name__ == "__main__":
    main()
