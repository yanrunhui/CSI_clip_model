from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.caption import CaptionGenerator
from data.dataset import (
    PHYSICS_TARGET_NAMES,
    PreprocessedCSIDataset,
    SyntheticCSIDataset,
    apply_semantic_key_mode,
    build_synthetic_samples,
    collate_fn,
    expand_physics_aux_targets,
    physics_aux_target_choices,
    semantic_key_mode_choices,
)
from data.semantic_key import (
    FIRST_POWER_DBW_BIN_LABELS,
    FIRST_POWER_DBW_BINS,
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
    DELAY_SPREAD_BIN_LABELS,
    DELAY_SPREAD_TAIL_LABELS,
    FIRST_PATH_DELAY_BIN_LABELS,
    K_FACTOR_STRONG_BIN_LABELS,
    REFLECTION_COUNT_BIN_LABELS,
)
from models.text_encoder import PhysicsTextEncoder
from training.scheduler import build_lr_scheduler
from training.trainer import TrainConfig, Trainer


def load_train_config(path: str | None) -> dict:
    config_path = Path(path) if path is not None else ROOT / "configs" / "train.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("train", data)


def cfg_get(config: dict, key: str, fallback):
    value = config.get(key, fallback)
    return fallback if value is None else value


DEFAULT_FIRST_PATH_POWER_BIN_WEIGHTS = {
    label: 1.0
    for label, _, _ in FIRST_POWER_DBW_BINS
}

DEFAULT_K_FACTOR_LOSS_WEIGHTS = {
    "weak": 1.0,
    "strong_low": 1.0,
    "strong_mid": 1.0,
    "strong_high": 1.0,
    "strong_very_high": 1.0,
}

DEFAULT_STRONG_K_BIN_WEIGHTS = {
    label: 1.0
    for label in K_FACTOR_STRONG_BIN_LABELS
}

DEFAULT_DELAY_SPREAD_BIN_WEIGHTS = {
    label: 1.0
    for label in DELAY_SPREAD_BIN_LABELS
}

DEFAULT_FIRST_PATH_DELAY_BIN_WEIGHTS = {
    label: 1.0
    for label in FIRST_PATH_DELAY_BIN_LABELS
}


def parse_first_path_power_bin_weights(value) -> dict[str, float]:
    if value is None:
        return dict(DEFAULT_FIRST_PATH_POWER_BIN_WEIGHTS)
    entries = [value] if isinstance(value, str) else value
    if isinstance(entries, dict):
        parsed = {str(label): float(weight) for label, weight in entries.items()}
    else:
        parsed = dict(DEFAULT_FIRST_PATH_POWER_BIN_WEIGHTS)
        for entry in entries:
            if "=" not in str(entry):
                raise ValueError(
                    "--first-path-power-bin-weight entries must use LABEL=WEIGHT, "
                    f"got {entry!r}."
                )
            label, raw_weight = str(entry).split("=", 1)
            parsed[label.strip()] = float(raw_weight)
    unknown = [label for label in parsed if label not in DEFAULT_FIRST_PATH_POWER_BIN_WEIGHTS]
    if unknown:
        raise ValueError(
            f"Unknown first-path-power bin labels: {unknown}. "
            f"Choose from: {', '.join(DEFAULT_FIRST_PATH_POWER_BIN_WEIGHTS)}"
        )
    for label, weight in parsed.items():
        if weight <= 0.0:
            raise ValueError(f"first-path-power bin weight for {label!r} must be positive.")
    return parsed


def format_first_path_power_bin_weights(weights: dict[str, float]) -> str:
    return ",".join(
        f"{label}={float(weights[label]):.3f}"
        for label in DEFAULT_FIRST_PATH_POWER_BIN_WEIGHTS
    )


def parse_delay_spread_bin_weights(value) -> dict[str, float] | None:
    if value is None:
        return None
    entries = [value] if isinstance(value, str) else value
    parsed = dict(DEFAULT_DELAY_SPREAD_BIN_WEIGHTS)
    if isinstance(entries, dict):
        parsed.update({str(label): float(weight) for label, weight in entries.items()})
    else:
        for entry in entries:
            if "=" not in str(entry):
                raise ValueError(
                    "--delay-spread-bin-weight entries must use LABEL=WEIGHT, "
                    f"got {entry!r}."
                )
            label, raw_weight = str(entry).split("=", 1)
            parsed[label.strip()] = float(raw_weight)
    unknown = [label for label in parsed if label not in DEFAULT_DELAY_SPREAD_BIN_WEIGHTS]
    if unknown:
        raise ValueError(
            f"Unknown delay-spread bin labels: {unknown}. "
            f"Choose from: {', '.join(DEFAULT_DELAY_SPREAD_BIN_WEIGHTS)}"
        )
    for label, weight in parsed.items():
        if weight <= 0.0:
            raise ValueError(f"delay-spread bin weight for {label!r} must be positive.")
    return parsed


def format_delay_spread_bin_weights(weights: dict[str, float] | None) -> str:
    if weights is None:
        return "none"
    return ",".join(
        f"{label}={float(weights.get(label, 1.0)):.3f}"
        for label in DEFAULT_DELAY_SPREAD_BIN_WEIGHTS
    )


def parse_first_path_delay_bin_weights(value) -> dict[str, float] | None:
    if value is None:
        return None
    entries = [value] if isinstance(value, str) else value
    parsed = dict(DEFAULT_FIRST_PATH_DELAY_BIN_WEIGHTS)
    if isinstance(entries, dict):
        parsed.update({str(label): float(weight) for label, weight in entries.items()})
    else:
        for entry in entries:
            if "=" not in str(entry):
                raise ValueError(
                    "--first-path-delay-bin-weight entries must use LABEL=WEIGHT, "
                    f"got {entry!r}."
                )
            label, raw_weight = str(entry).split("=", 1)
            parsed[label.strip()] = float(raw_weight)
    unknown = [label for label in parsed if label not in DEFAULT_FIRST_PATH_DELAY_BIN_WEIGHTS]
    if unknown:
        raise ValueError(
            f"Unknown first-path-delay bin labels: {unknown}. "
            f"Choose from: {', '.join(DEFAULT_FIRST_PATH_DELAY_BIN_WEIGHTS)}"
        )
    for label, weight in parsed.items():
        if weight <= 0.0:
            raise ValueError(f"first-path-delay bin weight for {label!r} must be positive.")
    return parsed


def format_first_path_delay_bin_weights(weights: dict[str, float] | None) -> str:
    if weights is None:
        return "none"
    return ",".join(
        f"{label}={float(weights.get(label, 1.0)):.3f}"
        for label in DEFAULT_FIRST_PATH_DELAY_BIN_WEIGHTS
    )


def parse_first_path_delay_tail_labels(value) -> tuple[str, ...]:
    if value is None:
        return ("1040_1280",)
    entries = (value,) if isinstance(value, str) else tuple(value)
    labels = tuple(
        label.strip()
        for entry in entries
        for label in str(entry).split(",")
        if label.strip()
    )
    if not labels:
        raise ValueError("estimated PDP tail labels must contain at least one label.")
    unknown = [label for label in labels if label not in FIRST_PATH_DELAY_BIN_LABELS]
    if unknown:
        raise ValueError(
            f"Unknown estimated PDP tail labels: {unknown}. "
            f"Choose from: {', '.join(FIRST_PATH_DELAY_BIN_LABELS)}"
        )
    return labels


def parse_k_factor_loss_weights(value) -> dict[str, float] | None:
    if value is None:
        return None
    entries = [value] if isinstance(value, str) else value
    parsed = dict(DEFAULT_K_FACTOR_LOSS_WEIGHTS)
    if isinstance(entries, dict):
        parsed.update({str(label): float(weight) for label, weight in entries.items()})
    else:
        for entry in entries:
            if "=" not in str(entry):
                raise ValueError(
                    "--k-factor-loss-weight entries must use LABEL=WEIGHT, "
                    f"got {entry!r}."
                )
            label, raw_weight = str(entry).split("=", 1)
            parsed[label.strip()] = float(raw_weight)
    unknown = [label for label in parsed if label not in DEFAULT_K_FACTOR_LOSS_WEIGHTS]
    if unknown:
        raise ValueError(
            f"Unknown K-factor loss weight labels: {unknown}. "
            f"Choose from: {', '.join(DEFAULT_K_FACTOR_LOSS_WEIGHTS)}"
        )
    for label, weight in parsed.items():
        if weight <= 0.0:
            raise ValueError(f"K-factor loss weight for {label!r} must be positive.")
    return parsed


def format_k_factor_loss_weights(weights: dict[str, float] | None) -> str:
    if weights is None:
        return "none"
    return ",".join(
        f"{label}={float(weights[label]):.3f}"
        for label in DEFAULT_K_FACTOR_LOSS_WEIGHTS
    )


def parse_strong_k_bin_weights(value) -> dict[str, float] | None:
    if value is None:
        return None
    entries = [value] if isinstance(value, str) else value
    parsed = dict(DEFAULT_STRONG_K_BIN_WEIGHTS)
    if isinstance(entries, dict):
        parsed.update({str(label): float(weight) for label, weight in entries.items()})
    else:
        for entry in entries:
            if "=" not in str(entry):
                raise ValueError(
                    "--strong-k-bin-weight entries must use LABEL=WEIGHT, "
                    f"got {entry!r}."
                )
            label, raw_weight = str(entry).split("=", 1)
            parsed[label.strip()] = float(raw_weight)
    unknown = [label for label in parsed if label not in DEFAULT_STRONG_K_BIN_WEIGHTS]
    if unknown:
        raise ValueError(
            f"Unknown strong K-factor bin labels: {unknown}. "
            f"Choose from: {', '.join(DEFAULT_STRONG_K_BIN_WEIGHTS)}"
        )
    for label, weight in parsed.items():
        if weight <= 0.0:
            raise ValueError(f"strong K-factor bin weight for {label!r} must be positive.")
    return parsed


def format_strong_k_bin_weights(weights: dict[str, float] | None) -> str:
    if weights is None:
        return "none"
    return ",".join(
        f"{label}={float(weights[label]):.3f}"
        for label in DEFAULT_STRONG_K_BIN_WEIGHTS
    )


def parse_histogram_text(value: str, num_bins: int) -> list[int]:
    if not value:
        return [0] * num_bins
    parts = [part for part in str(value).split(",") if part != ""]
    if len(parts) != num_bins:
        return [0] * num_bins
    return [int(part) for part in parts]


def sum_histogram_metric(
    metrics: list[dict[str, float]],
    key: str,
    num_bins: int,
) -> list[int]:
    total = [0] * num_bins
    for item in metrics:
        values = parse_histogram_text(str(item.get(key, "")), num_bins)
        total = [left + right for left, right in zip(total, values)]
    return total


def format_histogram_counts(labels: tuple[str, ...], counts: list[int]) -> str:
    return ",".join(f"{label}:{count}" for label, count in zip(labels, counts))


def parse_aux_regression_targets(value) -> tuple[str, ...]:
    if value is None:
        return ("all",)
    if isinstance(value, str):
        targets = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        targets = tuple(str(part) for part in value)
    if not targets:
        return ("all",)
    if "all" in targets:
        if len(targets) > 1:
            raise ValueError("--aux-regression-targets cannot combine 'all' with specific targets.")
        return ("all",)
    choices = physics_aux_target_choices()
    unknown = [target for target in targets if target not in choices]
    if unknown:
        raise ValueError(
            f"Unknown aux regression targets: {unknown}. "
            f"Choose from: {', '.join(choices)}"
        )
    return expand_physics_aux_targets(targets)


def aux_regression_indices(targets: tuple[str, ...]) -> tuple[int, ...] | None:
    if targets == ("all",):
        return None
    return tuple(PHYSICS_TARGET_NAMES.index(target) for target in targets)


def parse_attribute_fields(value) -> tuple[str, ...]:
    if value is None:
        return default_attribute_fields()
    if isinstance(value, str):
        fields = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        fields = tuple(str(part) for part in value)
    unknown = [field for field in fields if field not in semantic_key_field_choices()]
    if unknown:
        raise ValueError(
            f"Unknown attribute classifier fields: {unknown}. "
            f"Choose from: {', '.join(semantic_key_field_choices())}"
        )
    return fields


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


def format_attribute_remap(remap: dict[str, dict[str, tuple[str, ...]]]) -> str:
    if not remap:
        return "none"
    return ";".join(
        f"{field}="
        + ",".join(f"{mapped}:{'|'.join(values)}" for mapped, values in mapping.items())
        for field, mapping in sorted(remap.items())
    )


def parse_interaction_count_fields(value) -> tuple[str, ...]:
    if value is None:
        return ("reflection_count",)
    if isinstance(value, str):
        fields = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        fields = tuple(str(part) for part in value)
    if not fields:
        return ("reflection_count",)
    unknown = [field for field in fields if field != "reflection_count"]
    if unknown:
        raise ValueError(
            f"Unknown interaction_count_fields: {unknown}. "
            "Choose from: reflection_count"
        )
    return fields


def format_interaction_count_fields(fields: tuple[str, ...]) -> str:
    return ",".join(fields) if fields else "none"


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


def serialize_prototype_keys(keys: list[SemanticKey]) -> list[dict[str, str]]:
    fields = tuple(SemanticKey.__dataclass_fields__)
    return [{field: str(getattr(key, field)) for field in fields} for key in keys]


def deserialize_prototype_keys(value) -> list[SemanticKey] | None:
    if value is None:
        return None
    fields = tuple(SemanticKey.__dataclass_fields__)
    keys: list[SemanticKey] = []
    for item in value:
        if isinstance(item, SemanticKey):
            keys.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(
                "checkpoint prototype_keys entries must be SemanticKey objects or field mappings."
            )
        missing = [field for field in fields if field not in item]
        if missing:
            raise ValueError(
                f"checkpoint prototype_keys entry is missing fields: {', '.join(missing)}"
            )
        keys.append(SemanticKey(**{field: str(item[field]) for field in fields}))
    return keys


def _format_prototype_key(key: SemanticKey) -> str:
    fields = tuple(SemanticKey.__dataclass_fields__)
    return ",".join(f"{field}={getattr(key, field)}" for field in fields)


def assert_checkpoint_prototype_compatibility(
    checkpoint: dict | None,
    current_keys: list[SemanticKey],
    expected_shape: tuple[int, ...] | None = None,
    context: str = "checkpoint",
) -> None:
    if checkpoint is None:
        return
    prototype_tensor = checkpoint.get("model_state", {}).get("prototypes")
    if prototype_tensor is None:
        return
    if expected_shape is not None and tuple(prototype_tensor.shape) != tuple(expected_shape):
        raise ValueError(
            f"{context} prototype tensor shape mismatch: "
            f"checkpoint={tuple(prototype_tensor.shape)} current={tuple(expected_shape)}."
        )
    checkpoint_keys = deserialize_prototype_keys(checkpoint.get("prototype_keys"))
    if checkpoint_keys is None:
        raise ValueError(
            f"{context} contains learnable prototypes but is missing prototype_keys metadata; "
            "cannot verify prototype ordering."
        )
    if len(checkpoint_keys) != len(current_keys):
        raise ValueError(
            f"{context} prototype key count mismatch: "
            f"checkpoint={len(checkpoint_keys)} current={len(current_keys)}."
        )
    for idx, (checkpoint_key, current_key) in enumerate(zip(checkpoint_keys, current_keys)):
        if checkpoint_key != current_key:
            raise ValueError(
                f"{context} prototype key mismatch at index {idx}: "
                f"checkpoint[{idx}]={_format_prototype_key(checkpoint_key)} "
                f"current[{idx}]={_format_prototype_key(current_key)}."
            )


def build_prototype_bank(
    samples,
    tokenizer: CaptionTokenizer,
    max_caption_len: int = 48,
) -> tuple[list[SemanticKey], list[str], torch.Tensor, torch.Tensor, dict[SemanticKey, int], torch.Tensor]:
    caption_generator = CaptionGenerator()
    unique_keys = sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
    prototype_captions = [caption_generator.generate_canonical(key) for key in unique_keys]
    tokenizer.build_vocab(prototype_captions)
    tokenizer.build_vocab(sample.prop_caption for sample in samples)
    tokenizer.build_vocab(sample.instance_caption for sample in samples)
    prototype_token_ids = torch.stack(
        [tokenizer.encode(caption, max_len=max_caption_len).ids for caption in prototype_captions],
        dim=0,
    )
    prototype_token_mask = torch.stack(
        [tokenizer.encode(caption, max_len=max_caption_len).mask for caption in prototype_captions],
        dim=0,
    )
    prototype_label_map = {key: idx for idx, key in enumerate(unique_keys)}
    key_counts = Counter(sample.semantic_key for sample in samples)
    prototype_class_counts = torch.tensor(
        [key_counts[key] for key in unique_keys],
        dtype=torch.long,
    )
    return (
        unique_keys,
        prototype_captions,
        prototype_token_ids,
        prototype_token_mask,
        prototype_label_map,
        prototype_class_counts,
    )


def build_attribute_banks(
    samples,
    fields: tuple[str, ...],
    attribute_remap: dict[str, dict[str, tuple[str, ...]]] | None = None,
):
    label_maps: dict[str, dict[str, int]] = {}
    class_counts: dict[str, torch.Tensor] = {}
    for field in fields:
        values = sorted(
            {
                semantic_key_attribute_value(sample.semantic_key, field, attribute_remap)
                for sample in samples
            }
        )
        label_map = {value: idx for idx, value in enumerate(values)}
        counts = Counter(
            semantic_key_attribute_value(sample.semantic_key, field, attribute_remap)
            for sample in samples
        )
        label_maps[field] = label_map
        class_counts[field] = torch.tensor(
            [counts[value] for value in values],
            dtype=torch.long,
        )
    return label_maps, class_counts


def build_components_from_samples(
    samples,
    device: torch.device,
    batch_size: int = 128,
    temperature: float = 0.07,
    token_norm_mode: str = "std",
    use_power_branch: bool = False,
    first_path_power_mode: str = "residual",
    first_path_power_use_internal_gate: bool = True,
    use_delay_spread_head: bool = False,
    detach_delay_spread_features: bool = False,
    detach_first_path_delay_features: bool = True,
    use_delay_specific_encoder: bool = False,
    use_los_angle_context_encoder: bool = False,
    use_first_path_angle_context_encoder: bool = False,
    attribute_fields: tuple[str, ...] = (),
    attribute_remap: dict[str, dict[str, tuple[str, ...]]] | None = None,
    tokenizer_word2id: dict[str, int] | None = None,
):
    source_samples = samples if isinstance(samples, list) else samples.samples
    tokenizer = CaptionTokenizer()
    if tokenizer_word2id is not None:
        tokenizer.word2id = dict(tokenizer_word2id)
        tokenizer.id2word = {idx: word for word, idx in tokenizer.word2id.items()}
        tokenizer.next_id = max(tokenizer.id2word) + 1
    (
        prototype_keys,
        prototype_captions,
        prototype_token_ids,
        prototype_token_mask,
        prototype_label_map,
        prototype_class_counts,
    ) = build_prototype_bank(source_samples, tokenizer)
    attribute_label_maps, attribute_class_counts = build_attribute_banks(
        source_samples,
        attribute_fields,
        attribute_remap=attribute_remap,
    )

    dataset = SyntheticCSIDataset(source_samples) if isinstance(samples, list) else samples
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )

    csi_encoder = CSIEncoder(
        d_token=8,
        d_model=384,
        d_clip=256,
        token_norm_mode=token_norm_mode,
    )
    text_encoder = PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300))
    model = CSIClip(
        csi_encoder,
        text_encoder,
        num_prototypes=len(prototype_keys),
        semantic_num_classes=len(prototype_keys),
        embed_dim=256,
        temperature=temperature,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        use_power_branch=use_power_branch,
        first_path_power_mode=first_path_power_mode,
        first_path_power_use_internal_gate=first_path_power_use_internal_gate,
        use_delay_spread_head=use_delay_spread_head,
        detach_delay_spread_features=detach_delay_spread_features,
        detach_first_path_delay_features=detach_first_path_delay_features,
        use_delay_specific_encoder=use_delay_specific_encoder,
        use_los_angle_context_encoder=use_los_angle_context_encoder,
        use_first_path_angle_context_encoder=use_first_path_angle_context_encoder,
        los_angle_context_token_norm_mode=token_norm_mode,
        attribute_num_classes={
            field: len(label_map)
            for field, label_map in attribute_label_maps.items()
        },
    ).to(device)
    prototype_bank = {
        "keys": prototype_keys,
        "captions": prototype_captions,
        "token_ids": prototype_token_ids,
        "token_mask": prototype_token_mask,
        "label_map": prototype_label_map,
        "class_counts": prototype_class_counts,
        "attribute_label_maps": attribute_label_maps,
        "attribute_class_counts": attribute_class_counts,
    }
    return loader, model, tokenizer, prototype_bank


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def freeze_text_and_prototypes(model: CSIClip) -> None:
    freeze_module(model.text)
    if model.prototypes is not None:
        model.prototypes.requires_grad = False


def checkpoint_has_compatible_prototypes(
    model: CSIClip,
    checkpoint: dict | None,
    prototype_keys: list[SemanticKey] | None = None,
) -> bool:
    if checkpoint is None or model.prototypes is None:
        return False
    tensor = checkpoint.get("model_state", {}).get("prototypes")
    if tensor is None or tuple(tensor.shape) != tuple(model.prototypes.shape):
        return False
    if prototype_keys is None:
        return True
    try:
        assert_checkpoint_prototype_compatibility(
            checkpoint,
            prototype_keys,
            expected_shape=tuple(model.prototypes.shape),
            context="checkpoint",
        )
    except ValueError:
        return False
    return True


def initialize_prototypes_from_canonical_text(
    model: CSIClip,
    prototype_token_ids: torch.Tensor,
    prototype_token_mask: torch.Tensor,
) -> None:
    if model.prototypes is None:
        return
    was_training = model.text.training
    model.text.eval()
    with torch.no_grad():
        text_features = model.encode_text(
            prototype_token_ids.to(next(model.parameters()).device),
            prototype_token_mask.to(next(model.parameters()).device),
            normalize=True,
        )
        model.initialize_prototypes(text_features, normalize=False)
    if was_training:
        model.text.train()


def trainable_parameters(model: torch.nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_transfer_checkpoint(path: str | None, device: torch.device) -> dict | None:
    if path is None:
        return None
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def checkpoint_first_path_delay_bin_label_mismatch(checkpoint: dict) -> bool:
    checkpoint_labels = checkpoint.get("args", {}).get("first_path_delay_bin_label_order")
    if checkpoint_labels is None:
        return False
    return tuple(checkpoint_labels) != tuple(FIRST_PATH_DELAY_BIN_LABELS)


def load_model_state_compatible(model: torch.nn.Module, checkpoint: dict) -> None:
    state_dict = checkpoint["model_state"]
    model_state = model.state_dict()
    skip_prefixes: tuple[str, ...] = ()
    if checkpoint_first_path_delay_bin_label_mismatch(checkpoint):
        skip_prefixes = (
            "first_path_delay_bin_classifier.",
            "first_path_delay_bin_position_head.",
        )
        print(
            "checkpoint first-path-delay bin labels differ from current labels; "
            "skipping first_path_delay_bin_classifier and first_path_delay_bin_position_head."
        )
    compatible_state = {
        name: value
        for name, value in state_dict.items()
        if (
            name in model_state
            and model_state[name].shape == value.shape
            and not name.startswith(skip_prefixes)
        )
    }
    skipped = sorted(set(state_dict) - set(compatible_state))
    missing = sorted(set(model_state) - set(compatible_state))
    model.load_state_dict(compatible_state, strict=False)
    print(
        f"loaded_checkpoint_keys={len(compatible_state)} "
        f"skipped_checkpoint_keys={len(skipped)} "
        f"new_model_keys={len(missing)}"
    )
    if skipped:
        print(f"skipped_checkpoint_key_examples={','.join(skipped[:20])}")
    if missing:
        print(f"new_model_key_examples={','.join(missing[:20])}")


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


def build_demo_components(
    device: torch.device,
    semantic_key_mode: str = "full",
    token_norm_mode: str = "std",
    use_power_branch: bool = False,
    first_path_power_mode: str = "residual",
    first_path_power_use_internal_gate: bool = True,
    use_delay_spread_head: bool = False,
    detach_delay_spread_features: bool = False,
    detach_first_path_delay_features: bool = True,
    use_delay_specific_encoder: bool = False,
    use_los_angle_context_encoder: bool = False,
    use_first_path_angle_context_encoder: bool = False,
    attribute_fields: tuple[str, ...] = (),
    attribute_remap: dict[str, dict[str, tuple[str, ...]]] | None = None,
):
    caption_generator = CaptionGenerator()
    samples = build_synthetic_samples(128, caption_generator=caption_generator)
    samples = apply_semantic_key_mode(samples, semantic_key_mode)
    samples = filter_samples_by_attribute_values(
        samples,
        implied_attribute_value_filters(attribute_fields, attribute_remap),
    )
    return build_components_from_samples(
        samples,
        device=device,
        batch_size=32,
        token_norm_mode=token_norm_mode,
        use_power_branch=use_power_branch,
        first_path_power_mode=first_path_power_mode,
        first_path_power_use_internal_gate=first_path_power_use_internal_gate,
        use_delay_spread_head=use_delay_spread_head,
        detach_delay_spread_features=detach_delay_spread_features,
        detach_first_path_delay_features=detach_first_path_delay_features,
        use_delay_specific_encoder=use_delay_specific_encoder,
        use_los_angle_context_encoder=use_los_angle_context_encoder,
        use_first_path_angle_context_encoder=use_first_path_angle_context_encoder,
        attribute_fields=attribute_fields,
        attribute_remap=attribute_remap,
    )


def build_real_components(
    data_path: str,
    device: torch.device,
    batch_size: int = 128,
    temperature: float = 0.07,
    token_norm_mode: str = "std",
    use_power_branch: bool = False,
    first_path_power_mode: str = "residual",
    first_path_power_use_internal_gate: bool = True,
    use_delay_spread_head: bool = False,
    detach_delay_spread_features: bool = False,
    detach_first_path_delay_features: bool = True,
    use_delay_specific_encoder: bool = False,
    use_los_angle_context_encoder: bool = False,
    use_first_path_angle_context_encoder: bool = False,
    min_class_size: int = 1,
    semantic_key_mode: str = "full",
    attribute_fields: tuple[str, ...] = (),
    attribute_remap: dict[str, dict[str, tuple[str, ...]]] | None = None,
    filter_attribute_values: dict[str, tuple[str, ...]] | None = None,
    limit_samples: int | None = None,
    limit_samples_by_attribute: str | None = None,
    limit_samples_per_attribute_value: int | None = None,
    max_delay_spread_ns: float | None = None,
    tokenizer_word2id: dict[str, int] | None = None,
):
    dataset = PreprocessedCSIDataset.from_pt(data_path)
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
    filter_attribute_values = {
        **implied_attribute_value_filters(attribute_fields, attribute_remap),
        **(filter_attribute_values or {}),
    }
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
    return build_components_from_samples(
        samples,
        device=device,
        batch_size=batch_size,
        temperature=temperature,
        token_norm_mode=token_norm_mode,
        use_power_branch=use_power_branch,
        first_path_power_mode=first_path_power_mode,
        first_path_power_use_internal_gate=first_path_power_use_internal_gate,
        use_delay_spread_head=use_delay_spread_head,
        detach_delay_spread_features=detach_delay_spread_features,
        detach_first_path_delay_features=detach_first_path_delay_features,
        use_delay_specific_encoder=use_delay_specific_encoder,
        use_los_angle_context_encoder=use_los_angle_context_encoder,
        use_first_path_angle_context_encoder=use_first_path_angle_context_encoder,
        attribute_fields=attribute_fields,
        attribute_remap=attribute_remap,
        tokenizer_word2id=tokenizer_word2id,
    )


def run_smoke_test(
    device: torch.device,
    text_mode: str = "prototype",
    csi_to_text_weight: float = 1.0,
    semantic_classifier_weight: float = 0.0,
    semantic_classifier_class_weight: str = "none",
    semantic_classifier_logit_adjustment: float = 0.0,
    attribute_classifier_weight: float = 0.0,
    attribute_classifier_fields: tuple[str, ...] = (),
    attribute_classifier_class_weight: str = "none",
    attribute_classifier_logit_adjustment: float = 0.0,
    aux_regression_weight: float = 0.0,
    aux_regression_targets: tuple[str, ...] = ("all",),
    k_factor_loss_weights: dict[str, float] | None = None,
    strong_k_bin_classifier_weight: float = 0.0,
    strong_k_position_weight: float = 0.0,
    strong_k_bin_weights: dict[str, float] | None = None,
    direct_power_weight: float = 0.0,
    delay_spread_weight: float = 0.0,
    delay_spread_raw_weight: float = 0.0,
    delay_spread_raw_beta_ns: float = 20.0,
    first_path_delay_weight: float = 0.0,
    first_path_delay_raw_weight: float = 0.0,
    first_path_delay_fused_raw_weight: float = 0.0,
    first_path_delay_raw_beta_ns: float = 20.0,
    first_path_delay_bin_classifier_weight: float = 0.0,
    first_path_delay_bin_position_weight: float = 0.0,
    first_path_delay_bin_consistency_weight: float = 0.0,
    first_path_delay_bin_weights: dict[str, float] | None = None,
    estimated_pdp_tail_bin_weight: float = 0.0,
    estimated_pdp_tail_labels: tuple[str, ...] = ("1040_1280",),
    estimated_pdp_tail_gate_mode: str = "target",
    first_path_delay_tail_underestimate_weight: float = 0.0,
    los_delay_weight: float = 0.0,
    los_delay_nonnegative_weight: float = 0.0,
    use_physics_calibration_loss: bool = False,
    los_delay_consistency_weight: float = 0.0,
    los_angle_weight: float = 0.0,
    first_path_angle_weight: float = 0.0,
    first_path_angle_nlos_weight: float = 0.0,
    delay_spread_teacher_weight: float = 0.1,
    delay_spread_bin_weights: dict[str, float] | None = None,
    delay_spread_bin_classifier_weight: float = 0.0,
    delay_spread_bin_position_weight: float = 0.0,
    delay_spread_tail_classifier_weight: float = 0.0,
    interaction_count_classifier_weight: float = 0.0,
    interaction_count_regression_weight: float = 0.0,
    reflection_count_classifier_weight: float = 0.0,
    reflection_count_regression_weight: float = 0.0,
    reflection_count_nlos_weight: float = 1.0,
    interaction_count_soft_labels: bool = False,
    interaction_count_fields: tuple[str, ...] = ("reflection_count",),
    first_path_power_bin_classifier_weight: float = 0.0,
    first_path_power_bin_position_weight: float = 0.0,
    first_path_power_bin_weights: dict[str, float] | None = None,
    first_path_power_nlos_weight: float = 1.0,
    first_path_power_gate_mode: str = "none",
    first_path_power_mode: str = "residual",
    first_path_power_use_internal_gate: bool = True,
    nlos_enhanced_power_loss: bool = False,
    first_path_power_delta_limit: float = 0.5,
    multipositive_distance_threshold: float = 0.25,
    multipositive_positive_mode: str = "semantic_and_physics",
    min_class_size_for_multipositive: int = 2,
    semantic_key_mode: str = "full",
    token_norm_mode: str = "std",
    use_power_branch: bool = False,
    detach_delay_spread_features: bool = False,
    detach_first_path_delay_features: bool = True,
    use_delay_specific_encoder: bool = False,
    use_los_angle_context_encoder: bool = False,
    use_first_path_angle_context_encoder: bool = False,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> None:
    loader, model, _, prototype_bank = build_demo_components(
        device,
        semantic_key_mode=semantic_key_mode,
        token_norm_mode=token_norm_mode,
        use_power_branch=use_power_branch,
        first_path_power_mode=first_path_power_mode,
        first_path_power_use_internal_gate=first_path_power_use_internal_gate,
        use_delay_spread_head=delay_spread_weight > 0.0 or delay_spread_raw_weight > 0.0,
        detach_delay_spread_features=detach_delay_spread_features,
        detach_first_path_delay_features=detach_first_path_delay_features,
        use_delay_specific_encoder=use_delay_specific_encoder,
        use_los_angle_context_encoder=use_los_angle_context_encoder,
        use_first_path_angle_context_encoder=use_first_path_angle_context_encoder,
        attribute_fields=attribute_classifier_fields,
        attribute_remap=attribute_remap,
    )
    model.first_path_power_delta_limit = float(first_path_power_delta_limit)
    initialize_prototypes_from_canonical_text(
        model,
        prototype_bank["token_ids"],
        prototype_bank["token_mask"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = build_lr_scheduler(optimizer, total_epochs=2, warmup_epochs=1)
    trainer = Trainer(
        model,
        optimizer,
        device,
        prototype_token_ids=prototype_bank["token_ids"],
        prototype_token_mask=prototype_bank["token_mask"],
        prototype_label_map=prototype_bank["label_map"],
        prototype_class_counts=prototype_bank["class_counts"],
        attribute_label_maps=prototype_bank["attribute_label_maps"],
        attribute_class_counts=prototype_bank["attribute_class_counts"],
        attribute_remap=attribute_remap,
    )

    for epoch in range(1, 3):
        for batch in loader:
            metrics = trainer.train_step(
                batch,
                epoch=epoch,
                cfg=TrainConfig(
                    epochs=2,
                    text_mode=text_mode,
                    csi_to_text_weight=csi_to_text_weight,
                    semantic_classifier_weight=semantic_classifier_weight,
                    semantic_classifier_class_weight=semantic_classifier_class_weight,
                    semantic_classifier_logit_adjustment=semantic_classifier_logit_adjustment,
                    attribute_classifier_weight=attribute_classifier_weight,
                    attribute_classifier_class_weight=attribute_classifier_class_weight,
                    attribute_classifier_logit_adjustment=attribute_classifier_logit_adjustment,
                    aux_regression_weight=aux_regression_weight,
                    aux_regression_indices=aux_regression_indices(aux_regression_targets),
                    k_factor_loss_weights=k_factor_loss_weights,
                    strong_k_bin_classifier_weight=strong_k_bin_classifier_weight,
                    strong_k_position_weight=strong_k_position_weight,
                    strong_k_bin_weights=strong_k_bin_weights,
                    first_path_power_bin_classifier_weight=first_path_power_bin_classifier_weight,
                    first_path_power_bin_position_weight=first_path_power_bin_position_weight,
                    direct_power_weight=direct_power_weight,
                    delay_spread_weight=delay_spread_weight,
                    delay_spread_raw_weight=delay_spread_raw_weight,
                    delay_spread_raw_beta_ns=delay_spread_raw_beta_ns,
                    first_path_delay_weight=first_path_delay_weight,
                    first_path_delay_raw_weight=first_path_delay_raw_weight,
                    first_path_delay_fused_raw_weight=first_path_delay_fused_raw_weight,
                    first_path_delay_raw_beta_ns=first_path_delay_raw_beta_ns,
                    first_path_delay_bin_classifier_weight=first_path_delay_bin_classifier_weight,
                    first_path_delay_bin_position_weight=first_path_delay_bin_position_weight,
                    first_path_delay_bin_consistency_weight=first_path_delay_bin_consistency_weight,
                    first_path_delay_bin_weights=first_path_delay_bin_weights,
                    estimated_pdp_tail_bin_weight=estimated_pdp_tail_bin_weight,
                    estimated_pdp_tail_labels=estimated_pdp_tail_labels,
                    estimated_pdp_tail_gate_mode=estimated_pdp_tail_gate_mode,
                    first_path_delay_tail_underestimate_weight=first_path_delay_tail_underestimate_weight,
                    los_delay_weight=los_delay_weight,
                    los_delay_nonnegative_weight=los_delay_nonnegative_weight,
                    use_physics_calibration_loss=use_physics_calibration_loss,
                    los_delay_consistency_weight=los_delay_consistency_weight,
                    los_angle_weight=los_angle_weight,
                    first_path_angle_weight=first_path_angle_weight,
                    first_path_angle_nlos_weight=first_path_angle_nlos_weight,
                    delay_spread_teacher_weight=delay_spread_teacher_weight,
                    delay_spread_bin_weights=delay_spread_bin_weights,
                    delay_spread_bin_classifier_weight=delay_spread_bin_classifier_weight,
                    delay_spread_bin_position_weight=delay_spread_bin_position_weight,
                    delay_spread_tail_classifier_weight=delay_spread_tail_classifier_weight,
                    interaction_count_classifier_weight=interaction_count_classifier_weight,
                    interaction_count_regression_weight=interaction_count_regression_weight,
                    reflection_count_classifier_weight=reflection_count_classifier_weight,
                    reflection_count_regression_weight=reflection_count_regression_weight,
                    reflection_count_nlos_weight=reflection_count_nlos_weight,
                    interaction_count_soft_labels=interaction_count_soft_labels,
                    first_path_power_bin_weights=first_path_power_bin_weights,
                    first_path_power_nlos_weight=first_path_power_nlos_weight,
                    first_path_power_gate_mode=first_path_power_gate_mode,
                    first_path_power_mode=first_path_power_mode,
                    first_path_power_use_internal_gate=first_path_power_use_internal_gate,
                    nlos_enhanced_power_loss=nlos_enhanced_power_loss,
                    prototype_warmup_epochs=1,
                    multipositive_distance_threshold=multipositive_distance_threshold,
                    multipositive_positive_mode=multipositive_positive_mode,
                    min_class_size_for_multipositive=min_class_size_for_multipositive,
                ),
            )
            print(
                f"epoch={epoch} loss={metrics['loss_total']:.4f} "
                f"contrastive={metrics['contrastive_loss']:.4f}"
            )
            break
        scheduler.step()


def run_real_pretrain(
    data_path: str,
    checkpoint_path: str | None,
    device: torch.device,
    epochs: int,
    max_steps_per_epoch: int | None,
    lr: float,
    weight_decay: float,
    batch_size: int,
    temperature: float,
    token_norm_mode: str,
    use_power_branch: bool,
    detach_delay_spread_features: bool,
    detach_first_path_delay_features: bool,
    use_delay_specific_encoder: bool,
    use_los_angle_context_encoder: bool,
    use_first_path_angle_context_encoder: bool,
    warmup_epochs: int,
    min_lr: float,
    prototype_weight: float,
    csi_to_text_weight: float,
    text_prototype_weight: float,
    text_mode: str,
    prototype_warmup_epochs: int,
    semantic_classifier_weight: float,
    semantic_classifier_class_weight: str,
    semantic_classifier_logit_adjustment: float,
    attribute_classifier_weight: float,
    attribute_classifier_fields: tuple[str, ...],
    attribute_classifier_class_weight: str,
    attribute_classifier_logit_adjustment: float,
    aux_regression_weight: float,
    aux_regression_targets: tuple[str, ...],
    k_factor_loss_weights: dict[str, float] | None,
    strong_k_bin_classifier_weight: float,
    strong_k_position_weight: float,
    strong_k_bin_weights: dict[str, float] | None,
    first_path_power_bin_classifier_weight: float,
    first_path_power_bin_position_weight: float,
    direct_power_weight: float,
    nlos_enhanced_power_loss: bool,
    first_path_power_delta_limit: float,
    first_path_power_mode: str,
    first_path_power_use_internal_gate: bool,
    delay_spread_weight: float,
    first_path_delay_weight: float,
    first_path_delay_raw_weight: float,
    first_path_delay_fused_raw_weight: float,
    first_path_delay_raw_beta_ns: float,
    first_path_delay_bin_classifier_weight: float,
    first_path_delay_bin_position_weight: float,
    first_path_delay_bin_consistency_weight: float,
    first_path_delay_bin_weights: dict[str, float] | None,
    estimated_pdp_tail_bin_weight: float,
    estimated_pdp_tail_labels: tuple[str, ...],
    estimated_pdp_tail_gate_mode: str,
    first_path_delay_tail_underestimate_weight: float,
    los_delay_weight: float,
    los_delay_nonnegative_weight: float,
    use_physics_calibration_loss: bool,
    los_delay_consistency_weight: float,
    los_angle_weight: float,
    first_path_angle_weight: float,
    first_path_angle_nlos_weight: float,
    delay_spread_raw_weight: float,
    delay_spread_raw_beta_ns: float,
    delay_spread_teacher_weight: float,
    delay_spread_bin_weights: dict[str, float] | None,
    delay_spread_bin_classifier_weight: float,
    delay_spread_bin_position_weight: float,
    delay_spread_tail_classifier_weight: float,
    interaction_count_classifier_weight: float,
    interaction_count_regression_weight: float,
    reflection_count_classifier_weight: float,
    reflection_count_regression_weight: float,
    reflection_count_nlos_weight: float,
    interaction_count_soft_labels: bool,
    interaction_count_fields: tuple[str, ...],
    first_path_power_bin_weights: dict[str, float],
    first_path_power_nlos_weight: float,
    first_path_power_gate_mode: str,
    multipositive_distance_threshold: float,
    multipositive_positive_mode: str,
    min_class_size_for_multipositive: int,
    min_class_size: int,
    semantic_key_mode: str,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]],
    filter_attribute_values: dict[str, tuple[str, ...]],
    limit_samples: int | None,
    limit_samples_by_attribute: str | None,
    limit_samples_per_attribute_value: int | None,
    max_delay_spread_ns: float | None,
    allow_prototype_mismatch_transfer: bool,
    freeze_csi: bool,
    freeze_text_prototypes: bool,
    output_dir: str,
    save_every: int,
) -> None:
    transfer_checkpoint = load_transfer_checkpoint(checkpoint_path, device)
    effective_filter_attribute_values = {
        **implied_attribute_value_filters(attribute_classifier_fields, attribute_remap),
        **filter_attribute_values,
    }
    loader, model, tokenizer, prototype_bank = build_real_components(
        data_path=data_path,
        device=device,
        batch_size=batch_size,
        temperature=temperature,
        token_norm_mode=token_norm_mode,
        use_power_branch=use_power_branch,
        first_path_power_mode=first_path_power_mode,
        first_path_power_use_internal_gate=first_path_power_use_internal_gate,
        use_delay_spread_head=delay_spread_weight > 0.0 or delay_spread_raw_weight > 0.0,
        detach_delay_spread_features=detach_delay_spread_features,
        detach_first_path_delay_features=detach_first_path_delay_features,
        use_delay_specific_encoder=use_delay_specific_encoder,
        use_los_angle_context_encoder=use_los_angle_context_encoder,
        use_first_path_angle_context_encoder=use_first_path_angle_context_encoder,
        min_class_size=min_class_size,
        semantic_key_mode=semantic_key_mode,
        attribute_fields=attribute_classifier_fields,
        attribute_remap=attribute_remap,
        filter_attribute_values=effective_filter_attribute_values,
        limit_samples=limit_samples,
        limit_samples_by_attribute=limit_samples_by_attribute,
        limit_samples_per_attribute_value=limit_samples_per_attribute_value,
        max_delay_spread_ns=max_delay_spread_ns,
        tokenizer_word2id=(
            transfer_checkpoint.get("tokenizer_word2id")
            if transfer_checkpoint is not None
            else None
        ),
    )
    model.first_path_power_delta_limit = float(first_path_power_delta_limit)
    if transfer_checkpoint is not None:
        try:
            assert_checkpoint_prototype_compatibility(
                transfer_checkpoint,
                prototype_bank["keys"],
                expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
                context="transfer checkpoint",
            )
        except ValueError:
            if not allow_prototype_mismatch_transfer:
                raise
            print(
                "allowing prototype-mismatch transfer: compatible checkpoint weights "
                "will be loaded and current prototypes will be reinitialized from text."
            )
    if transfer_checkpoint is not None:
        load_model_state_compatible(model, transfer_checkpoint)
    if not checkpoint_has_compatible_prototypes(model, transfer_checkpoint, prototype_bank["keys"]):
        initialize_prototypes_from_canonical_text(
            model,
            prototype_bank["token_ids"],
            prototype_bank["token_mask"],
        )
    if freeze_csi:
        freeze_module(model.csi)
    if freeze_text_prototypes:
        freeze_text_and_prototypes(model)
    optimizer_parameters = trainable_parameters(model)
    if not optimizer_parameters:
        raise ValueError("No trainable parameters remain after applying freeze options.")
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=lr, weight_decay=weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        total_epochs=epochs,
        warmup_epochs=min(warmup_epochs, epochs),
        min_lr_scale=min_lr / lr,
    )
    cfg = TrainConfig(
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        csi_to_text_weight=csi_to_text_weight,
        prototype_weight=prototype_weight,
        text_prototype_weight=text_prototype_weight,
        text_mode=text_mode,
        prototype_warmup_epochs=prototype_warmup_epochs,
        semantic_classifier_weight=semantic_classifier_weight,
        semantic_classifier_class_weight=semantic_classifier_class_weight,
        semantic_classifier_logit_adjustment=semantic_classifier_logit_adjustment,
        attribute_classifier_weight=attribute_classifier_weight,
        attribute_classifier_class_weight=attribute_classifier_class_weight,
        attribute_classifier_logit_adjustment=attribute_classifier_logit_adjustment,
        aux_regression_weight=aux_regression_weight,
        aux_regression_indices=aux_regression_indices(aux_regression_targets),
        k_factor_loss_weights=k_factor_loss_weights,
        strong_k_bin_classifier_weight=strong_k_bin_classifier_weight,
        strong_k_position_weight=strong_k_position_weight,
        strong_k_bin_weights=strong_k_bin_weights,
        first_path_power_bin_classifier_weight=first_path_power_bin_classifier_weight,
        first_path_power_bin_position_weight=first_path_power_bin_position_weight,
        direct_power_weight=direct_power_weight,
        delay_spread_weight=delay_spread_weight,
        delay_spread_raw_weight=delay_spread_raw_weight,
        delay_spread_raw_beta_ns=delay_spread_raw_beta_ns,
        first_path_delay_weight=first_path_delay_weight,
        first_path_delay_raw_weight=first_path_delay_raw_weight,
        first_path_delay_fused_raw_weight=first_path_delay_fused_raw_weight,
        first_path_delay_raw_beta_ns=first_path_delay_raw_beta_ns,
        first_path_delay_bin_classifier_weight=first_path_delay_bin_classifier_weight,
        first_path_delay_bin_position_weight=first_path_delay_bin_position_weight,
        first_path_delay_bin_consistency_weight=first_path_delay_bin_consistency_weight,
        first_path_delay_bin_weights=first_path_delay_bin_weights,
        estimated_pdp_tail_bin_weight=estimated_pdp_tail_bin_weight,
        estimated_pdp_tail_labels=estimated_pdp_tail_labels,
        estimated_pdp_tail_gate_mode=estimated_pdp_tail_gate_mode,
        first_path_delay_tail_underestimate_weight=first_path_delay_tail_underestimate_weight,
        los_delay_weight=los_delay_weight,
        los_delay_nonnegative_weight=los_delay_nonnegative_weight,
        use_physics_calibration_loss=use_physics_calibration_loss,
        los_delay_consistency_weight=los_delay_consistency_weight,
        los_angle_weight=los_angle_weight,
        first_path_angle_weight=first_path_angle_weight,
        first_path_angle_nlos_weight=first_path_angle_nlos_weight,
        delay_spread_teacher_weight=delay_spread_teacher_weight,
        delay_spread_bin_weights=delay_spread_bin_weights,
        delay_spread_bin_classifier_weight=delay_spread_bin_classifier_weight,
        delay_spread_bin_position_weight=delay_spread_bin_position_weight,
        delay_spread_tail_classifier_weight=delay_spread_tail_classifier_weight,
        interaction_count_classifier_weight=interaction_count_classifier_weight,
        interaction_count_regression_weight=interaction_count_regression_weight,
        reflection_count_classifier_weight=reflection_count_classifier_weight,
        reflection_count_regression_weight=reflection_count_regression_weight,
        reflection_count_nlos_weight=reflection_count_nlos_weight,
        interaction_count_soft_labels=interaction_count_soft_labels,
        first_path_power_bin_weights=first_path_power_bin_weights,
        first_path_power_nlos_weight=first_path_power_nlos_weight,
        first_path_power_gate_mode=first_path_power_gate_mode,
        first_path_power_mode=first_path_power_mode,
        first_path_power_use_internal_gate=first_path_power_use_internal_gate,
        nlos_enhanced_power_loss=nlos_enhanced_power_loss,
        freeze_csi=freeze_csi,
        freeze_text_prototypes=freeze_text_prototypes,
        multipositive_distance_threshold=multipositive_distance_threshold,
        multipositive_positive_mode=multipositive_positive_mode,
        min_class_size_for_multipositive=min_class_size_for_multipositive,
    )
    trainer = Trainer(
        model,
        optimizer,
        device,
        prototype_token_ids=prototype_bank["token_ids"],
        prototype_token_mask=prototype_bank["token_mask"],
        prototype_label_map=prototype_bank["label_map"],
        prototype_class_counts=prototype_bank["class_counts"],
        attribute_label_maps=prototype_bank["attribute_label_maps"],
        attribute_class_counts=prototype_bank["attribute_class_counts"],
        attribute_remap=attribute_remap,
    )
    print(f"training on {data_path}")
    print(f"checkpoint={checkpoint_path}")
    print(
        f"device={device} epochs={epochs} batch_size={batch_size} "
        f"temperature={temperature} token_norm_mode={token_norm_mode} "
        f"use_power_branch={use_power_branch} "
        f"detach_delay_spread_features={detach_delay_spread_features} "
        f"detach_first_path_delay_features={detach_first_path_delay_features} "
        f"use_delay_specific_encoder={use_delay_specific_encoder} "
        f"use_los_angle_context_encoder={use_los_angle_context_encoder} "
        f"use_first_path_angle_context_encoder={use_first_path_angle_context_encoder} "
        f"warmup_epochs={warmup_epochs} min_lr={min_lr}"
    )
    print(
        f"semantic_prototypes={len(prototype_bank['keys'])} "
        f"csi_to_text_weight={csi_to_text_weight} "
        f"prototype_weight={prototype_weight} text_prototype_weight={text_prototype_weight} "
        f"text_mode={text_mode} "
        f"prototype_warmup_epochs={prototype_warmup_epochs} "
        f"semantic_classifier_weight={semantic_classifier_weight} "
        f"semantic_classifier_class_weight={semantic_classifier_class_weight} "
        f"semantic_classifier_logit_adjustment={semantic_classifier_logit_adjustment} "
        f"attribute_classifier_weight={attribute_classifier_weight} "
        f"attribute_classifier_fields={','.join(attribute_classifier_fields)} "
        f"attribute_classifier_class_weight={attribute_classifier_class_weight} "
        f"attribute_classifier_logit_adjustment={attribute_classifier_logit_adjustment} "
        f"attribute_remap={format_attribute_remap(attribute_remap)} "
        f"aux_regression_weight={aux_regression_weight} "
        f"aux_regression_targets={','.join(aux_regression_targets)} "
        f"k_factor_loss_weights={format_k_factor_loss_weights(k_factor_loss_weights)} "
        f"strong_k_bin_classifier_weight={strong_k_bin_classifier_weight} "
        f"strong_k_position_weight={strong_k_position_weight} "
        f"strong_k_bin_weights={format_strong_k_bin_weights(strong_k_bin_weights)} "
        f"strong_k_bin_label_order={','.join(K_FACTOR_STRONG_BIN_LABELS)} "
        f"first_path_power_bin_classifier_weight={first_path_power_bin_classifier_weight} "
        f"first_path_power_bin_position_weight={first_path_power_bin_position_weight} "
        f"first_path_power_bin_label_order={','.join(FIRST_POWER_DBW_BIN_LABELS)} "
        f"direct_power_weight={direct_power_weight} "
        f"first_path_power_gate_mode={first_path_power_gate_mode} "
        f"first_path_power_mode={first_path_power_mode} "
        f"first_path_power_use_internal_gate={first_path_power_use_internal_gate} "
        f"nlos_enhanced_power_loss={nlos_enhanced_power_loss} "
        f"first_path_power_delta_limit={first_path_power_delta_limit} "
        f"delay_spread_weight={delay_spread_weight} "
        f"delay_spread_raw_weight={delay_spread_raw_weight} "
        f"delay_spread_raw_beta_ns={delay_spread_raw_beta_ns} "
        f"first_path_delay_weight={first_path_delay_weight} "
        f"first_path_delay_raw_weight={first_path_delay_raw_weight} "
        f"first_path_delay_fused_raw_weight={first_path_delay_fused_raw_weight} "
        f"first_path_delay_raw_beta_ns={first_path_delay_raw_beta_ns} "
        f"first_path_delay_bin_classifier_weight={first_path_delay_bin_classifier_weight} "
        f"first_path_delay_bin_position_weight={first_path_delay_bin_position_weight} "
        f"first_path_delay_bin_consistency_weight={first_path_delay_bin_consistency_weight} "
        f"first_path_delay_bin_weights={format_first_path_delay_bin_weights(first_path_delay_bin_weights)} "
        f"estimated_pdp_tail_bin_weight={estimated_pdp_tail_bin_weight} "
        f"estimated_pdp_tail_labels={','.join(estimated_pdp_tail_labels)} "
        f"estimated_pdp_tail_gate_mode={estimated_pdp_tail_gate_mode} "
        f"first_path_delay_tail_underestimate_weight={first_path_delay_tail_underestimate_weight} "
        f"first_path_delay_bin_label_order={','.join(FIRST_PATH_DELAY_BIN_LABELS)} "
        f"los_delay_weight={los_delay_weight} "
        f"los_delay_nonnegative_weight={los_delay_nonnegative_weight} "
        f"use_physics_calibration_loss={use_physics_calibration_loss} "
        f"los_delay_consistency_weight={los_delay_consistency_weight} "
        f"los_angle_weight={los_angle_weight} "
        f"first_path_angle_weight={first_path_angle_weight} "
        f"first_path_angle_nlos_weight={first_path_angle_nlos_weight} "
        f"delay_spread_teacher_weight={delay_spread_teacher_weight} "
        f"delay_spread_bin_classifier_weight={delay_spread_bin_classifier_weight} "
        f"delay_spread_bin_position_weight={delay_spread_bin_position_weight} "
        f"delay_spread_tail_classifier_weight={delay_spread_tail_classifier_weight} "
        f"reflection_count_classifier_weight={reflection_count_classifier_weight} "
        f"reflection_count_regression_weight={reflection_count_regression_weight} "
        f"reflection_count_nlos_weight={reflection_count_nlos_weight} "
        f"interaction_count_soft_labels={interaction_count_soft_labels} "
        f"interaction_count_fields={format_interaction_count_fields(interaction_count_fields)} "
        f"interaction_count_classifier_weight={interaction_count_classifier_weight} "
        f"interaction_count_regression_weight={interaction_count_regression_weight} "
        f"reflection_count_bin_label_order={','.join(REFLECTION_COUNT_BIN_LABELS)} "
        f"delay_spread_bin_weights={format_delay_spread_bin_weights(delay_spread_bin_weights)} "
        f"delay_spread_bin_label_order={','.join(DELAY_SPREAD_BIN_LABELS)} "
        f"delay_spread_tail_label_order={','.join(DELAY_SPREAD_TAIL_LABELS)} "
        f"first_path_power_bin_weights={format_first_path_power_bin_weights(first_path_power_bin_weights)} "
        f"first_path_power_nlos_weight={first_path_power_nlos_weight} "
        f"multipositive_distance_threshold={multipositive_distance_threshold} "
        f"multipositive_positive_mode={multipositive_positive_mode} "
        f"min_class_size_for_multipositive={min_class_size_for_multipositive} "
        f"min_class_size={min_class_size} semantic_key_mode={semantic_key_mode} "
        f"filter_attribute_values={format_attribute_value_filters(effective_filter_attribute_values)} "
        f"limit_samples={limit_samples} "
        f"limit_samples_by_attribute={limit_samples_by_attribute} "
        f"limit_samples_per_attribute_value={limit_samples_per_attribute_value} "
        f"max_delay_spread_ns={max_delay_spread_ns} "
        f"freeze_csi={freeze_csi} freeze_text_prototypes={freeze_text_prototypes} "
        f"trainable_parameters={count_trainable_parameters(model)}"
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "train_log.jsonl"

    for epoch in range(1, epochs + 1):
        epoch_metrics = []
        for step, batch in enumerate(loader, start=1):
            metrics = trainer.train_step(batch, epoch=epoch, cfg=cfg)
            epoch_metrics.append(metrics)
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
        scheduler.step()
        if not epoch_metrics:
            raise SystemExit("No batches were produced from the dataset.")

        mean_total = sum(m["loss_total"] for m in epoch_metrics) / len(epoch_metrics)
        mean_contrastive = sum(m["contrastive_loss"] for m in epoch_metrics) / len(epoch_metrics)
        mean_csi_to_text = sum(m["loss_csi_to_text"] for m in epoch_metrics) / len(epoch_metrics)
        mean_csi_to_prototype = sum(m["loss_csi_to_prototype"] for m in epoch_metrics) / len(epoch_metrics)
        mean_text_to_prototype = sum(m["loss_text_to_prototype"] for m in epoch_metrics) / len(epoch_metrics)
        mean_semantic_classifier = sum(m.get("loss_semantic_classifier", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_semantic_accuracy = sum(m.get("accuracy_semantic_classifier", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_semantic_logit_std = sum(m.get("logit_std_semantic_classifier", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_semantic_logit_mean = sum(m.get("semantic_logit_mean", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_semantic_logit_max_mean = sum(m.get("semantic_logit_max_mean", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_semantic_pred_majority_fraction = sum(
            m.get("batch_semantic_prediction_majority_fraction", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_semantic_pred_unique_classes = sum(
            m.get("batch_semantic_prediction_unique_classes", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_semantic_head_bias_mean = sum(
            m.get("semantic_head_bias_mean", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_semantic_head_bias_std = sum(
            m.get("semantic_head_bias_std", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_semantic_head_bias_argmax = sum(
            m.get("semantic_head_bias_argmax", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_attribute_classifier = sum(m.get("loss_attribute_classifier", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_attribute_accuracy = sum(m.get("accuracy_attribute_classifier", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_attribute_logit_std = sum(m.get("logit_std_attribute_classifier", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_grad_csi_encoder = sum(m.get("grad_norm_csi_encoder", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_grad_semantic_classifier = sum(m.get("grad_norm_semantic_classifier", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_grad_attribute_classifiers = sum(m.get("grad_norm_attribute_classifiers", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_csi_feature_raw_std = sum(m.get("csi_feature_raw_std", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_csi_feature_normalized_std = sum(m.get("csi_feature_normalized_std", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_batch_label_majority_fraction = sum(
            m.get("batch_label_majority_fraction", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_batch_label_unique_classes = sum(
            m.get("batch_label_unique_classes", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_aux_regression = sum(m.get("loss_aux_regression", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_strong_k_bin_classifier = sum(
            m.get("loss_strong_k_bin_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_strong_k_bin_accuracy = sum(
            m.get("accuracy_strong_k_bin_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_strong_k_position = sum(
            m.get("loss_strong_k_position", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_strong_k_position_mae = sum(
            m.get("strong_k_position_mae", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_strong_k_bin_loss_denominator = sum(
            m.get("strong_k_bin_loss_denominator", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        last_strong_k_bin_class_weights = epoch_metrics[-1].get("strong_k_bin_class_weights", "")
        strong_k_bin_target_histogram = sum_histogram_metric(
            epoch_metrics,
            "strong_k_bin_target_histogram",
            len(K_FACTOR_STRONG_BIN_LABELS),
        )
        strong_k_bin_prediction_histogram = sum_histogram_metric(
            epoch_metrics,
            "strong_k_bin_prediction_histogram",
            len(K_FACTOR_STRONG_BIN_LABELS),
        )
        strong_k_bin_target_distribution = format_histogram_counts(
            K_FACTOR_STRONG_BIN_LABELS,
            strong_k_bin_target_histogram,
        )
        strong_k_bin_prediction_distribution = format_histogram_counts(
            K_FACTOR_STRONG_BIN_LABELS,
            strong_k_bin_prediction_histogram,
        )
        mean_first_path_power_bin_classifier = sum(
            m.get("loss_first_path_power_bin_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_power_bin_accuracy = sum(
            m.get("accuracy_first_path_power_bin_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_power_bin_position = sum(
            m.get("loss_first_path_power_bin_position", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_power_bin_position_mae = sum(
            m.get("first_path_power_bin_position_mae", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_power_bin_loss_denominator = sum(
            m.get("first_path_power_bin_loss_denominator", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_bin_classifier = sum(
            m.get("loss_delay_spread_bin_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_bin_accuracy = sum(
            m.get("accuracy_delay_spread_bin_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_bin_position = sum(
            m.get("loss_delay_spread_bin_position", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_bin_position_mae = sum(
            m.get("delay_spread_bin_position_mae", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_bin_loss_denominator = sum(
            m.get("delay_spread_bin_loss_denominator", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread = sum(
            m.get("loss_delay_spread", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_raw = sum(
            m.get("loss_delay_spread_raw", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_raw_mae_ns = sum(
            m.get("delay_spread_raw_mae_ns", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_normalized_mae_ns = sum(
            m.get("delay_spread_normalized_mae_ns", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_tail_classifier = sum(
            m.get("loss_delay_spread_tail_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_tail_accuracy = sum(
            m.get("delay_spread_tail_accuracy", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_tail_recall = sum(
            m.get("delay_spread_tail_recall", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_delay_spread_tail_false_positive = sum(
            m.get("delay_spread_tail_false_positive", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        last_delay_spread_tail_positive_fraction = epoch_metrics[-1].get(
            "delay_spread_tail_positive_fraction",
            "",
        )
        last_delay_spread_tail_prediction_fraction = epoch_metrics[-1].get(
            "delay_spread_tail_prediction_fraction",
            "",
        )
        last_delay_spread_bin_class_weights = epoch_metrics[-1].get("delay_spread_bin_class_weights", "")
        delay_spread_bin_target_histogram = sum_histogram_metric(
            epoch_metrics,
            "delay_spread_bin_target_histogram",
            len(DELAY_SPREAD_BIN_LABELS),
        )
        delay_spread_bin_prediction_histogram = sum_histogram_metric(
            epoch_metrics,
            "delay_spread_bin_prediction_histogram",
            len(DELAY_SPREAD_BIN_LABELS),
        )
        delay_spread_bin_target_distribution = format_histogram_counts(
            DELAY_SPREAD_BIN_LABELS,
            delay_spread_bin_target_histogram,
        )
        delay_spread_bin_prediction_distribution = format_histogram_counts(
            DELAY_SPREAD_BIN_LABELS,
            delay_spread_bin_prediction_histogram,
        )
        first_path_power_bin_target_histogram = sum_histogram_metric(
            epoch_metrics,
            "first_path_power_bin_target_histogram",
            len(FIRST_POWER_DBW_BIN_LABELS),
        )
        first_path_power_bin_prediction_histogram = sum_histogram_metric(
            epoch_metrics,
            "first_path_power_bin_prediction_histogram",
            len(FIRST_POWER_DBW_BIN_LABELS),
        )
        first_path_power_bin_target_distribution = format_histogram_counts(
            FIRST_POWER_DBW_BIN_LABELS,
            first_path_power_bin_target_histogram,
        )
        first_path_power_bin_prediction_distribution = format_histogram_counts(
            FIRST_POWER_DBW_BIN_LABELS,
            first_path_power_bin_prediction_histogram,
        )
        mean_first_path_delay_bin_classifier = sum(
            m.get("loss_first_path_delay_bin_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_bin_accuracy = sum(
            m.get("accuracy_first_path_delay_bin_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_estimated_pdp_tail_bin_classifier = sum(
            m.get("loss_estimated_pdp_tail_bin_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_estimated_pdp_tail_bin_accuracy = sum(
            m.get("accuracy_estimated_pdp_tail_bin_classifier", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_estimated_pdp_tail_gate_fraction = sum(
            m.get("estimated_pdp_tail_gate_fraction", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_estimated_pdp_tail_target_fraction = sum(
            m.get("estimated_pdp_tail_target_fraction", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_estimated_pdp_tail_argmax_fraction = sum(
            m.get("estimated_pdp_tail_argmax_fraction", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_tail_underestimate = sum(
            m.get("loss_first_path_delay_tail_underestimate", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_tail_underestimate_mae_ns = sum(
            m.get("first_path_delay_tail_underestimate_mae_ns", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_tail_underestimate_mean_ns = sum(
            m.get("first_path_delay_tail_underestimate_mean_ns", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_tail_underestimate_fraction = sum(
            m.get("first_path_delay_tail_underestimate_fraction", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_bin_position = sum(
            m.get("loss_first_path_delay_bin_position", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_bin_position_mae = sum(
            m.get("first_path_delay_bin_position_mae", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_bin_consistency = sum(
            m.get("loss_first_path_delay_bin_consistency", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_raw = sum(
            m.get("loss_first_path_delay_raw", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_fused_raw = sum(
            m.get("loss_first_path_delay_fused_raw", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_raw_mae_ns = sum(
            m.get("first_path_delay_raw_mae_ns", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_fused_raw_mae_ns = sum(
            m.get("first_path_delay_fused_raw_mae_ns", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_bin_consistency_violation_ns = sum(
            m.get("first_path_delay_bin_consistency_violation_ns", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        max_first_path_delay_bin_consistency_violation_ns = max(
            m.get("first_path_delay_bin_consistency_max_violation_ns", 0.0)
            for m in epoch_metrics
        )
        mean_first_path_delay_bin_loss_denominator = sum(
            m.get("first_path_delay_bin_loss_denominator", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        last_first_path_delay_bin_class_weights = epoch_metrics[-1].get(
            "first_path_delay_bin_class_weights",
            "",
        )
        first_path_delay_bin_target_histogram = sum_histogram_metric(
            epoch_metrics,
            "first_path_delay_bin_target_histogram",
            len(FIRST_PATH_DELAY_BIN_LABELS),
        )
        first_path_delay_bin_prediction_histogram = sum_histogram_metric(
            epoch_metrics,
            "first_path_delay_bin_prediction_histogram",
            len(FIRST_PATH_DELAY_BIN_LABELS),
        )
        first_path_delay_bin_target_distribution = format_histogram_counts(
            FIRST_PATH_DELAY_BIN_LABELS,
            first_path_delay_bin_target_histogram,
        )
        first_path_delay_bin_prediction_distribution = format_histogram_counts(
            FIRST_PATH_DELAY_BIN_LABELS,
            first_path_delay_bin_prediction_histogram,
        )
        mean_direct_power = sum(m.get("loss_direct_power", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        nlos_power_enhanced_mae_values = [
            m["nlos_first_path_power_enhanced_mae_db"]
            for m in epoch_metrics
            if "nlos_first_path_power_enhanced_mae_db" in m
        ]
        mean_nlos_first_path_power_enhanced_mae_db = (
            sum(nlos_power_enhanced_mae_values) / len(nlos_power_enhanced_mae_values)
            if nlos_power_enhanced_mae_values
            else 0.0
        )
        nlos_power_base_mae_values = [
            m["nlos_first_path_power_base_mae_db"]
            for m in epoch_metrics
            if "nlos_first_path_power_base_mae_db" in m
        ]
        mean_nlos_first_path_power_base_mae_db = (
            sum(nlos_power_base_mae_values) / len(nlos_power_base_mae_values)
            if nlos_power_base_mae_values
            else 0.0
        )
        mean_first_path_power_delta_saturation = sum(
            m.get("first_path_power_delta_saturation_fraction", 0.0)
            for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay = sum(m.get("loss_first_path_delay", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_los_delay = sum(m.get("loss_los_delay", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_los_delay_nonnegative = sum(
            m.get("loss_los_delay_nonnegative", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_los_delay_consistency = sum(
            m.get("loss_los_delay_consistency", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        los_delay_consistency_mae_values = [
            m["los_delay_consistency_mae_ns"]
            for m in epoch_metrics
            if "los_delay_consistency_mae_ns" in m
        ]
        mean_los_delay_consistency_mae_ns = (
            sum(los_delay_consistency_mae_values) / len(los_delay_consistency_mae_values)
            if los_delay_consistency_mae_values
            else 0.0
        )
        mean_los_angle = sum(m.get("loss_los_angle", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_first_path_angle = sum(m.get("loss_first_path_angle", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_first_path_angle_nlos = sum(
            m.get("loss_first_path_angle_nlos", 0.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        los_angle_mae_values = [
            m["los_angle_mae_deg"]
            for m in epoch_metrics
            if "los_angle_mae_deg" in m
        ]
        mean_los_angle_mae_deg = (
            sum(los_angle_mae_values) / len(los_angle_mae_values)
            if los_angle_mae_values
            else 0.0
        )
        first_path_angle_mae_values = [
            m["first_path_angle_mae_deg"]
            for m in epoch_metrics
            if "first_path_angle_mae_deg" in m
        ]
        mean_first_path_angle_mae_deg = (
            sum(first_path_angle_mae_values) / len(first_path_angle_mae_values)
            if first_path_angle_mae_values
            else 0.0
        )
        first_path_angle_nlos_mae_values = [
            m["first_path_angle_nlos_mae_deg"]
            for m in epoch_metrics
            if "first_path_angle_nlos_mae_deg" in m
        ]
        mean_first_path_angle_nlos_mae_deg = (
            sum(first_path_angle_nlos_mae_values) / len(first_path_angle_nlos_mae_values)
            if first_path_angle_nlos_mae_values
            else 0.0
        )
        mean_k_factor_sample_weight = sum(
            m.get("k_factor_sample_weight_mean", 1.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_power_sample_weight = sum(
            m.get("first_path_power_sample_weight_mean", 1.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_first_path_delay_sample_weight = sum(
            m.get("first_path_delay_sample_weight_mean", 1.0) for m in epoch_metrics
        ) / len(epoch_metrics)
        mean_positive_count = sum(m.get("multipositive_positive_count_mean", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_logit_scale = sum(m["logit_scale"] for m in epoch_metrics) / len(epoch_metrics)
        mean_prototype_warmup_active = sum(m.get("prototype_warmup_active", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        last_batch_label_histogram = epoch_metrics[-1].get("batch_label_histogram", "")
        last_batch_semantic_argmax_histogram = epoch_metrics[-1].get("batch_semantic_argmax_histogram", "")
        last_batch_semantic_head_bias_values = epoch_metrics[-1].get("semantic_head_bias_values", "")
        attribute_debug = (
            f" sem_logit_std={mean_semantic_logit_std:.4f} "
            f"sem_logit_mean={mean_semantic_logit_mean:.4f} "
            f"sem_logit_max={mean_semantic_logit_max_mean:.4f} "
            f"sem_grad={mean_grad_semantic_classifier:.4e} "
            f"sem_pred_maj={mean_semantic_pred_majority_fraction:.4f} "
            f"sem_pred_u={mean_semantic_pred_unique_classes:.2f} "
            f"sem_bias_mean={mean_semantic_head_bias_mean:.4f} "
            f"sem_bias_std={mean_semantic_head_bias_std:.4f} "
            f"sem_bias_argmax={mean_semantic_head_bias_argmax:.2f} "
            f" attr_logit_std={mean_attribute_logit_std:.4f} "
            f"raw_std={mean_csi_feature_raw_std:.4f} "
            f"norm_std={mean_csi_feature_normalized_std:.4f} "
            f"label_maj={mean_batch_label_majority_fraction:.4f} "
            f"label_u={mean_batch_label_unique_classes:.2f} "
            f"grad_csi={mean_grad_csi_encoder:.4e} "
            f"grad_attr={mean_grad_attribute_classifiers:.4e}"
            if semantic_classifier_weight > 0 or attribute_classifier_weight > 0
            else ""
        )
        print(
            f"epoch={epoch:03d}/{epochs} steps={len(epoch_metrics)} "
            f"loss={mean_total:.4f} c2t={mean_csi_to_text:.4f} "
            f"sem_acc={mean_semantic_accuracy:.4f} attr_acc={mean_attribute_accuracy:.4f} "
            f"k_acc={mean_strong_k_bin_accuracy:.4f} k_pos_mae={mean_strong_k_position_mae:.4f} "
            f"delay_acc={mean_delay_spread_bin_accuracy:.4f} "
            f"delay_pos_mae={mean_delay_spread_bin_position_mae:.4f} "
            f"delay_norm_loss={mean_delay_spread:.4f} "
            f"delay_norm_mae={mean_delay_spread_normalized_mae_ns:.2f}ns "
            f"delay_raw_mae={mean_delay_spread_raw_mae_ns:.2f}ns "
            f"delay_raw_loss={mean_delay_spread_raw:.4f} "
            f"first_delay_acc={mean_first_path_delay_bin_accuracy:.4f} "
            f"pdp_tail_acc={mean_estimated_pdp_tail_bin_accuracy:.4f} "
            f"pdp_tail_gate={mean_estimated_pdp_tail_gate_fraction:.4f} "
            f"tail_under={mean_first_path_delay_tail_underestimate_mean_ns:.2f}ns "
            f"first_delay_pos_mae={mean_first_path_delay_bin_position_mae:.4f} "
            f"first_delay_raw_mae={mean_first_path_delay_raw_mae_ns:.2f}ns "
            f"first_delay_fused_mae={mean_first_path_delay_fused_raw_mae_ns:.2f}ns "
            f"first_delay_bin_violate={mean_first_path_delay_bin_consistency_violation_ns:.2f}ns "
            f"los_nonneg={mean_los_delay_nonnegative:.4f} "
            f"los_cons={mean_los_delay_consistency:.4f} "
            f"los_cons_mae={mean_los_delay_consistency_mae_ns:.2f}ns "
            f"los_angle_mae={mean_los_angle_mae_deg:.2f}deg "
            f"first_angle_mae={mean_first_path_angle_mae_deg:.2f}deg "
            f"first_angle_nlos_mae={mean_first_path_angle_nlos_mae_deg:.2f}deg "
            f"nlos_power_base_mae={mean_nlos_first_path_power_base_mae_db:.2f}dB "
            f"nlos_power_enh_mae={mean_nlos_first_path_power_enhanced_mae_db:.2f}dB "
            f"power_delta_sat={mean_first_path_power_delta_saturation:.4f} "
            f"aux={mean_aux_regression:.4f} grad_csi={mean_grad_csi_encoder:.2e} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "steps": len(epoch_metrics),
                        "loss_total": mean_total,
                        "contrastive_loss": mean_contrastive,
                        "checkpoint": checkpoint_path,
                        "csi_to_text_weight": csi_to_text_weight,
                        "loss_csi_to_text": mean_csi_to_text,
                        "loss_csi_to_prototype": mean_csi_to_prototype,
                        "loss_text_to_prototype": mean_text_to_prototype,
                        "loss_semantic_classifier": mean_semantic_classifier,
                        "accuracy_semantic_classifier": mean_semantic_accuracy,
                        "logit_std_semantic_classifier": mean_semantic_logit_std,
                        "semantic_logit_mean": mean_semantic_logit_mean,
                        "semantic_logit_max_mean": mean_semantic_logit_max_mean,
                        "batch_semantic_prediction_majority_fraction": mean_semantic_pred_majority_fraction,
                        "batch_semantic_prediction_unique_classes": mean_semantic_pred_unique_classes,
                        "semantic_head_bias_mean": mean_semantic_head_bias_mean,
                        "semantic_head_bias_std": mean_semantic_head_bias_std,
                        "semantic_head_bias_argmax": mean_semantic_head_bias_argmax,
                        "loss_attribute_classifier": mean_attribute_classifier,
                        "accuracy_attribute_classifier": mean_attribute_accuracy,
                        "logit_std_attribute_classifier": mean_attribute_logit_std,
                        "csi_feature_raw_std": mean_csi_feature_raw_std,
                        "csi_feature_normalized_std": mean_csi_feature_normalized_std,
                        "batch_label_majority_fraction": mean_batch_label_majority_fraction,
                        "batch_label_unique_classes": mean_batch_label_unique_classes,
                        "last_batch_label_histogram": last_batch_label_histogram,
                        "last_batch_semantic_argmax_histogram": last_batch_semantic_argmax_histogram,
                        "last_batch_semantic_head_bias_values": last_batch_semantic_head_bias_values,
                        "grad_norm_csi_encoder": mean_grad_csi_encoder,
                        "grad_norm_semantic_classifier": mean_grad_semantic_classifier,
                        "grad_norm_attribute_classifiers": mean_grad_attribute_classifiers,
                        "loss_aux_regression": mean_aux_regression,
                        "loss_strong_k_bin_classifier": mean_strong_k_bin_classifier,
                        "accuracy_strong_k_bin_classifier": mean_strong_k_bin_accuracy,
                        "loss_strong_k_position": mean_strong_k_position,
                        "strong_k_position_mae": mean_strong_k_position_mae,
                        "strong_k_bin_loss_denominator": mean_strong_k_bin_loss_denominator,
                        "strong_k_bin_class_weights": last_strong_k_bin_class_weights,
                        "strong_k_bin_label_order": list(K_FACTOR_STRONG_BIN_LABELS),
                        "strong_k_bin_target_histogram": strong_k_bin_target_histogram,
                        "strong_k_bin_prediction_histogram": strong_k_bin_prediction_histogram,
                        "loss_first_path_power_bin_classifier": mean_first_path_power_bin_classifier,
                        "accuracy_first_path_power_bin_classifier": mean_first_path_power_bin_accuracy,
                        "loss_first_path_power_bin_position": mean_first_path_power_bin_position,
                        "first_path_power_bin_position_mae": mean_first_path_power_bin_position_mae,
                        "first_path_power_bin_loss_denominator": mean_first_path_power_bin_loss_denominator,
                        "first_path_power_bin_label_order": list(FIRST_POWER_DBW_BIN_LABELS),
                        "first_path_power_bin_target_histogram": first_path_power_bin_target_histogram,
                        "first_path_power_bin_prediction_histogram": first_path_power_bin_prediction_histogram,
                        "loss_first_path_delay_bin_classifier": mean_first_path_delay_bin_classifier,
                        "accuracy_first_path_delay_bin_classifier": mean_first_path_delay_bin_accuracy,
                        "loss_estimated_pdp_tail_bin_classifier": mean_estimated_pdp_tail_bin_classifier,
                        "accuracy_estimated_pdp_tail_bin_classifier": mean_estimated_pdp_tail_bin_accuracy,
                        "estimated_pdp_tail_gate_fraction": mean_estimated_pdp_tail_gate_fraction,
                        "estimated_pdp_tail_target_fraction": mean_estimated_pdp_tail_target_fraction,
                        "estimated_pdp_tail_argmax_fraction": mean_estimated_pdp_tail_argmax_fraction,
                        "loss_first_path_delay_tail_underestimate": mean_first_path_delay_tail_underestimate,
                        "first_path_delay_tail_underestimate_mae_ns": mean_first_path_delay_tail_underestimate_mae_ns,
                        "first_path_delay_tail_underestimate_mean_ns": mean_first_path_delay_tail_underestimate_mean_ns,
                        "first_path_delay_tail_underestimate_fraction": mean_first_path_delay_tail_underestimate_fraction,
                        "loss_first_path_delay_bin_position": mean_first_path_delay_bin_position,
                        "first_path_delay_bin_position_mae": mean_first_path_delay_bin_position_mae,
                        "loss_first_path_delay_raw": mean_first_path_delay_raw,
                        "loss_first_path_delay_fused_raw": mean_first_path_delay_fused_raw,
                        "first_path_delay_raw_mae_ns": mean_first_path_delay_raw_mae_ns,
                        "first_path_delay_fused_raw_mae_ns": mean_first_path_delay_fused_raw_mae_ns,
                        "loss_first_path_delay_bin_consistency": mean_first_path_delay_bin_consistency,
                        "first_path_delay_bin_consistency_violation_ns": mean_first_path_delay_bin_consistency_violation_ns,
                        "first_path_delay_bin_consistency_max_violation_ns": max_first_path_delay_bin_consistency_violation_ns,
                        "first_path_delay_bin_loss_denominator": mean_first_path_delay_bin_loss_denominator,
                        "first_path_delay_bin_class_weights": last_first_path_delay_bin_class_weights,
                        "first_path_delay_bin_label_order": list(FIRST_PATH_DELAY_BIN_LABELS),
                        "first_path_delay_bin_target_histogram": first_path_delay_bin_target_histogram,
                        "first_path_delay_bin_prediction_histogram": first_path_delay_bin_prediction_histogram,
                        "first_path_delay_bin_target_distribution": first_path_delay_bin_target_distribution,
                        "first_path_delay_bin_prediction_distribution": first_path_delay_bin_prediction_distribution,
                        "loss_delay_spread_bin_classifier": mean_delay_spread_bin_classifier,
                        "accuracy_delay_spread_bin_classifier": mean_delay_spread_bin_accuracy,
                        "loss_delay_spread_bin_position": mean_delay_spread_bin_position,
                        "delay_spread_bin_position_mae": mean_delay_spread_bin_position_mae,
                        "delay_spread_bin_loss_denominator": mean_delay_spread_bin_loss_denominator,
                        "delay_spread_bin_class_weights": last_delay_spread_bin_class_weights,
                        "delay_spread_bin_label_order": list(DELAY_SPREAD_BIN_LABELS),
                        "delay_spread_bin_target_histogram": delay_spread_bin_target_histogram,
                        "delay_spread_bin_prediction_histogram": delay_spread_bin_prediction_histogram,
                        "loss_delay_spread": mean_delay_spread,
                        "loss_delay_spread_raw": mean_delay_spread_raw,
                        "delay_spread_raw_mae_ns": mean_delay_spread_raw_mae_ns,
                        "delay_spread_normalized_mae_ns": mean_delay_spread_normalized_mae_ns,
                        "loss_delay_spread_tail_classifier": mean_delay_spread_tail_classifier,
                        "accuracy_delay_spread_tail_classifier": mean_delay_spread_tail_accuracy,
                        "recall_delay_spread_tail_classifier": mean_delay_spread_tail_recall,
                        "false_positive_delay_spread_tail_classifier": mean_delay_spread_tail_false_positive,
                        "delay_spread_tail_label_order": list(DELAY_SPREAD_TAIL_LABELS),
                        "delay_spread_tail_positive_fraction": last_delay_spread_tail_positive_fraction,
                        "delay_spread_tail_prediction_fraction": last_delay_spread_tail_prediction_fraction,
                        "loss_direct_power": mean_direct_power,
                        "nlos_first_path_power_base_mae_db": mean_nlos_first_path_power_base_mae_db,
                        "nlos_first_path_power_enhanced_mae_db": mean_nlos_first_path_power_enhanced_mae_db,
                        "first_path_power_delta_saturation_fraction": mean_first_path_power_delta_saturation,
                        "loss_first_path_delay": mean_first_path_delay,
                        "loss_los_delay": mean_los_delay,
                        "loss_los_delay_nonnegative": mean_los_delay_nonnegative,
                        "loss_los_delay_consistency": mean_los_delay_consistency,
                        "los_delay_consistency_mae_ns": mean_los_delay_consistency_mae_ns,
                        "loss_los_angle": mean_los_angle,
                        "los_angle_mae_deg": mean_los_angle_mae_deg,
                        "loss_first_path_angle": mean_first_path_angle,
                        "first_path_angle_mae_deg": mean_first_path_angle_mae_deg,
                        "loss_first_path_angle_nlos": mean_first_path_angle_nlos,
                        "first_path_angle_nlos_mae_deg": mean_first_path_angle_nlos_mae_deg,
                        "k_factor_sample_weight_mean": mean_k_factor_sample_weight,
                        "first_path_power_sample_weight_mean": mean_first_path_power_sample_weight,
                        "first_path_delay_sample_weight_mean": mean_first_path_delay_sample_weight,
                        "logit_scale": mean_logit_scale,
                        "text_mode": text_mode,
                        "prototype_warmup_epochs": prototype_warmup_epochs,
                        "prototype_warmup_active": mean_prototype_warmup_active,
                        "semantic_classifier_weight": semantic_classifier_weight,
                        "semantic_classifier_class_weight": semantic_classifier_class_weight,
                        "semantic_classifier_logit_adjustment": semantic_classifier_logit_adjustment,
                        "attribute_classifier_weight": attribute_classifier_weight,
                        "attribute_classifier_fields": list(attribute_classifier_fields),
                        "attribute_classifier_class_weight": attribute_classifier_class_weight,
                        "attribute_classifier_logit_adjustment": attribute_classifier_logit_adjustment,
                        "attribute_remap": {
                            field: {
                                mapped: list(values)
                                for mapped, values in mapping.items()
                            }
                            for field, mapping in attribute_remap.items()
                        },
                        "token_norm_mode": token_norm_mode,
                        "use_power_branch": use_power_branch,
                        "detach_delay_spread_features": detach_delay_spread_features,
                        "detach_first_path_delay_features": detach_first_path_delay_features,
                        "use_delay_specific_encoder": use_delay_specific_encoder,
                        "use_los_angle_context_encoder": use_los_angle_context_encoder,
                        "use_first_path_angle_context_encoder": use_first_path_angle_context_encoder,
                        "aux_regression_weight": aux_regression_weight,
                        "aux_regression_targets": list(aux_regression_targets),
                        "k_factor_loss_weights": k_factor_loss_weights,
                        "strong_k_bin_classifier_weight": strong_k_bin_classifier_weight,
                        "strong_k_position_weight": strong_k_position_weight,
                        "strong_k_bin_weights": strong_k_bin_weights,
                        "strong_k_bin_label_order": list(K_FACTOR_STRONG_BIN_LABELS),
                        "first_path_power_bin_classifier_weight": first_path_power_bin_classifier_weight,
                        "first_path_power_bin_position_weight": first_path_power_bin_position_weight,
                        "first_path_power_bin_label_order": list(FIRST_POWER_DBW_BIN_LABELS),
                        "direct_power_weight": direct_power_weight,
                        "first_path_power_gate_mode": first_path_power_gate_mode,
                        "first_path_power_mode": first_path_power_mode,
                        "first_path_power_use_internal_gate": first_path_power_use_internal_gate,
                        "nlos_enhanced_power_loss": nlos_enhanced_power_loss,
                        "first_path_power_delta_limit": first_path_power_delta_limit,
                        "delay_spread_weight": delay_spread_weight,
                        "delay_spread_raw_weight": delay_spread_raw_weight,
                        "delay_spread_raw_beta_ns": delay_spread_raw_beta_ns,
                        "first_path_delay_weight": first_path_delay_weight,
                        "first_path_delay_raw_weight": first_path_delay_raw_weight,
                        "first_path_delay_fused_raw_weight": first_path_delay_fused_raw_weight,
                        "first_path_delay_raw_beta_ns": first_path_delay_raw_beta_ns,
                        "first_path_delay_bin_classifier_weight": first_path_delay_bin_classifier_weight,
                        "first_path_delay_bin_position_weight": first_path_delay_bin_position_weight,
                        "first_path_delay_bin_consistency_weight": first_path_delay_bin_consistency_weight,
                        "first_path_delay_bin_weights": first_path_delay_bin_weights,
                        "estimated_pdp_tail_bin_weight": estimated_pdp_tail_bin_weight,
                        "estimated_pdp_tail_labels": list(estimated_pdp_tail_labels),
                        "estimated_pdp_tail_gate_mode": estimated_pdp_tail_gate_mode,
                        "first_path_delay_tail_underestimate_weight": first_path_delay_tail_underestimate_weight,
                        "first_path_delay_bin_label_order": list(FIRST_PATH_DELAY_BIN_LABELS),
                        "los_delay_weight": los_delay_weight,
                        "los_delay_nonnegative_weight": los_delay_nonnegative_weight,
                        "use_physics_calibration_loss": use_physics_calibration_loss,
                        "los_delay_consistency_weight": los_delay_consistency_weight,
                        "los_angle_weight": los_angle_weight,
                        "first_path_angle_weight": first_path_angle_weight,
                        "first_path_angle_nlos_weight": first_path_angle_nlos_weight,
                        "delay_spread_teacher_weight": delay_spread_teacher_weight,
                        "delay_spread_bin_classifier_weight": delay_spread_bin_classifier_weight,
                        "delay_spread_bin_position_weight": delay_spread_bin_position_weight,
                        "delay_spread_tail_classifier_weight": delay_spread_tail_classifier_weight,
                        "reflection_count_classifier_weight": reflection_count_classifier_weight,
                        "reflection_count_regression_weight": reflection_count_regression_weight,
                        "reflection_count_nlos_weight": reflection_count_nlos_weight,
                        "interaction_count_soft_labels": interaction_count_soft_labels,
                        "interaction_count_fields": list(interaction_count_fields),
                        "interaction_count_classifier_weight": interaction_count_classifier_weight,
                        "interaction_count_regression_weight": interaction_count_regression_weight,
                        "reflection_count_bin_label_order": list(REFLECTION_COUNT_BIN_LABELS),
                        "delay_spread_bin_weights": delay_spread_bin_weights,
                        "delay_spread_bin_label_order": list(DELAY_SPREAD_BIN_LABELS),
                        "delay_spread_tail_label_order": list(DELAY_SPREAD_TAIL_LABELS),
                        "first_path_power_bin_weights": first_path_power_bin_weights,
                        "first_path_power_nlos_weight": first_path_power_nlos_weight,
                        "multipositive_distance_threshold": multipositive_distance_threshold,
                        "multipositive_positive_mode": multipositive_positive_mode,
                        "min_class_size_for_multipositive": min_class_size_for_multipositive,
                        "min_class_size": min_class_size,
                        "semantic_key_mode": semantic_key_mode,
                        "filter_attribute_values": {
                            field: list(values)
                            for field, values in effective_filter_attribute_values.items()
                        },
                        "limit_samples": limit_samples,
                        "limit_samples_by_attribute": limit_samples_by_attribute,
                        "limit_samples_per_attribute_value": limit_samples_per_attribute_value,
                        "max_delay_spread_ns": max_delay_spread_ns,
                        "freeze_csi": freeze_csi,
                        "freeze_text_prototypes": freeze_text_prototypes,
                        "multipositive_positive_count_mean": mean_positive_count,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
                + "\n"
            )
        if epoch % save_every == 0 or epoch == epochs:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "tokenizer_word2id": tokenizer.word2id,
                "prototype_keys": serialize_prototype_keys(prototype_bank["keys"]),
                "prototype_captions": prototype_bank["captions"],
                "args": {
                    "data_path": data_path,
                    "checkpoint": checkpoint_path,
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "temperature": temperature,
                    "token_norm_mode": token_norm_mode,
                    "use_power_branch": use_power_branch,
                    "detach_delay_spread_features": detach_delay_spread_features,
                    "detach_first_path_delay_features": detach_first_path_delay_features,
                    "use_delay_specific_encoder": use_delay_specific_encoder,
                    "use_los_angle_context_encoder": use_los_angle_context_encoder,
                    "use_first_path_angle_context_encoder": use_first_path_angle_context_encoder,
                    "warmup_epochs": warmup_epochs,
                    "min_lr": min_lr,
                    "csi_to_text_weight": csi_to_text_weight,
                    "prototype_weight": prototype_weight,
                    "text_prototype_weight": text_prototype_weight,
                    "text_mode": text_mode,
                    "prototype_warmup_epochs": prototype_warmup_epochs,
                    "semantic_classifier_weight": semantic_classifier_weight,
                    "semantic_classifier_class_weight": semantic_classifier_class_weight,
                    "semantic_classifier_logit_adjustment": semantic_classifier_logit_adjustment,
                    "attribute_classifier_weight": attribute_classifier_weight,
                    "attribute_classifier_fields": list(attribute_classifier_fields),
                    "attribute_classifier_class_weight": attribute_classifier_class_weight,
                    "attribute_classifier_logit_adjustment": attribute_classifier_logit_adjustment,
                    "attribute_remap": {
                        field: {
                            mapped: list(values)
                            for mapped, values in mapping.items()
                        }
                        for field, mapping in attribute_remap.items()
                    },
                    "aux_regression_weight": aux_regression_weight,
                    "aux_regression_targets": list(aux_regression_targets),
                    "k_factor_loss_weights": k_factor_loss_weights,
                    "strong_k_bin_classifier_weight": strong_k_bin_classifier_weight,
                    "strong_k_position_weight": strong_k_position_weight,
                    "strong_k_bin_weights": strong_k_bin_weights,
                    "strong_k_bin_label_order": list(K_FACTOR_STRONG_BIN_LABELS),
                    "first_path_power_bin_classifier_weight": first_path_power_bin_classifier_weight,
                    "first_path_power_bin_position_weight": first_path_power_bin_position_weight,
                    "first_path_power_bin_label_order": list(FIRST_POWER_DBW_BIN_LABELS),
                    "direct_power_weight": direct_power_weight,
                    "first_path_power_gate_mode": first_path_power_gate_mode,
                    "first_path_power_mode": first_path_power_mode,
                    "first_path_power_use_internal_gate": first_path_power_use_internal_gate,
                    "nlos_enhanced_power_loss": nlos_enhanced_power_loss,
                    "first_path_power_delta_limit": first_path_power_delta_limit,
                    "delay_spread_weight": delay_spread_weight,
                    "delay_spread_raw_weight": delay_spread_raw_weight,
                    "delay_spread_raw_beta_ns": delay_spread_raw_beta_ns,
                    "first_path_delay_weight": first_path_delay_weight,
                    "first_path_delay_raw_weight": first_path_delay_raw_weight,
                    "first_path_delay_fused_raw_weight": first_path_delay_fused_raw_weight,
                    "first_path_delay_raw_beta_ns": first_path_delay_raw_beta_ns,
                    "first_path_delay_bin_classifier_weight": first_path_delay_bin_classifier_weight,
                    "first_path_delay_bin_position_weight": first_path_delay_bin_position_weight,
                    "first_path_delay_bin_consistency_weight": first_path_delay_bin_consistency_weight,
                    "first_path_delay_bin_weights": first_path_delay_bin_weights,
                    "estimated_pdp_tail_bin_weight": estimated_pdp_tail_bin_weight,
                    "estimated_pdp_tail_labels": list(estimated_pdp_tail_labels),
                    "estimated_pdp_tail_gate_mode": estimated_pdp_tail_gate_mode,
                    "first_path_delay_tail_underestimate_weight": first_path_delay_tail_underestimate_weight,
                    "first_path_delay_bin_label_order": list(FIRST_PATH_DELAY_BIN_LABELS),
                    "los_delay_weight": los_delay_weight,
                    "los_delay_nonnegative_weight": los_delay_nonnegative_weight,
                    "use_physics_calibration_loss": use_physics_calibration_loss,
                    "los_delay_consistency_weight": los_delay_consistency_weight,
                    "los_angle_weight": los_angle_weight,
                    "first_path_angle_weight": first_path_angle_weight,
                    "first_path_angle_nlos_weight": first_path_angle_nlos_weight,
                    "delay_spread_teacher_weight": delay_spread_teacher_weight,
                    "delay_spread_bin_classifier_weight": delay_spread_bin_classifier_weight,
                    "delay_spread_bin_position_weight": delay_spread_bin_position_weight,
                    "delay_spread_tail_classifier_weight": delay_spread_tail_classifier_weight,
                    "reflection_count_classifier_weight": reflection_count_classifier_weight,
                    "reflection_count_regression_weight": reflection_count_regression_weight,
                    "reflection_count_nlos_weight": reflection_count_nlos_weight,
                    "interaction_count_soft_labels": interaction_count_soft_labels,
                    "interaction_count_fields": list(interaction_count_fields),
                    "interaction_count_classifier_weight": interaction_count_classifier_weight,
                    "interaction_count_regression_weight": interaction_count_regression_weight,
                    "reflection_count_bin_label_order": list(REFLECTION_COUNT_BIN_LABELS),
                    "delay_spread_bin_weights": delay_spread_bin_weights,
                    "delay_spread_bin_label_order": list(DELAY_SPREAD_BIN_LABELS),
                    "first_path_power_bin_weights": first_path_power_bin_weights,
                    "first_path_power_nlos_weight": first_path_power_nlos_weight,
                    "multipositive_distance_threshold": multipositive_distance_threshold,
                    "multipositive_positive_mode": multipositive_positive_mode,
                    "min_class_size_for_multipositive": min_class_size_for_multipositive,
                    "min_class_size": min_class_size,
                    "semantic_key_mode": semantic_key_mode,
                    "filter_attribute_values": {
                        field: list(values)
                        for field, values in effective_filter_attribute_values.items()
                    },
                    "limit_samples": limit_samples,
                    "limit_samples_by_attribute": limit_samples_by_attribute,
                    "limit_samples_per_attribute_value": limit_samples_per_attribute_value,
                    "max_delay_spread_ns": max_delay_spread_ns,
                    "allow_prototype_mismatch_transfer": allow_prototype_mismatch_transfer,
                    "freeze_csi": freeze_csi,
                    "freeze_text_prototypes": freeze_text_prototypes,
                    "phase": f"csi_clip_{text_mode}_text",
                },
            }
            ckpt_path = output_path / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, ckpt_path)
            torch.save(checkpoint, output_path / "checkpoint_last.pt")
            print(f"saved checkpoint to {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a synthetic end-to-end training step.")
    parser.add_argument("--data-path", type=str, help="Path to preprocessed .pt samples.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Load compatible model/tokenizer weights from a previous checkpoint before training.",
    )
    parser.add_argument(
        "--allow-prototype-mismatch-transfer",
        action="store_true",
        help=(
            "Allow transfer from a checkpoint with different learnable prototypes. "
            "Compatible weights are loaded, while prototypes and incompatible heads are reinitialized."
        ),
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--token-norm-mode", choices=["std", "rms", "none"])
    parser.add_argument("--enable-power-branch", action="store_true")
    parser.add_argument(
        "--detach-delay-spread-features",
        action="store_true",
        help="Detach CSI features before delay-spread heads so delay losses do not update the shared CSI encoder.",
    )
    parser.add_argument(
        "--detach-first-path-delay-features",
        action=argparse.BooleanOptionalAction,
        help=(
            "Detach first-path-delay head input from the delay-specific encoder. "
            "Use --no-detach-first-path-delay-features for first-delay-only upper-bound experiments."
        ),
    )
    parser.add_argument(
        "--use-delay-specific-encoder",
        action="store_true",
        help="Use a separate convolutional/attention encoder for delay-spread heads.",
    )
    parser.add_argument(
        "--use-los-angle-context-encoder",
        action="store_true",
        help="Use a beam-aware CSI encoder branch dedicated to the LoS angle head.",
    )
    parser.add_argument(
        "--use-first-path-angle-context-encoder",
        action="store_true",
        help="Use a first-path selector context branch dedicated to first-path angle prediction.",
    )
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--csi-to-text-weight", type=float)
    parser.add_argument("--prototype-weight", type=float)
    parser.add_argument("--text-prototype-weight", type=float)
    parser.add_argument("--text-mode", choices=["prototype", "instance", "multipositive"])
    parser.add_argument(
        "--prototype-warmup-epochs",
        type=int,
        help="Train only text-to-prototype alignment for the first N epochs before enabling CSI losses.",
    )
    parser.add_argument("--semantic-classifier-weight", type=float)
    parser.add_argument(
        "--semantic-classifier-class-weight",
        choices=["none", "mild", "balanced"],
        help=(
            "Optional class weighting for the direct semantic classifier. "
            "mild uses clipped inverse-fourth-root frequency weights; balanced uses clipped inverse-sqrt weights."
        ),
    )
    parser.add_argument(
        "--semantic-classifier-logit-adjustment",
        type=float,
        help=(
            "Optional training-time logit adjustment strength for the direct semantic classifier. "
            "Try small values such as 0.1 or 0.25; defaults to config or 0.0."
        ),
    )
    parser.add_argument("--semantic-key-mode", choices=semantic_key_mode_choices())
    parser.add_argument("--attribute-classifier-weight", type=float)
    parser.add_argument(
        "--attribute-classifier-fields",
        nargs="+",
        choices=semantic_key_field_choices(),
        help="SemanticKey fields supervised by multi-head attribute classification.",
    )
    parser.add_argument(
        "--attribute-classifier-class-weight",
        choices=["none", "mild", "balanced"],
        help=(
            "Optional per-attribute class weighting. mild uses clipped inverse-fourth-root "
            "frequency weights; balanced uses clipped inverse-sqrt weights."
        ),
    )
    parser.add_argument(
        "--attribute-classifier-logit-adjustment",
        type=float,
        help=(
            "Optional training-time logit adjustment strength for long-tail attribute heads. "
            "Try small values such as 0.1 or 0.25; defaults to config or 0.0."
        ),
    )
    parser.add_argument("--aux-regression-weight", type=float)
    parser.add_argument(
        "--k-factor-loss-weight",
        action="append",
        help=(
            "Per-bin weighting for K-factor aux regression as LABEL=WEIGHT. "
            "Labels: weak, strong_low, strong_mid, strong_high, strong_very_high."
        ),
    )
    parser.add_argument(
        "--strong-k-bin-classifier-weight",
        type=float,
        help="Auxiliary classification weight for strong K-factor low/mid/high/very_high prediction.",
    )
    parser.add_argument(
        "--strong-k-position-weight",
        type=float,
        help="Auxiliary regression weight for strong K-factor position within its predicted dB bin.",
    )
    parser.add_argument(
        "--strong-k-bin-weight",
        action="append",
        help=(
            "Per-class weighting for strong K-factor bin classification as LABEL=WEIGHT. "
            "Labels: low, mid, high, very_high."
        ),
    )
    parser.add_argument(
        "--first-path-power-bin-classifier-weight",
        type=float,
        help="Auxiliary classification weight for first-path-power bin prediction.",
    )
    parser.add_argument(
        "--first-path-power-bin-position-weight",
        type=float,
        help="Auxiliary regression weight for first-path-power position within its dBW bin.",
    )
    parser.add_argument(
        "--aux-regression-targets",
        nargs="+",
        choices=physics_aux_target_choices(),
        help="Physics targets used by aux regression. Defaults to train config, or all.",
    )
    parser.add_argument(
        "--direct-power-weight",
        type=float,
        help="Small explicit supervision weight for the direct first-path-power head.",
    )
    parser.add_argument(
        "--first-path-power-nlos-weight",
        type=float,
        help="Sample-weight multiplier for NLoS first-path-power supervision.",
    )
    parser.add_argument(
        "--first-path-power-gate-mode",
        choices=("none", "base"),
        help=(
            "How to route the final first-path-power prediction. base uses the base "
            "physics head; none uses the model default."
        ),
    )
    parser.add_argument(
        "--first-path-power-mode",
        choices=("residual", "absolute"),
        help=(
            "Prediction mode for the enhanced first-path-power branch. "
            "absolute predicts normalized power directly; residual predicts a correction from base."
        ),
    )
    parser.add_argument(
        "--first-path-power-use-internal-gate",
        action=argparse.BooleanOptionalAction,
        help=(
            "Use the enhanced branch's internal sigmoid gate for first-path-power "
            "residual correction. The outer predicted-LoS gate is still controlled "
            "by --first-path-power-gate-mode."
        ),
    )
    parser.add_argument(
        "--nlos-enhanced-power-loss",
        action=argparse.BooleanOptionalAction,
        help=(
            "When direct power supervision is enabled, train enhanced first-path "
            "power only on NLoS samples."
        ),
    )
    parser.add_argument(
        "--first-path-power-delta-limit",
        type=float,
        help="Clamp limit for the enhanced first-path-power delta in normalized units.",
    )
    parser.add_argument(
        "--delay-spread-weight",
        type=float,
        help="Explicit supervision weight for the independent delay-spread head.",
    )
    parser.add_argument(
        "--delay-spread-raw-weight",
        type=float,
        help="Supervision weight for delay-spread context prediction using raw ns Huber loss.",
    )
    parser.add_argument(
        "--delay-spread-raw-beta-ns",
        type=float,
        help="Huber transition beta in ns for --delay-spread-raw-weight.",
    )
    parser.add_argument(
        "--first-path-delay-weight",
        type=float,
        help="Supervision weight for first-path delay prediction from the shared delay context.",
    )
    parser.add_argument(
        "--first-path-delay-raw-weight",
        type=float,
        help="Supervision weight for first-path delay prediction using raw ns Huber loss.",
    )
    parser.add_argument(
        "--first-path-delay-fused-raw-weight",
        type=float,
        help=(
            "Supervision weight for bin+position fused first-path delay using raw ns Huber loss."
        ),
    )
    parser.add_argument(
        "--first-path-delay-raw-beta-ns",
        type=float,
        help="Huber transition beta in ns for --first-path-delay-raw-weight.",
    )
    parser.add_argument(
        "--first-path-delay-bin-classifier-weight",
        type=float,
        help="Auxiliary classification weight for first-path-delay bin prediction.",
    )
    parser.add_argument(
        "--first-path-delay-bin-position-weight",
        type=float,
        help="Auxiliary regression weight for first-path-delay position within its ns bin.",
    )
    parser.add_argument(
        "--first-path-delay-bin-consistency-weight",
        type=float,
        help=(
            "Penalty weight for continuous first-path-delay predictions that fall outside "
            "the target first-delay bin."
        ),
    )
    parser.add_argument(
        "--los-delay-weight",
        type=float,
        help="Supervision weight for LoS delay prediction from the shared delay context on LoS samples.",
    )
    parser.add_argument(
        "--los-delay-nonnegative-weight",
        type=float,
        help="Penalty weight for negative LoS delay predictions on LoS samples.",
    )
    parser.add_argument(
        "--use-physics-calibration-loss",
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable the Physics Calibration Loss Pack. The initial pack adds "
            "LoS first-path-delay/LoS-delay consistency."
        ),
    )
    parser.add_argument(
        "--los-delay-consistency-weight",
        type=float,
        help=(
            "Physics calibration weight for LoS consistency between "
            "first_path_delay and los_delay."
        ),
    )
    parser.add_argument(
        "--los-angle-weight",
        type=float,
        help="Supervision weight for LoS azimuth angle sin/cos prediction on LoS samples.",
    )
    parser.add_argument(
        "--first-path-angle-weight",
        type=float,
        help="Supervision weight for first-path azimuth angle sin/cos prediction.",
    )
    parser.add_argument(
        "--first-path-angle-nlos-weight",
        type=float,
        help="Extra supervision weight for first-path azimuth angle on NLoS samples.",
    )
    parser.add_argument(
        "--delay-spread-teacher-weight",
        type=float,
        help="Distillation weight from the profile-direct delay-spread teacher to the CSI-only delay head.",
    )
    parser.add_argument(
        "--delay-spread-bin-classifier-weight",
        type=float,
        help="Auxiliary classification weight for CSI-only delay-spread bin prediction.",
    )
    parser.add_argument(
        "--delay-spread-bin-position-weight",
        type=float,
        help="Auxiliary regression weight for CSI-only delay-spread position within its ns bin.",
    )
    parser.add_argument(
        "--delay-spread-tail-classifier-weight",
        type=float,
        help="Auxiliary binary classification weight for CSI-only delay-spread >=100ns/>=200ns prediction.",
    )
    parser.add_argument(
        "--reflection-count-classifier-weight",
        type=float,
        help=(
            "Auxiliary classification weight for total reflection-count bins "
            "from global CSI, delay context, and power stats."
        ),
    )
    parser.add_argument(
        "--reflection-count-regression-weight",
        type=float,
        help=(
            "Auxiliary regression weight for total reflection count "
            "from global CSI, delay context, and power stats."
        ),
    )
    parser.add_argument(
        "--reflection-count-nlos-weight",
        type=float,
        help="Sample weight multiplier for NLoS reflection-count supervision.",
    )
    parser.add_argument(
        "--interaction-count-soft-labels",
        action=argparse.BooleanOptionalAction,
        help=(
            "Use ordinal soft labels for interaction/reflection count classification."
        ),
    )
    parser.add_argument(
        "--interaction-count-fields",
        nargs="+",
        choices=("reflection_count",),
        help=(
            "Interaction-count targets included in the current calibration pack. "
            "Currently only reflection_count is implemented."
        ),
    )
    parser.add_argument(
        "--interaction-count-classifier-weight",
        type=float,
        help=(
            "Auxiliary classification weight for total reflection-count bins "
            "from global CSI, delay context, and power stats."
        ),
    )
    parser.add_argument(
        "--interaction-count-regression-weight",
        type=float,
        help=(
            "Auxiliary regression weight for total reflection count "
            "from global CSI, delay context, and power stats."
        ),
    )
    parser.add_argument(
        "--delay-spread-bin-weight",
        action="append",
        help=(
            "Per-bin weighting for delay-spread losses as LABEL=WEIGHT. "
            "Labels: 0_25, 25_50, 50_100, 100_200, 200_400, 400_plus."
        ),
    )
    parser.add_argument(
        "--first-path-power-bin-weight",
        action="append",
        help=(
            "Per-bin weighting for first-path-power losses as LABEL=WEIGHT. "
            "Labels: very_weak, weak, moderate, strong."
        ),
    )
    parser.add_argument(
        "--first-path-delay-bin-weight",
        action="append",
        help=(
            "Per-bin weighting for first-path-delay bin losses as LABEL=WEIGHT. "
            "Labels: 0_25, 25_50, 50_100, 100_200, 200_400, "
            "400_600, 600_800, 800_1040, 1040_1280, 1280_plus."
        ),
    )
    parser.add_argument(
        "--estimated-pdp-tail-bin-weight",
        type=float,
        help=(
            "Small auxiliary CE weight on first-path-delay bin logits, enabled "
            "only for configured tail samples/PDP-tail-gated samples."
        ),
    )
    parser.add_argument(
        "--estimated-pdp-tail-label",
        action="append",
        help=(
            "First-path-delay bin label treated as long-delay tail for the "
            "estimated-PDP auxiliary loss. Can be repeated."
        ),
    )
    parser.add_argument(
        "--estimated-pdp-tail-gate-mode",
        choices=(
            "target",
            "pdp_argmax",
            "target_or_pdp_argmax",
            "target_and_pdp_argmax",
        ),
        help=(
            "Sample gate for estimated-PDP tail auxiliary loss. target uses "
            "true tail labels; pdp_argmax uses the estimated-PDP peak-delay bin."
        ),
    )
    parser.add_argument(
        "--first-path-delay-tail-underestimate-weight",
        type=float,
        help=(
            "One-sided raw-ns loss weight for configured first-path-delay tail "
            "bins. Only underestimates are penalized."
        ),
    )
    parser.add_argument("--multipositive-distance-threshold", type=float)
    parser.add_argument(
        "--multipositive-positive-mode",
        choices=["semantic_and_physics", "semantic_or_physics", "semantic", "physics"],
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        help="Drop semantic classes with fewer than this many samples before training.",
    )
    parser.add_argument(
        "--filter-attribute-values",
        action="append",
        help=(
            "Keep only samples whose SemanticKey field matches listed values. "
            "Use FIELD=VALUE[,VALUE...], e.g. k_factor_bin=weak,strong."
        ),
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="Keep only the first N samples after semantic remapping and class-size filtering. For overfit debugging.",
    )
    parser.add_argument(
        "--limit-samples-by-attribute",
        choices=semantic_key_field_choices(),
        help="For overfit debugging, keep up to N samples per value of this SemanticKey field.",
    )
    parser.add_argument(
        "--limit-samples-per-attribute-value",
        type=int,
        help="Number of samples to keep per value when --limit-samples-by-attribute is set.",
    )
    parser.add_argument(
        "--max-delay-spread-ns",
        type=float,
        help="Drop samples whose delay_spread_ns is greater than or equal to this value.",
    )
    parser.add_argument(
        "--freeze-csi",
        action="store_true",
        help="Freeze the CSI encoder. Useful for attribute-head-only overfit diagnostics.",
    )
    parser.add_argument(
        "--freeze-text-prototypes",
        action="store_true",
        help="Freeze the text encoder and learnable prototypes.",
    )
    parser.add_argument("--min-class-size-for-multipositive", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--save-every", type=int)
    args = parser.parse_args()

    train_cfg = load_train_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs if args.epochs is not None else int(cfg_get(train_cfg, "epochs", 3))
    lr = args.lr if args.lr is not None else float(cfg_get(train_cfg, "lr", 3e-4))
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(cfg_get(train_cfg, "weight_decay", 1e-2))
    )
    batch_size = args.batch_size if args.batch_size is not None else int(cfg_get(train_cfg, "batch_size", 128))
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(cfg_get(train_cfg, "temperature", 0.07))
    )
    token_norm_mode = (
        args.token_norm_mode
        if args.token_norm_mode is not None
        else str(cfg_get(train_cfg, "token_norm_mode", "std"))
    )
    use_power_branch = bool(
        args.enable_power_branch
        or cfg_get(train_cfg, "use_power_branch", False)
    )
    detach_delay_spread_features = bool(
        args.detach_delay_spread_features
        or cfg_get(train_cfg, "detach_delay_spread_features", False)
    )
    detach_first_path_delay_features = bool(
        args.detach_first_path_delay_features
        if args.detach_first_path_delay_features is not None
        else cfg_get(train_cfg, "detach_first_path_delay_features", True)
    )
    use_delay_specific_encoder = bool(
        args.use_delay_specific_encoder
        or cfg_get(train_cfg, "use_delay_specific_encoder", False)
    )
    use_los_angle_context_encoder = bool(
        args.use_los_angle_context_encoder
        or cfg_get(train_cfg, "use_los_angle_context_encoder", False)
    )
    use_first_path_angle_context_encoder = bool(
        args.use_first_path_angle_context_encoder
        or cfg_get(train_cfg, "use_first_path_angle_context_encoder", False)
    )
    warmup_epochs = (
        args.warmup_epochs
        if args.warmup_epochs is not None
        else int(cfg_get(train_cfg, "warmup_epochs", 5))
    )
    min_lr = args.min_lr if args.min_lr is not None else float(cfg_get(train_cfg, "min_lr", 1e-5))
    csi_to_text_weight = (
        args.csi_to_text_weight
        if args.csi_to_text_weight is not None
        else float(cfg_get(train_cfg, "csi_to_text_weight", 1.0))
    )
    prototype_weight = (
        args.prototype_weight
        if args.prototype_weight is not None
        else float(cfg_get(train_cfg, "prototype_weight", 1.0))
    )
    text_prototype_weight = (
        args.text_prototype_weight
        if args.text_prototype_weight is not None
        else float(cfg_get(train_cfg, "text_prototype_weight", 1.0))
    )
    text_mode = args.text_mode if args.text_mode is not None else str(cfg_get(train_cfg, "text_mode", "prototype"))
    prototype_warmup_epochs = (
        args.prototype_warmup_epochs
        if args.prototype_warmup_epochs is not None
        else int(cfg_get(train_cfg, "prototype_warmup_epochs", 0))
    )
    semantic_key_mode = (
        args.semantic_key_mode
        if args.semantic_key_mode is not None
        else str(cfg_get(train_cfg, "semantic_key_mode", "full"))
    )
    semantic_classifier_weight = (
        args.semantic_classifier_weight
        if args.semantic_classifier_weight is not None
        else float(cfg_get(train_cfg, "semantic_classifier_weight", 0.0))
    )
    semantic_classifier_class_weight = (
        args.semantic_classifier_class_weight
        if args.semantic_classifier_class_weight is not None
        else str(cfg_get(train_cfg, "semantic_classifier_class_weight", "none"))
    )
    if semantic_classifier_class_weight not in ("none", "mild", "balanced"):
        raise ValueError("--semantic-classifier-class-weight must be one of: none, mild, balanced")
    semantic_classifier_logit_adjustment = (
        args.semantic_classifier_logit_adjustment
        if args.semantic_classifier_logit_adjustment is not None
        else float(cfg_get(train_cfg, "semantic_classifier_logit_adjustment", 0.0))
    )
    if semantic_classifier_logit_adjustment < 0.0:
        raise ValueError("--semantic-classifier-logit-adjustment must be non-negative.")
    attribute_classifier_weight = (
        args.attribute_classifier_weight
        if args.attribute_classifier_weight is not None
        else float(cfg_get(train_cfg, "attribute_classifier_weight", 0.0))
    )
    attribute_classifier_fields = parse_attribute_fields(
        args.attribute_classifier_fields
        if args.attribute_classifier_fields is not None
        else cfg_get(train_cfg, "attribute_classifier_fields", default_attribute_fields())
    )
    attribute_remap = parse_attribute_remap(cfg_get(train_cfg, "attribute_remap", None))
    attribute_classifier_class_weight = (
        args.attribute_classifier_class_weight
        if args.attribute_classifier_class_weight is not None
        else str(cfg_get(train_cfg, "attribute_classifier_class_weight", "none"))
    )
    if attribute_classifier_class_weight not in ("none", "mild", "balanced"):
        raise ValueError("--attribute-classifier-class-weight must be one of: none, mild, balanced")
    attribute_classifier_logit_adjustment = (
        args.attribute_classifier_logit_adjustment
        if args.attribute_classifier_logit_adjustment is not None
        else float(cfg_get(train_cfg, "attribute_classifier_logit_adjustment", 0.0))
    )
    if attribute_classifier_logit_adjustment < 0.0:
        raise ValueError("--attribute-classifier-logit-adjustment must be non-negative.")
    aux_regression_weight = (
        args.aux_regression_weight
        if args.aux_regression_weight is not None
        else float(cfg_get(train_cfg, "aux_regression_weight", 0.0))
    )
    k_factor_loss_weights = parse_k_factor_loss_weights(
        args.k_factor_loss_weight
        if args.k_factor_loss_weight is not None
        else cfg_get(train_cfg, "k_factor_loss_weights", None)
    )
    strong_k_bin_classifier_weight = (
        args.strong_k_bin_classifier_weight
        if args.strong_k_bin_classifier_weight is not None
        else float(cfg_get(train_cfg, "strong_k_bin_classifier_weight", 0.0))
    )
    strong_k_position_weight = (
        args.strong_k_position_weight
        if args.strong_k_position_weight is not None
        else float(cfg_get(train_cfg, "strong_k_position_weight", 0.0))
    )
    strong_k_bin_weights = parse_strong_k_bin_weights(
        args.strong_k_bin_weight
        if args.strong_k_bin_weight is not None
        else cfg_get(train_cfg, "strong_k_bin_weights", None)
    )
    first_path_power_bin_classifier_weight = (
        args.first_path_power_bin_classifier_weight
        if args.first_path_power_bin_classifier_weight is not None
        else float(cfg_get(train_cfg, "first_path_power_bin_classifier_weight", 0.0))
    )
    first_path_power_bin_position_weight = (
        args.first_path_power_bin_position_weight
        if args.first_path_power_bin_position_weight is not None
        else float(cfg_get(train_cfg, "first_path_power_bin_position_weight", 0.0))
    )
    direct_power_weight = (
        args.direct_power_weight
        if args.direct_power_weight is not None
        else float(cfg_get(train_cfg, "direct_power_weight", 0.0))
    )
    first_path_power_nlos_weight = (
        args.first_path_power_nlos_weight
        if args.first_path_power_nlos_weight is not None
        else float(cfg_get(train_cfg, "first_path_power_nlos_weight", 1.0))
    )
    if first_path_power_nlos_weight <= 0.0:
        raise ValueError("--first-path-power-nlos-weight must be positive.")
    first_path_power_gate_mode = (
        args.first_path_power_gate_mode
        if args.first_path_power_gate_mode is not None
        else str(cfg_get(train_cfg, "first_path_power_gate_mode", "none"))
    )
    if first_path_power_gate_mode not in {"none", "base"}:
        raise ValueError(
            "first_path_power_gate_mode must be one of: none, base."
        )
    first_path_power_mode = (
        args.first_path_power_mode
        if args.first_path_power_mode is not None
        else str(cfg_get(train_cfg, "first_path_power_mode", "residual"))
    )
    if first_path_power_mode not in {"residual", "absolute"}:
        raise ValueError(
            "first_path_power_mode must be one of: residual, absolute."
        )
    first_path_power_use_internal_gate = (
        args.first_path_power_use_internal_gate
        if args.first_path_power_use_internal_gate is not None
        else bool(cfg_get(train_cfg, "first_path_power_use_internal_gate", True))
    )
    nlos_enhanced_power_loss = (
        args.nlos_enhanced_power_loss
        if args.nlos_enhanced_power_loss is not None
        else bool(cfg_get(train_cfg, "nlos_enhanced_power_loss", False))
    )
    first_path_power_delta_limit = (
        args.first_path_power_delta_limit
        if args.first_path_power_delta_limit is not None
        else float(cfg_get(train_cfg, "first_path_power_delta_limit", 0.5))
    )
    if first_path_power_delta_limit <= 0.0:
        raise ValueError("--first-path-power-delta-limit must be positive.")
    delay_spread_weight = (
        args.delay_spread_weight
        if args.delay_spread_weight is not None
        else float(cfg_get(train_cfg, "delay_spread_weight", 0.0))
    )
    delay_spread_raw_weight = (
        args.delay_spread_raw_weight
        if args.delay_spread_raw_weight is not None
        else float(cfg_get(train_cfg, "delay_spread_raw_weight", 0.0))
    )
    delay_spread_raw_beta_ns = (
        args.delay_spread_raw_beta_ns
        if args.delay_spread_raw_beta_ns is not None
        else float(cfg_get(train_cfg, "delay_spread_raw_beta_ns", 20.0))
    )
    if delay_spread_raw_beta_ns <= 0.0:
        raise ValueError("--delay-spread-raw-beta-ns must be positive.")
    first_path_delay_weight = (
        args.first_path_delay_weight
        if args.first_path_delay_weight is not None
        else float(cfg_get(train_cfg, "first_path_delay_weight", 0.0))
    )
    first_path_delay_raw_weight = (
        args.first_path_delay_raw_weight
        if args.first_path_delay_raw_weight is not None
        else float(cfg_get(train_cfg, "first_path_delay_raw_weight", 0.0))
    )
    first_path_delay_fused_raw_weight = (
        args.first_path_delay_fused_raw_weight
        if args.first_path_delay_fused_raw_weight is not None
        else float(cfg_get(train_cfg, "first_path_delay_fused_raw_weight", 0.0))
    )
    first_path_delay_raw_beta_ns = (
        args.first_path_delay_raw_beta_ns
        if args.first_path_delay_raw_beta_ns is not None
        else float(cfg_get(train_cfg, "first_path_delay_raw_beta_ns", 20.0))
    )
    if first_path_delay_raw_beta_ns <= 0.0:
        raise ValueError("--first-path-delay-raw-beta-ns must be positive.")
    first_path_delay_bin_classifier_weight = (
        args.first_path_delay_bin_classifier_weight
        if args.first_path_delay_bin_classifier_weight is not None
        else float(cfg_get(train_cfg, "first_path_delay_bin_classifier_weight", 0.0))
    )
    first_path_delay_bin_position_weight = (
        args.first_path_delay_bin_position_weight
        if args.first_path_delay_bin_position_weight is not None
        else float(cfg_get(train_cfg, "first_path_delay_bin_position_weight", 0.0))
    )
    first_path_delay_bin_consistency_weight = (
        args.first_path_delay_bin_consistency_weight
        if args.first_path_delay_bin_consistency_weight is not None
        else float(cfg_get(train_cfg, "first_path_delay_bin_consistency_weight", 0.0))
    )
    first_path_delay_bin_weights = parse_first_path_delay_bin_weights(
        args.first_path_delay_bin_weight
        if args.first_path_delay_bin_weight is not None
        else cfg_get(train_cfg, "first_path_delay_bin_weights", None)
    )
    estimated_pdp_tail_bin_weight = (
        args.estimated_pdp_tail_bin_weight
        if args.estimated_pdp_tail_bin_weight is not None
        else float(cfg_get(train_cfg, "estimated_pdp_tail_bin_weight", 0.0))
    )
    if estimated_pdp_tail_bin_weight < 0.0:
        raise ValueError("--estimated-pdp-tail-bin-weight must be non-negative.")
    estimated_pdp_tail_labels = parse_first_path_delay_tail_labels(
        args.estimated_pdp_tail_label
        if args.estimated_pdp_tail_label is not None
        else cfg_get(train_cfg, "estimated_pdp_tail_labels", ("1040_1280",))
    )
    estimated_pdp_tail_gate_mode = (
        args.estimated_pdp_tail_gate_mode
        if args.estimated_pdp_tail_gate_mode is not None
        else str(cfg_get(train_cfg, "estimated_pdp_tail_gate_mode", "target"))
    )
    if estimated_pdp_tail_gate_mode not in {
        "target",
        "pdp_argmax",
        "target_or_pdp_argmax",
        "target_and_pdp_argmax",
    }:
        raise ValueError(
            "estimated_pdp_tail_gate_mode must be one of target, pdp_argmax, "
            "target_or_pdp_argmax, target_and_pdp_argmax."
        )
    first_path_delay_tail_underestimate_weight = (
        args.first_path_delay_tail_underestimate_weight
        if args.first_path_delay_tail_underestimate_weight is not None
        else float(cfg_get(train_cfg, "first_path_delay_tail_underestimate_weight", 0.0))
    )
    if first_path_delay_tail_underestimate_weight < 0.0:
        raise ValueError(
            "--first-path-delay-tail-underestimate-weight must be non-negative."
        )
    los_delay_weight = (
        args.los_delay_weight
        if args.los_delay_weight is not None
        else float(cfg_get(train_cfg, "los_delay_weight", 0.0))
    )
    los_delay_nonnegative_weight = (
        args.los_delay_nonnegative_weight
        if args.los_delay_nonnegative_weight is not None
        else float(cfg_get(train_cfg, "los_delay_nonnegative_weight", 0.0))
    )
    use_physics_calibration_loss = (
        args.use_physics_calibration_loss
        if args.use_physics_calibration_loss is not None
        else bool(cfg_get(train_cfg, "use_physics_calibration_loss", False))
    )
    los_delay_consistency_weight = (
        args.los_delay_consistency_weight
        if args.los_delay_consistency_weight is not None
        else float(cfg_get(train_cfg, "los_delay_consistency_weight", 0.0))
    )
    if los_delay_consistency_weight < 0.0:
        raise ValueError("--los-delay-consistency-weight must be non-negative.")
    los_angle_weight = (
        args.los_angle_weight
        if args.los_angle_weight is not None
        else float(cfg_get(train_cfg, "los_angle_weight", 0.0))
    )
    if los_angle_weight < 0.0:
        raise ValueError("--los-angle-weight must be non-negative.")
    first_path_angle_weight = (
        args.first_path_angle_weight
        if args.first_path_angle_weight is not None
        else float(cfg_get(train_cfg, "first_path_angle_weight", 0.0))
    )
    if first_path_angle_weight < 0.0:
        raise ValueError("--first-path-angle-weight must be non-negative.")
    first_path_angle_nlos_weight = (
        args.first_path_angle_nlos_weight
        if args.first_path_angle_nlos_weight is not None
        else float(cfg_get(train_cfg, "first_path_angle_nlos_weight", 0.0))
    )
    if first_path_angle_nlos_weight < 0.0:
        raise ValueError("--first-path-angle-nlos-weight must be non-negative.")
    delay_spread_teacher_weight = (
        args.delay_spread_teacher_weight
        if args.delay_spread_teacher_weight is not None
        else float(cfg_get(train_cfg, "delay_spread_teacher_weight", 0.1))
    )
    delay_spread_bin_weights = parse_delay_spread_bin_weights(
        args.delay_spread_bin_weight
        if args.delay_spread_bin_weight is not None
        else cfg_get(train_cfg, "delay_spread_bin_weights", None)
    )
    delay_spread_bin_classifier_weight = (
        args.delay_spread_bin_classifier_weight
        if args.delay_spread_bin_classifier_weight is not None
        else float(cfg_get(train_cfg, "delay_spread_bin_classifier_weight", 0.0))
    )
    delay_spread_bin_position_weight = (
        args.delay_spread_bin_position_weight
        if args.delay_spread_bin_position_weight is not None
        else float(cfg_get(train_cfg, "delay_spread_bin_position_weight", 0.0))
    )
    delay_spread_tail_classifier_weight = (
        args.delay_spread_tail_classifier_weight
        if args.delay_spread_tail_classifier_weight is not None
        else float(cfg_get(train_cfg, "delay_spread_tail_classifier_weight", 0.0))
    )
    reflection_count_classifier_weight = (
        args.reflection_count_classifier_weight
        if args.reflection_count_classifier_weight is not None
        else float(
            cfg_get(
                train_cfg,
                "reflection_count_classifier_weight",
                cfg_get(train_cfg, "interaction_count_classifier_weight", 0.0),
            )
        )
    )
    if reflection_count_classifier_weight < 0.0:
        raise ValueError("--reflection-count-classifier-weight must be non-negative.")
    reflection_count_regression_weight = (
        args.reflection_count_regression_weight
        if args.reflection_count_regression_weight is not None
        else float(
            cfg_get(
                train_cfg,
                "reflection_count_regression_weight",
                cfg_get(train_cfg, "interaction_count_regression_weight", 0.0),
            )
        )
    )
    if reflection_count_regression_weight < 0.0:
        raise ValueError("--reflection-count-regression-weight must be non-negative.")
    reflection_count_nlos_weight = (
        args.reflection_count_nlos_weight
        if args.reflection_count_nlos_weight is not None
        else float(cfg_get(train_cfg, "reflection_count_nlos_weight", 1.0))
    )
    if reflection_count_nlos_weight <= 0.0:
        raise ValueError("--reflection-count-nlos-weight must be positive.")
    interaction_count_soft_labels = (
        args.interaction_count_soft_labels
        if args.interaction_count_soft_labels is not None
        else bool(cfg_get(train_cfg, "interaction_count_soft_labels", False))
    )
    interaction_count_fields = parse_interaction_count_fields(
        args.interaction_count_fields
        if args.interaction_count_fields is not None
        else cfg_get(train_cfg, "interaction_count_fields", ("reflection_count",))
    )
    interaction_count_classifier_weight = (
        args.interaction_count_classifier_weight
        if args.interaction_count_classifier_weight is not None
        else reflection_count_classifier_weight
    )
    if interaction_count_classifier_weight < 0.0:
        raise ValueError("--interaction-count-classifier-weight must be non-negative.")
    if (
        args.reflection_count_classifier_weight is None
        and args.interaction_count_classifier_weight is not None
    ):
        reflection_count_classifier_weight = interaction_count_classifier_weight
    interaction_count_regression_weight = (
        args.interaction_count_regression_weight
        if args.interaction_count_regression_weight is not None
        else reflection_count_regression_weight
    )
    if interaction_count_regression_weight < 0.0:
        raise ValueError("--interaction-count-regression-weight must be non-negative.")
    if (
        args.reflection_count_regression_weight is None
        and args.interaction_count_regression_weight is not None
    ):
        reflection_count_regression_weight = interaction_count_regression_weight
    first_path_power_bin_weights = parse_first_path_power_bin_weights(
        args.first_path_power_bin_weight
        if args.first_path_power_bin_weight is not None
        else cfg_get(train_cfg, "first_path_power_bin_weights", None)
    )
    if (
        (
            first_path_power_bin_classifier_weight > 0.0
            or first_path_power_bin_position_weight > 0.0
            or direct_power_weight > 0.0
            or reflection_count_classifier_weight > 0.0
            or reflection_count_regression_weight > 0.0
        )
        and not use_power_branch
    ):
        use_power_branch = True
    aux_regression_targets = parse_aux_regression_targets(
        args.aux_regression_targets
        if args.aux_regression_targets is not None
        else cfg_get(train_cfg, "aux_regression_targets", "all")
    )
    multipositive_distance_threshold = (
        args.multipositive_distance_threshold
        if args.multipositive_distance_threshold is not None
        else float(cfg_get(train_cfg, "multipositive_distance_threshold", 0.25))
    )
    multipositive_positive_mode = (
        args.multipositive_positive_mode
        if args.multipositive_positive_mode is not None
        else str(cfg_get(train_cfg, "multipositive_positive_mode", "semantic_and_physics"))
    )
    min_class_size_for_multipositive = (
        args.min_class_size_for_multipositive
        if args.min_class_size_for_multipositive is not None
        else int(cfg_get(train_cfg, "min_class_size_for_multipositive", 2))
    )
    min_class_size = (
        args.min_class_size
        if args.min_class_size is not None
        else int(cfg_get(train_cfg, "min_class_size", 1))
    )
    filter_attribute_values = parse_attribute_value_filters(
        args.filter_attribute_values
        if args.filter_attribute_values is not None
        else cfg_get(train_cfg, "filter_attribute_values", None)
    )
    limit_samples = (
        args.limit_samples
        if args.limit_samples is not None
        else cfg_get(train_cfg, "limit_samples", None)
    )
    if limit_samples is not None:
        limit_samples = int(limit_samples)
    limit_samples_by_attribute = (
        args.limit_samples_by_attribute
        if args.limit_samples_by_attribute is not None
        else cfg_get(train_cfg, "limit_samples_by_attribute", None)
    )
    limit_samples_per_attribute_value = (
        args.limit_samples_per_attribute_value
        if args.limit_samples_per_attribute_value is not None
        else cfg_get(train_cfg, "limit_samples_per_attribute_value", None)
    )
    if limit_samples_per_attribute_value is not None:
        limit_samples_per_attribute_value = int(limit_samples_per_attribute_value)
    max_delay_spread_ns = (
        args.max_delay_spread_ns
        if args.max_delay_spread_ns is not None
        else cfg_get(train_cfg, "max_delay_spread_ns", None)
    )
    if max_delay_spread_ns is not None:
        max_delay_spread_ns = float(max_delay_spread_ns)
    freeze_csi = bool(args.freeze_csi or cfg_get(train_cfg, "freeze_csi", False))
    freeze_text_prototypes = bool(
        args.freeze_text_prototypes
        or cfg_get(train_cfg, "freeze_text_prototypes", False)
    )
    output_dir = args.output_dir if args.output_dir is not None else str(cfg_get(train_cfg, "output_dir", "artifacts/pretrain_csi_clip"))
    save_every = args.save_every if args.save_every is not None else int(cfg_get(train_cfg, "save_every", 1))
    checkpoint_path = args.checkpoint if args.checkpoint is not None else cfg_get(train_cfg, "checkpoint", None)

    if args.smoke_test:
        run_smoke_test(
            device,
            text_mode=text_mode,
            csi_to_text_weight=csi_to_text_weight,
            semantic_classifier_weight=semantic_classifier_weight,
            semantic_classifier_class_weight=semantic_classifier_class_weight,
            semantic_classifier_logit_adjustment=semantic_classifier_logit_adjustment,
            attribute_classifier_weight=attribute_classifier_weight,
            attribute_classifier_fields=attribute_classifier_fields,
            attribute_remap=attribute_remap,
            attribute_classifier_class_weight=attribute_classifier_class_weight,
            attribute_classifier_logit_adjustment=attribute_classifier_logit_adjustment,
            aux_regression_weight=aux_regression_weight,
            aux_regression_targets=aux_regression_targets,
            k_factor_loss_weights=k_factor_loss_weights,
            strong_k_bin_classifier_weight=strong_k_bin_classifier_weight,
            strong_k_position_weight=strong_k_position_weight,
            strong_k_bin_weights=strong_k_bin_weights,
            first_path_power_bin_classifier_weight=first_path_power_bin_classifier_weight,
            first_path_power_bin_position_weight=first_path_power_bin_position_weight,
            direct_power_weight=direct_power_weight,
            first_path_power_gate_mode=first_path_power_gate_mode,
            first_path_power_mode=first_path_power_mode,
            first_path_power_use_internal_gate=first_path_power_use_internal_gate,
            nlos_enhanced_power_loss=nlos_enhanced_power_loss,
            first_path_power_delta_limit=first_path_power_delta_limit,
            delay_spread_weight=delay_spread_weight,
            delay_spread_raw_weight=delay_spread_raw_weight,
            delay_spread_raw_beta_ns=delay_spread_raw_beta_ns,
            first_path_delay_weight=first_path_delay_weight,
            first_path_delay_raw_weight=first_path_delay_raw_weight,
            first_path_delay_fused_raw_weight=first_path_delay_fused_raw_weight,
            first_path_delay_raw_beta_ns=first_path_delay_raw_beta_ns,
            first_path_delay_bin_classifier_weight=first_path_delay_bin_classifier_weight,
            first_path_delay_bin_position_weight=first_path_delay_bin_position_weight,
            first_path_delay_bin_consistency_weight=first_path_delay_bin_consistency_weight,
            first_path_delay_bin_weights=first_path_delay_bin_weights,
            estimated_pdp_tail_bin_weight=estimated_pdp_tail_bin_weight,
            estimated_pdp_tail_labels=estimated_pdp_tail_labels,
            estimated_pdp_tail_gate_mode=estimated_pdp_tail_gate_mode,
            first_path_delay_tail_underestimate_weight=first_path_delay_tail_underestimate_weight,
            los_delay_weight=los_delay_weight,
            los_delay_nonnegative_weight=los_delay_nonnegative_weight,
            use_physics_calibration_loss=use_physics_calibration_loss,
            los_delay_consistency_weight=los_delay_consistency_weight,
            los_angle_weight=los_angle_weight,
            first_path_angle_weight=first_path_angle_weight,
            first_path_angle_nlos_weight=first_path_angle_nlos_weight,
            delay_spread_teacher_weight=delay_spread_teacher_weight,
            delay_spread_bin_weights=delay_spread_bin_weights,
            delay_spread_bin_classifier_weight=delay_spread_bin_classifier_weight,
            delay_spread_bin_position_weight=delay_spread_bin_position_weight,
            delay_spread_tail_classifier_weight=delay_spread_tail_classifier_weight,
            interaction_count_classifier_weight=interaction_count_classifier_weight,
            interaction_count_regression_weight=interaction_count_regression_weight,
            reflection_count_classifier_weight=reflection_count_classifier_weight,
            reflection_count_regression_weight=reflection_count_regression_weight,
            reflection_count_nlos_weight=reflection_count_nlos_weight,
            interaction_count_soft_labels=interaction_count_soft_labels,
            interaction_count_fields=interaction_count_fields,
            first_path_power_bin_weights=first_path_power_bin_weights,
            first_path_power_nlos_weight=first_path_power_nlos_weight,
            multipositive_distance_threshold=multipositive_distance_threshold,
            multipositive_positive_mode=multipositive_positive_mode,
            min_class_size_for_multipositive=min_class_size_for_multipositive,
            semantic_key_mode=semantic_key_mode,
            token_norm_mode=token_norm_mode,
            use_power_branch=use_power_branch,
            detach_delay_spread_features=detach_delay_spread_features,
            detach_first_path_delay_features=detach_first_path_delay_features,
            use_delay_specific_encoder=use_delay_specific_encoder,
            use_los_angle_context_encoder=use_los_angle_context_encoder,
            use_first_path_angle_context_encoder=use_first_path_angle_context_encoder,
        )
        return

    data_path = args.data_path if args.data_path is not None else train_cfg.get("data_path")
    if data_path:
        run_real_pretrain(
            data_path=data_path,
            checkpoint_path=checkpoint_path,
            device=device,
            epochs=epochs,
            max_steps_per_epoch=args.max_steps_per_epoch,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            temperature=temperature,
            token_norm_mode=token_norm_mode,
            use_power_branch=use_power_branch,
            detach_delay_spread_features=detach_delay_spread_features,
            detach_first_path_delay_features=detach_first_path_delay_features,
            use_delay_specific_encoder=use_delay_specific_encoder,
            use_los_angle_context_encoder=use_los_angle_context_encoder,
            use_first_path_angle_context_encoder=use_first_path_angle_context_encoder,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
            csi_to_text_weight=csi_to_text_weight,
            prototype_weight=prototype_weight,
            text_prototype_weight=text_prototype_weight,
            text_mode=text_mode,
            prototype_warmup_epochs=prototype_warmup_epochs,
            semantic_classifier_weight=semantic_classifier_weight,
            semantic_classifier_class_weight=semantic_classifier_class_weight,
            semantic_classifier_logit_adjustment=semantic_classifier_logit_adjustment,
            attribute_classifier_weight=attribute_classifier_weight,
            attribute_classifier_fields=attribute_classifier_fields,
            attribute_classifier_class_weight=attribute_classifier_class_weight,
            attribute_classifier_logit_adjustment=attribute_classifier_logit_adjustment,
            aux_regression_weight=aux_regression_weight,
            aux_regression_targets=aux_regression_targets,
            k_factor_loss_weights=k_factor_loss_weights,
            strong_k_bin_classifier_weight=strong_k_bin_classifier_weight,
            strong_k_position_weight=strong_k_position_weight,
            strong_k_bin_weights=strong_k_bin_weights,
            first_path_power_bin_classifier_weight=first_path_power_bin_classifier_weight,
            first_path_power_bin_position_weight=first_path_power_bin_position_weight,
            direct_power_weight=direct_power_weight,
            first_path_power_gate_mode=first_path_power_gate_mode,
            first_path_power_mode=first_path_power_mode,
            first_path_power_use_internal_gate=first_path_power_use_internal_gate,
            nlos_enhanced_power_loss=nlos_enhanced_power_loss,
            first_path_power_delta_limit=first_path_power_delta_limit,
            delay_spread_weight=delay_spread_weight,
            delay_spread_raw_weight=delay_spread_raw_weight,
            delay_spread_raw_beta_ns=delay_spread_raw_beta_ns,
            first_path_delay_weight=first_path_delay_weight,
            first_path_delay_raw_weight=first_path_delay_raw_weight,
            first_path_delay_fused_raw_weight=first_path_delay_fused_raw_weight,
            first_path_delay_raw_beta_ns=first_path_delay_raw_beta_ns,
            first_path_delay_bin_classifier_weight=first_path_delay_bin_classifier_weight,
            first_path_delay_bin_position_weight=first_path_delay_bin_position_weight,
            first_path_delay_bin_consistency_weight=first_path_delay_bin_consistency_weight,
            first_path_delay_bin_weights=first_path_delay_bin_weights,
            estimated_pdp_tail_bin_weight=estimated_pdp_tail_bin_weight,
            estimated_pdp_tail_labels=estimated_pdp_tail_labels,
            estimated_pdp_tail_gate_mode=estimated_pdp_tail_gate_mode,
            first_path_delay_tail_underestimate_weight=first_path_delay_tail_underestimate_weight,
            los_delay_weight=los_delay_weight,
            los_delay_nonnegative_weight=los_delay_nonnegative_weight,
            use_physics_calibration_loss=use_physics_calibration_loss,
            los_delay_consistency_weight=los_delay_consistency_weight,
            los_angle_weight=los_angle_weight,
            first_path_angle_weight=first_path_angle_weight,
            first_path_angle_nlos_weight=first_path_angle_nlos_weight,
            delay_spread_teacher_weight=delay_spread_teacher_weight,
            delay_spread_bin_weights=delay_spread_bin_weights,
            delay_spread_bin_classifier_weight=delay_spread_bin_classifier_weight,
            delay_spread_bin_position_weight=delay_spread_bin_position_weight,
            delay_spread_tail_classifier_weight=delay_spread_tail_classifier_weight,
            interaction_count_classifier_weight=interaction_count_classifier_weight,
            interaction_count_regression_weight=interaction_count_regression_weight,
            reflection_count_classifier_weight=reflection_count_classifier_weight,
            reflection_count_regression_weight=reflection_count_regression_weight,
            reflection_count_nlos_weight=reflection_count_nlos_weight,
            interaction_count_soft_labels=interaction_count_soft_labels,
            interaction_count_fields=interaction_count_fields,
            first_path_power_bin_weights=first_path_power_bin_weights,
            first_path_power_nlos_weight=first_path_power_nlos_weight,
            multipositive_distance_threshold=multipositive_distance_threshold,
            multipositive_positive_mode=multipositive_positive_mode,
            min_class_size_for_multipositive=min_class_size_for_multipositive,
            min_class_size=min_class_size,
            semantic_key_mode=semantic_key_mode,
            attribute_remap=attribute_remap,
            filter_attribute_values=filter_attribute_values,
            limit_samples=limit_samples,
            limit_samples_by_attribute=limit_samples_by_attribute,
            limit_samples_per_attribute_value=limit_samples_per_attribute_value,
            max_delay_spread_ns=max_delay_spread_ns,
            allow_prototype_mismatch_transfer=args.allow_prototype_mismatch_transfer,
            freeze_csi=freeze_csi,
            freeze_text_prototypes=freeze_text_prototypes,
            output_dir=output_dir,
            save_every=save_every,
        )
        return

    raise SystemExit("Use --smoke-test, provide --data-path, or set train.data_path in the config.")


if __name__ == "__main__":
    main()
