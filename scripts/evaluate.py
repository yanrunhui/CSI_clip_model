from __future__ import annotations

import argparse
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
from models.model import CSIClip
from models.text_encoder import PhysicsTextEncoder
from scripts.pretrain import assert_checkpoint_prototype_compatibility
from training.losses import cosine_alignment_loss, paired_contrastive_loss

PHYSICAL_DESCRIPTION_FIELDS = (
    "delay_spread_ns",
    "k_factor_db",
    "azimuth_spread_deg",
    "first_path_power_dbw",
)

PHYSICAL_DESCRIPTION_TOLERANCES = {
    "delay_spread_ns": 20.0,
    "k_factor_db": 3.0,
    "azimuth_spread_deg": 10.0,
    "first_path_power_dbw": 3.0,
}


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
) -> tuple[list[SemanticKey], torch.Tensor, torch.Tensor, dict[SemanticKey, int]]:
    generator = CaptionGenerator()
    unique_keys = sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
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
        "first_path_power_dbw": float(raw_values[_physics_target_index("first_path_power_dbw")]),
    }


def _render_physical_description(record: dict[str, float | str]) -> str:
    los_text = "LoS" if str(record["los_status"]) == "los" else "NLoS"
    return (
        f"This channel is likely {los_text}, with a delay spread of about "
        f"{_format_scalar(float(record['delay_spread_ns']))} ns, a K-factor of about "
        f"{_format_scalar(float(record['k_factor_db']))} dB, an azimuth spread of about "
        f"{_format_scalar(float(record['azimuth_spread_deg']))} deg, and a first-path power "
        f"of about {_format_scalar(float(record['first_path_power_dbw']))} dBW."
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


def _first_path_power_residual_scale(model: torch.nn.Module) -> float | None:
    scale = getattr(model, "first_path_power_residual_scale", None)
    if scale is None:
        return None
    return float(scale.detach().cpu().item())


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
    attribute_fields_override: tuple[str, ...] | None = None,
    filter_attribute_values_override: dict[str, tuple[str, ...]] | None = None,
    limit_samples_override: int | None = None,
    limit_samples_by_attribute_override: str | None = None,
    limit_samples_per_attribute_value_override: int | None = None,
    attribute_binary_thresholds: dict[str, float] | None = None,
    physical_caption_examples: int = 3,
) -> None:
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) if checkpoint_path else None
    text_mode = _infer_text_mode(checkpoint, text_mode_override)
    min_class_size = _infer_min_class_size(checkpoint, min_class_size_override)
    semantic_key_mode = _infer_semantic_key_mode(checkpoint, semantic_key_mode_override)
    token_norm_mode = _infer_token_norm_mode(checkpoint, token_norm_mode_override)
    use_power_branch = _infer_use_power_branch(checkpoint, use_power_branch_override)
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
    tokenizer = build_tokenizer(samples, checkpoint)
    prototype_keys, prototype_token_ids, prototype_token_mask, prototype_label_map = build_prototype_bank(
        samples,
        tokenizer,
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
        attribute_num_classes={
            field: len(label_map)
            for field, label_map in attribute_label_maps.items()
        },
    ).to(device)
    if checkpoint is not None:
        assert_checkpoint_prototype_compatibility(
            checkpoint,
            prototype_keys,
            expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
            context="evaluation checkpoint",
        )
        _load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()
    residual_scale = _first_path_power_residual_scale(model)

    all_csi_features = []
    all_semantic_logits = []
    all_attribute_logits = {field: [] for field in attribute_label_maps}
    all_attribute_labels = {field: [] for field in attribute_label_maps}
    all_instance_text_features = []
    all_physics_predictions = []
    all_base_physics_predictions = []
    all_direct_first_path_power_predictions = []
    all_physics_targets = []
    all_physics_raw_targets = []
    all_physics_masks = []
    all_first_path_power_residual_scaled = []
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
        if use_power_branch:
            power_context = model.encode_power_context(
                batch["tokens"],
                batch["token_mask"],
            )
        physics_outputs = model.predict_physics_components(
            csi_features_raw,
            power_context=power_context,
        )
        physics_predictions = physics_outputs["final"]
        all_base_physics_predictions.append(physics_outputs["base"].cpu())
        all_direct_first_path_power_predictions.append(
            physics_outputs["direct_first_path_power"].cpu()
        )
        all_first_path_power_residual_scaled.append(physics_outputs["residual_scaled"].cpu())
        all_physics_predictions.append(physics_predictions.cpu())
        all_physics_targets.append(batch["physics_targets"].cpu())
        all_physics_raw_targets.append(batch["physics_raw_targets"].cpu())
        all_physics_masks.append(batch["physics_target_mask"].cpu())
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
    direct_first_path_power_predictions = torch.cat(all_direct_first_path_power_predictions, dim=0)
    physics_targets = torch.cat(all_physics_targets, dim=0)
    physics_raw_targets = torch.cat(all_physics_raw_targets, dim=0)
    physics_masks = torch.cat(all_physics_masks, dim=0)
    first_path_power_residual_scaled = torch.cat(all_first_path_power_residual_scaled, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    logit_scale = float(model.logit_scale.exp().detach().cpu().item())
    prototype_logits = logit_scale * csi_features @ prototype_features.T

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
    if residual_scale is not None:
        print(f"first_path_power_residual_scale={residual_scale:.6f}")
    print(f"text_mode={text_mode}")
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"token_norm_mode={token_norm_mode}")
    print(f"use_power_branch={use_power_branch}")
    print(f"min_class_size={min_class_size}")
    print(f"attribute_remap={format_attribute_remap(attribute_remap)}")
    print(f"filter_attribute_values={format_attribute_value_filters(filter_attribute_values)}")
    print(f"limit_samples={limit_samples}")
    print(f"limit_samples_by_attribute={limit_samples_by_attribute}")
    print(f"limit_samples_per_attribute_value={limit_samples_per_attribute_value}")
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
    first_path_power_idx = _physics_target_index("first_path_power_dbw")
    first_path_power_mask = physics_masks[:, first_path_power_idx]
    if bool(first_path_power_mask.any()):
        base_physics_raw_predictions = _physics_raw_predictions(base_physics_predictions)
        final_physics_raw_predictions = _physics_raw_predictions(physics_predictions)
        base_first_path_power_errors = (
            base_physics_raw_predictions[:, first_path_power_idx] - physics_raw_targets[:, first_path_power_idx]
        ).abs()
        final_first_path_power_errors = (
            final_physics_raw_predictions[:, first_path_power_idx] - physics_raw_targets[:, first_path_power_idx]
        ).abs()
        direct_first_path_power_errors = (
            (
                direct_first_path_power_predictions * PHYSICS_TARGET_SCALES[first_path_power_idx]
                + PHYSICS_TARGET_OFFSETS[first_path_power_idx]
            )
            - physics_raw_targets[:, first_path_power_idx]
        ).abs()
        print(f"base_first_power_MAE={float(base_first_path_power_errors[first_path_power_mask].mean()):.4f}")
        print(f"direct_power_head_MAE={float(direct_first_path_power_errors[first_path_power_mask].mean()):.4f}")
        print(f"residual_final_MAE={float(final_first_path_power_errors[first_path_power_mask].mean()):.4f}")
    else:
        print("base_first_power_MAE=nan")
        print("direct_power_head_MAE=nan")
        print("residual_final_MAE=nan")
    print(f"residual_scaled_mean={float(first_path_power_residual_scaled.mean()):.6f}")
    print(f"residual_scaled_std={float(first_path_power_residual_scaled.std()):.6f}")
    if residual_scale is not None:
        print(f"residual_scale={residual_scale:.6f}")
    _print_structured_physical_description_metrics(
        physics_predictions=physics_predictions,
        physics_raw_targets=physics_raw_targets,
        physics_masks=physics_masks,
        semantic_keys=all_semantic_keys,
        prototype_logits=prototype_logits,
        prototype_keys=prototype_keys,
        example_count=physical_caption_examples,
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
        if name in {"first_path_aoa_az_sin", "first_path_aoa_az_cos"}:
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
    parser.add_argument(
        "--enable-power-branch",
        action="store_true",
        help="Enable the power branch regardless of checkpoint args.",
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
        use_power_branch_override=True if args.enable_power_branch else None,
        attribute_fields_override=tuple(args.attribute_classifier_fields) if args.attribute_classifier_fields else None,
        filter_attribute_values_override=(
            parse_attribute_value_filters(args.filter_attribute_values)
            if args.filter_attribute_values is not None
            else None
        ),
        limit_samples_override=args.limit_samples,
        limit_samples_by_attribute_override=args.limit_samples_by_attribute,
        limit_samples_per_attribute_value_override=args.limit_samples_per_attribute_value,
        attribute_binary_thresholds=parse_attribute_binary_thresholds(args.attribute_binary_threshold),
        physical_caption_examples=args.physical_caption_examples,
    )


if __name__ == "__main__":
    main()
