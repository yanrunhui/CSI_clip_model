from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset, apply_semantic_key_mode, semantic_key_mode_choices
from data.semantic_key import SemanticKey, semantic_key_attribute_raw_value, semantic_key_attribute_value
from scripts.pretrain import (
    cfg_get,
    format_attribute_remap,
    format_attribute_value_filters,
    load_train_config,
    parse_attribute_remap,
    parse_attribute_value_filters,
)


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


def infer_arg(checkpoint: dict | None, name: str, override, default):
    if override is not None:
        return override
    if checkpoint is not None:
        value = checkpoint.get("args", {}).get(name)
        if value is not None:
            return value
    return default


def filter_samples_by_min_class_size(samples, min_class_size: int):
    if min_class_size <= 1:
        return samples
    key_counts = Counter(sample.semantic_key for sample in samples)
    filtered = [sample for sample in samples if key_counts[sample.semantic_key] >= min_class_size]
    if not filtered:
        raise ValueError(f"No samples remain after min_class_size={min_class_size}.")
    return filtered


def filter_samples_by_attribute_values(samples, filters: dict[str, tuple[str, ...]]):
    filtered = samples
    for field, values in filters.items():
        allowed_values = set(values)
        filtered = [
            sample
            for sample in filtered
            if semantic_key_attribute_raw_value(sample.semantic_key, field) in allowed_values
        ]
        if not filtered:
            raise ValueError(
                f"No samples remain after filtering {field} to values {','.join(values)}."
            )
    return filtered


def limit_samples_by_attribute_value(
    samples,
    attribute_field: str | None,
    samples_per_value: int | None,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]] | None = None,
):
    if attribute_field is None and samples_per_value is None:
        return samples
    if attribute_field is None or samples_per_value is None:
        raise ValueError("--limit-samples-by-attribute and --limit-samples-per-attribute-value must be used together.")
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
        raise ValueError(f"No samples remain after limit_samples_by_attribute={attribute_field!r}.")
    return limited


def limit_samples(samples, limit: int | None):
    if limit is None:
        return samples
    if limit <= 0:
        raise ValueError("--limit-samples must be a positive integer.")
    limited = samples[:limit]
    if not limited:
        raise ValueError(f"No samples remain after limit_samples={limit}.")
    return limited


def _print_counter(title: str, counts: Counter, total: int, top_k: int | None = None) -> None:
    print(title)
    if not counts:
        print("  <empty>")
        return
    for key, value in counts.most_common(top_k):
        ratio = 100.0 * value / max(total, 1)
        print(f"  {key}: {value} ({ratio:.2f}%)")


def diagnose_distribution(
    data_path: str,
    checkpoint_path: str | None,
    config_path: str,
    semantic_key_mode_override: str | None,
    min_class_size_override: int | None,
    filter_attribute_values_override: dict[str, tuple[str, ...]] | None,
    limit_samples_override: int | None,
    limit_samples_by_attribute_override: str | None,
    limit_samples_per_attribute_value_override: int | None,
    top_k: int,
) -> None:
    train_cfg = load_train_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path else None

    semantic_key_mode = str(
        infer_arg(checkpoint, "semantic_key_mode", semantic_key_mode_override, cfg_get(train_cfg, "semantic_key_mode", "full"))
    )
    min_class_size = int(
        infer_arg(checkpoint, "min_class_size", min_class_size_override, cfg_get(train_cfg, "min_class_size", 1))
    )
    attribute_remap = parse_attribute_remap(
        infer_arg(checkpoint, "attribute_remap", None, cfg_get(train_cfg, "attribute_remap", None))
    )
    filter_attribute_values = parse_attribute_value_filters(
        infer_arg(
            checkpoint,
            "filter_attribute_values",
            filter_attribute_values_override,
            cfg_get(train_cfg, "filter_attribute_values", None),
        )
    )
    limit = infer_arg(checkpoint, "limit_samples", limit_samples_override, cfg_get(train_cfg, "limit_samples", None))
    limit = int(limit) if limit is not None else None
    limit_by_attribute = infer_arg(
        checkpoint,
        "limit_samples_by_attribute",
        limit_samples_by_attribute_override,
        cfg_get(train_cfg, "limit_samples_by_attribute", None),
    )
    limit_per_value = infer_arg(
        checkpoint,
        "limit_samples_per_attribute_value",
        limit_samples_per_attribute_value_override,
        cfg_get(train_cfg, "limit_samples_per_attribute_value", None),
    )
    limit_per_value = int(limit_per_value) if limit_per_value is not None else None

    dataset = PreprocessedCSIDataset.from_pt(data_path)
    samples = apply_semantic_key_mode(dataset.samples, semantic_key_mode)
    samples = filter_samples_by_min_class_size(samples, min_class_size)
    samples = filter_samples_by_attribute_values(samples, filter_attribute_values)
    samples = limit_samples_by_attribute_value(
        samples,
        limit_by_attribute,
        limit_per_value,
        attribute_remap=attribute_remap,
    )
    samples = limit_samples(samples, limit)
    if not samples:
        raise ValueError("No samples remain after filtering.")

    semantic_counts = Counter(sample.semantic_key for sample in samples)
    los_counts = Counter(sample.semantic_key.los_status for sample in samples)
    path_counts = Counter(sample.semantic_key.path_richness for sample in samples)
    k_counts = Counter(sample.semantic_key.k_factor_bin for sample in samples)
    first_delay_counts = Counter(sample.semantic_key.first_delay_bin for sample in samples)
    first_power_counts = Counter(sample.semantic_key.first_power_bin for sample in samples)
    class_size_counts = Counter(semantic_counts.values())

    print(f"data_path={data_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"min_class_size={min_class_size}")
    print(f"attribute_remap={format_attribute_remap(attribute_remap)}")
    print(f"filter_attribute_values={format_attribute_value_filters(filter_attribute_values)}")
    print(f"limit_samples={limit}")
    print(f"limit_samples_by_attribute={limit_by_attribute}")
    print(f"limit_samples_per_attribute_value={limit_per_value}")
    print(f"num_samples={len(samples)}")
    print(f"semantic_prototypes={len(semantic_counts)}")
    print(f"largest_class={max(semantic_counts.values())}")
    print(f"smallest_class={min(semantic_counts.values())}")

    _print_counter("Class Size Histogram", class_size_counts, len(semantic_counts))
    print()
    _print_counter("LoS Counts", los_counts, len(samples))
    print()
    _print_counter("Path Richness Counts", path_counts, len(samples))
    print()
    _print_counter("K-factor Counts", k_counts, len(samples))
    print()
    _print_counter("First Delay Counts", first_delay_counts, len(samples))
    print()
    _print_counter("First Power Counts", first_power_counts, len(samples))
    print()
    print("Top coarse_k semantic classes")
    for key, count in semantic_counts.most_common(top_k):
        ratio = 100.0 * count / max(len(samples), 1)
        print(f"  {key}: {count} ({ratio:.2f}%)")
    print()
    print("Smallest coarse_k semantic classes")
    for key, count in sorted(semantic_counts.items(), key=lambda item: (item[1], semantic_key_sort_key(item[0])))[:top_k]:
        ratio = 100.0 * count / max(len(samples), 1)
        print(f"  {key}: {count} ({ratio:.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--semantic-key-mode", choices=semantic_key_mode_choices())
    parser.add_argument("--min-class-size", type=int)
    parser.add_argument("--filter-attribute-values", action="append")
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-samples-by-attribute")
    parser.add_argument("--limit-samples-per-attribute-value", type=int)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    diagnose_distribution(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        semantic_key_mode_override=args.semantic_key_mode,
        min_class_size_override=args.min_class_size,
        filter_attribute_values_override=(
            parse_attribute_value_filters(args.filter_attribute_values)
            if args.filter_attribute_values is not None
            else None
        ),
        limit_samples_override=args.limit_samples,
        limit_samples_by_attribute_override=args.limit_samples_by_attribute,
        limit_samples_per_attribute_value_override=args.limit_samples_per_attribute_value,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
