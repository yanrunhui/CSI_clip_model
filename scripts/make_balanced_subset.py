from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import apply_semantic_key_mode, semantic_key_mode_choices
from data.semantic_key import semantic_key_attribute_value, semantic_key_field_choices


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise ValueError(f"Config path does not exist: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("train", data)


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


def make_balanced_subset(
    input_path: str,
    output_path: str,
    max_per_key: int,
    min_per_key: int,
    seed: int,
    balance_fields: list[str],
    semantic_key_mode: str,
    candidate_max_per_key: int | None,
    candidate_min_per_key: int,
    candidate_balance_fields: list[str] | None,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]],
) -> None:
    if max_per_key <= 0:
        raise ValueError("--max-per-key must be a positive integer.")
    if min_per_key <= 0:
        raise ValueError("--min-per-key must be a positive integer.")
    if candidate_max_per_key is not None and candidate_max_per_key <= 0:
        raise ValueError("--candidate-max-per-key must be a positive integer when set.")
    if candidate_min_per_key <= 0:
        raise ValueError("--candidate-min-per-key must be a positive integer.")

    rng = random.Random(seed)
    samples = torch.load(input_path, weights_only=False)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {input_path}")
    samples = apply_semantic_key_mode(samples, semantic_key_mode)

    candidate_fields = candidate_balance_fields or ["semantic_key"]
    candidate_active = (
        candidate_balance_fields is not None
        or candidate_max_per_key is not None
        or candidate_min_per_key != 1
    )
    candidate = samples
    if candidate_active:
        candidate = _sample_balanced(
            samples,
            balance_fields=candidate_fields,
            max_per_key=candidate_max_per_key,
            min_per_key=candidate_min_per_key,
            rng=rng,
            attribute_remap=attribute_remap,
        )

    balanced = _sample_balanced(
        candidate,
        balance_fields=balance_fields,
        max_per_key=max_per_key,
        min_per_key=min_per_key,
        rng=rng,
        attribute_remap=attribute_remap,
    )
    rng.shuffle(balanced)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(balanced, output)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"attribute_remap_fields={sorted(attribute_remap) if attribute_remap else []}")
    _print_stage_summary(
        "input",
        samples,
        balance_fields=balance_fields,
        attribute_remap=attribute_remap,
    )
    if candidate_active:
        print(
            f"candidate_stage balance_fields={candidate_fields} "
            f"min_per_key={candidate_min_per_key} "
            f"max_per_key={candidate_max_per_key if candidate_max_per_key is not None else 'all'}"
        )
        _print_stage_summary(
            "candidate",
            candidate,
            balance_fields=balance_fields,
            attribute_remap=attribute_remap,
            stage_balance_fields=candidate_fields,
        )
    print(
        f"final_stage balance_fields={balance_fields} "
        f"min_per_key={min_per_key} max_per_key={max_per_key}"
    )
    _print_stage_summary(
        "output",
        balanced,
        balance_fields=balance_fields,
        attribute_remap=attribute_remap,
    )


FIELD_ALIASES = {
    "path": "path_richness",
    "delay_spread": "ds_bin",
    "ds": "ds_bin",
    "azimuth_spread": "as_az_bin",
    "as_az": "as_az_bin",
    "k": "k_factor_bin",
    "first_delay": "first_delay_bin",
    "first_power": "first_power_bin",
    "first_angle": "first_angle_bin",
    "reflection": "reflection_bin",
    "diffraction": "diffraction_bin",
}


def _sample_balanced(
    samples,
    balance_fields: list[str],
    max_per_key: int | None,
    min_per_key: int,
    rng: random.Random,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]],
):
    by_key = defaultdict(list)
    for sample in samples:
        by_key[_balance_key(sample, balance_fields, attribute_remap)].append(sample)

    balanced = []
    for key_samples in by_key.values():
        if len(key_samples) < min_per_key:
            continue
        chosen = key_samples[:]
        rng.shuffle(chosen)
        if max_per_key is None:
            balanced.extend(chosen)
        else:
            balanced.extend(chosen[:max_per_key])
    return balanced


def _balance_key(
    sample,
    balance_fields: list[str],
    attribute_remap: dict[str, dict[str, tuple[str, ...]]],
):
    if balance_fields == ["semantic_key"]:
        return sample.semantic_key

    values = []
    for field in balance_fields:
        attr = FIELD_ALIASES.get(field, field)
        if attr == "semantic_key":
            raise ValueError("semantic_key cannot be combined with other balance fields.")
        if attr in semantic_key_field_choices():
            values.append((attr, semantic_key_attribute_value(sample.semantic_key, attr, attribute_remap)))
            continue
        if not hasattr(sample.semantic_key, attr):
            raise ValueError(f"Unknown SemanticKey field for balancing: {field}")
        values.append((attr, getattr(sample.semantic_key, attr)))
    return tuple(values)


def _print_stage_summary(
    stage_name: str,
    samples,
    balance_fields: list[str],
    attribute_remap: dict[str, dict[str, tuple[str, ...]]],
    stage_balance_fields: list[str] | None = None,
) -> None:
    semantic_counts = Counter(sample.semantic_key for sample in samples)
    effective_stage_fields = stage_balance_fields or balance_fields
    stage_balance_counts = Counter(
        _balance_key(sample, effective_stage_fields, attribute_remap)
        for sample in samples
    )
    final_balance_counts = Counter(
        _balance_key(sample, balance_fields, attribute_remap)
        for sample in samples
    )
    print(
        f"{stage_name}_samples={len(samples)} "
        f"{stage_name}_semantic_keys={len(semantic_counts)} "
        f"{stage_name}_stage_balance_keys={len(stage_balance_counts)} "
        f"{stage_name}_final_balance_keys={len(final_balance_counts)}"
    )
    print(f"top 20 {stage_name} stage-balance keys:")
    for key, count in stage_balance_counts.most_common(20):
        print(count, key)
    if effective_stage_fields != balance_fields:
        print(f"top 20 {stage_name} final-balance keys:")
        for key, count in final_balance_counts.most_common(20):
            print(count, key)
    print(f"top 20 {stage_name} semantic keys:")
    for key, count in semantic_counts.most_common(20):
        print(count, key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config. If set, train.attribute_remap is used for attribute balancing.",
    )
    parser.add_argument("--max-per-key", type=int, default=100)
    parser.add_argument("--min-per-key", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-max-per-key",
        type=int,
        default=None,
        help="Optional first-stage cap per candidate group before final balancing.",
    )
    parser.add_argument(
        "--candidate-min-per-key",
        type=int,
        default=1,
        help="Optional first-stage minimum count per candidate group.",
    )
    parser.add_argument(
        "--candidate-balance-fields",
        nargs="+",
        default=None,
        help=(
            "Optional first-stage grouping fields used to build a larger candidate pool, "
            "for example semantic_key or los_status path_richness."
        ),
    )
    parser.add_argument(
        "--balance-fields",
        nargs="+",
        default=["semantic_key"],
        help=(
            "Fields used to form final balancing groups. Use semantic_key for the full key, "
            "or fields like k_factor_bin path_richness k_factor_binary."
        ),
    )
    parser.add_argument(
        "--semantic-key-mode",
        choices=semantic_key_mode_choices(),
        default="full",
        help="Apply semantic-key remapping before balancing and saving the subset.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    make_balanced_subset(
        input_path=args.input,
        output_path=args.output,
        max_per_key=args.max_per_key,
        min_per_key=args.min_per_key,
        seed=args.seed,
        balance_fields=args.balance_fields,
        semantic_key_mode=args.semantic_key_mode,
        candidate_max_per_key=args.candidate_max_per_key,
        candidate_min_per_key=args.candidate_min_per_key,
        candidate_balance_fields=args.candidate_balance_fields,
        attribute_remap=parse_attribute_remap(config.get("attribute_remap")),
    )


if __name__ == "__main__":
    main()
