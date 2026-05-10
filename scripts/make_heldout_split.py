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


def stratify_key(sample, stratify_fields: list[str], attribute_remap: dict[str, dict[str, tuple[str, ...]]]):
    if stratify_fields == ["semantic_key"]:
        return sample.semantic_key

    values = []
    for field in stratify_fields:
        attr = FIELD_ALIASES.get(field, field)
        if attr == "semantic_key":
            raise ValueError("semantic_key cannot be combined with other stratify fields.")
        if attr in semantic_key_field_choices():
            values.append((attr, semantic_key_attribute_value(sample.semantic_key, attr, attribute_remap)))
            continue
        if not hasattr(sample.semantic_key, attr):
            raise ValueError(f"Unknown SemanticKey field for stratification: {field}")
        values.append((attr, getattr(sample.semantic_key, attr)))
    return tuple(values)


def summarize(stage: str, samples, stratify_fields: list[str], attribute_remap: dict[str, dict[str, tuple[str, ...]]]) -> None:
    semantic_counts = Counter(sample.semantic_key for sample in samples)
    stratify_counts = Counter(stratify_key(sample, stratify_fields, attribute_remap) for sample in samples)
    print(
        f"{stage}_samples={len(samples)} "
        f"{stage}_semantic_keys={len(semantic_counts)} "
        f"{stage}_strata={len(stratify_counts)}"
    )
    print(f"top 20 {stage} strata:")
    for key, count in stratify_counts.most_common(20):
        print(count, key)


def make_heldout_split(
    input_path: str,
    train_output_path: str,
    test_output_path: str,
    semantic_key_mode: str,
    stratify_fields: list[str],
    test_fraction: float,
    seed: int,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]],
) -> None:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")

    rng = random.Random(seed)
    samples = torch.load(input_path, weights_only=False)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {input_path}")
    samples = apply_semantic_key_mode(samples, semantic_key_mode)

    by_key = defaultdict(list)
    for sample in samples:
        by_key[stratify_key(sample, stratify_fields, attribute_remap)].append(sample)

    train_samples = []
    test_samples = []
    singleton_strata = 0
    for key_samples in by_key.values():
        chosen = key_samples[:]
        rng.shuffle(chosen)
        if len(chosen) == 1:
            singleton_strata += 1
            train_samples.extend(chosen)
            continue
        test_count = int(round(len(chosen) * test_fraction))
        test_count = min(max(test_count, 1), len(chosen) - 1)
        test_samples.extend(chosen[:test_count])
        train_samples.extend(chosen[test_count:])

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    train_output = Path(train_output_path)
    test_output = Path(test_output_path)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_samples, train_output)
    torch.save(test_samples, test_output)

    print(f"input={input_path}")
    print(f"train_output={train_output_path}")
    print(f"test_output={test_output_path}")
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"stratify_fields={stratify_fields}")
    print(f"test_fraction={test_fraction}")
    print(f"seed={seed}")
    print(f"attribute_remap_fields={sorted(attribute_remap) if attribute_remap else []}")
    print(f"singleton_strata_routed_to_train={singleton_strata}")
    summarize("input", samples, stratify_fields, attribute_remap)
    summarize("train", train_samples, stratify_fields, attribute_remap)
    summarize("test", test_samples, stratify_fields, attribute_remap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument(
        "--semantic-key-mode",
        choices=semantic_key_mode_choices(),
        default="full",
        help="Apply semantic remapping before splitting.",
    )
    parser.add_argument(
        "--stratify-fields",
        nargs="+",
        default=["semantic_key"],
        help=(
            "Fields used to stratify the split. Use `semantic_key` for full-key stratification, "
            "or fields like `path_richness`, `k_factor_bin`, `first_power_bin`."
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config. If set, train.attribute_remap is used for stratification labels.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    attribute_remap = parse_attribute_remap(config.get("attribute_remap"))
    make_heldout_split(
        input_path=args.input,
        train_output_path=args.train_output,
        test_output_path=args.test_output,
        semantic_key_mode=args.semantic_key_mode,
        stratify_fields=args.stratify_fields,
        test_fraction=args.test_fraction,
        seed=args.seed,
        attribute_remap=attribute_remap,
    )


if __name__ == "__main__":
    main()
