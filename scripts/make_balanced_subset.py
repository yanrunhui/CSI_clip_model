from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_balanced_subset(
    input_path: str,
    output_path: str,
    max_per_key: int,
    min_per_key: int,
    seed: int,
    balance_fields: list[str],
) -> None:
    rng = random.Random(seed)
    samples = torch.load(input_path, weights_only=False)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {input_path}")

    by_key = defaultdict(list)
    for sample in samples:
        by_key[_balance_key(sample, balance_fields)].append(sample)

    balanced = []
    for key, key_samples in by_key.items():
        if len(key_samples) < min_per_key:
            continue
        chosen = key_samples[:]
        rng.shuffle(chosen)
        balanced.extend(chosen[:max_per_key])
    rng.shuffle(balanced)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(balanced, output)

    before = Counter(sample.semantic_key for sample in samples)
    after = Counter(sample.semantic_key for sample in balanced)
    before_balance = Counter(_balance_key(sample, balance_fields) for sample in samples)
    after_balance = Counter(_balance_key(sample, balance_fields) for sample in balanced)
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(
        f"balance_fields={balance_fields} "
        f"input_samples={len(samples)} input_balance_keys={len(before_balance)} "
        f"input_semantic_keys={len(before)}"
    )
    print(
        f"output_samples={len(balanced)} output_balance_keys={len(after_balance)} "
        f"output_semantic_keys={len(after)} "
        f"min_per_key={min_per_key} max_per_key={max_per_key}"
    )
    print("top 20 balance keys before:")
    for key, count in before_balance.most_common(20):
        print(count, key)
    print("top 20 balance keys after:")
    for key, count in after_balance.most_common(20):
        print(count, key)
    print("top 20 semantic keys before:")
    for key, count in before.most_common(20):
        print(count, key)
    print("top 20 semantic keys after:")
    for key, count in after.most_common(20):
        print(count, key)


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


def _balance_key(sample, balance_fields: list[str]):
    if balance_fields == ["semantic_key"]:
        return sample.semantic_key

    values = []
    for field in balance_fields:
        attr = FIELD_ALIASES.get(field, field)
        if not hasattr(sample.semantic_key, attr):
            raise ValueError(f"Unknown SemanticKey field for balancing: {field}")
        values.append((attr, getattr(sample.semantic_key, attr)))
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-key", type=int, default=100)
    parser.add_argument("--min-per-key", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--balance-fields",
        nargs="+",
        default=["semantic_key"],
        help=(
            "Fields used to form balancing groups. Use semantic_key for the full key, "
            "or fields like first_delay_bin reflection_bin."
        ),
    )
    args = parser.parse_args()
    make_balanced_subset(
        input_path=args.input,
        output_path=args.output,
        max_per_key=args.max_per_key,
        min_per_key=args.min_per_key,
        seed=args.seed,
        balance_fields=args.balance_fields,
    )


if __name__ == "__main__":
    main()
