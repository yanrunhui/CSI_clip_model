from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _print_counter(title: str, counts: Counter, total: int, top_k: int | None = None) -> None:
    print(title)
    if not counts:
        print("  <empty>")
        return

    items = counts.most_common(top_k)
    for key, value in items:
        ratio = 100.0 * value / max(total, 1)
        print(f"  {key}: {value} ({ratio:.2f}%)")


def diagnose_path_distribution(
    input_path: str,
    samples_per_key: int,
    num_keys_per_batch: int,
    top_k: int,
) -> None:
    samples = torch.load(input_path, weights_only=False)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {input_path}")

    total = len(samples)
    path_bins = Counter()
    los_bins = Counter()
    delay_bins = Counter()
    reflection_bins = Counter()
    diffraction_bins = Counter()
    config_bins = Counter()
    path_los = Counter()
    path_delay = Counter()
    path_reflection = Counter()
    full_combo = Counter()
    path_to_indices = defaultdict(list)

    exact_path_counts = Counter()
    has_exact_path_count = False

    for idx, sample in enumerate(samples):
        key = sample.semantic_key
        path_bins[key.path_richness] += 1
        los_bins[key.los_status] += 1
        delay_bins[key.ds_bin] += 1
        reflection_bins[key.reflection_bin] += 1
        diffraction_bins[key.diffraction_bin] += 1
        config_bins[sample.config_key] += 1
        path_los[(key.path_richness, key.los_status)] += 1
        path_delay[(key.path_richness, key.ds_bin)] += 1
        path_reflection[(key.path_richness, key.reflection_bin)] += 1
        full_combo[
            (
                key.path_richness,
                key.los_status,
                key.ds_bin,
                key.reflection_bin,
                key.diffraction_bin,
            )
        ] += 1
        path_to_indices[key.path_richness].append(idx)

        if hasattr(sample, "num_paths"):
            exact_path_counts[int(sample.num_paths)] += 1
            has_exact_path_count = True

    valid_keys = {key: len(indices) for key, indices in path_to_indices.items() if len(indices) >= samples_per_key}
    total_chunks = sum(len(indices) // samples_per_key for indices in path_to_indices.values())
    estimated_batches = (total_chunks + num_keys_per_batch - 1) // max(num_keys_per_batch, 1)

    print(f"input={input_path}")
    print(f"num_samples={total}")
    print(
        f"samples_per_key={samples_per_key} num_keys_per_batch={num_keys_per_batch} "
        f"valid_n_paths_keys={len(valid_keys)} estimated_sampler_batches={estimated_batches}"
    )
    print()

    if has_exact_path_count:
        _print_counter("Exact Path Counts", exact_path_counts, total, top_k=top_k)
        print()
    else:
        print("Exact path counts are not stored in the current preprocessed .pt samples.")
        print("Showing `semantic_key.path_richness` bins instead.")
        print()

    _print_counter("Path Richness Bins", path_bins, total)
    print()
    _print_counter("LoS Bins", los_bins, total)
    print()
    _print_counter("Delay Spread Bins", delay_bins, total)
    print()
    _print_counter("Reflection Bins", reflection_bins, total)
    print()
    _print_counter("Diffraction Bins", diffraction_bins, total)
    print()
    _print_counter("Config Keys", config_bins, total)
    print()
    _print_counter("Path Richness x LoS", path_los, total, top_k=top_k)
    print()
    _print_counter("Path Richness x Delay Spread", path_delay, total, top_k=top_k)
    print()
    _print_counter("Path Richness x Reflection", path_reflection, total, top_k=top_k)
    print()
    _print_counter("Top Full Semantic Combos", full_combo, total, top_k=top_k)
    print()
    print("n_paths keys usable by SemanticKeyBatchSampler:")
    for key, count in sorted(valid_keys.items(), key=lambda item: (-item[1], item[0])):
        chunks = count // samples_per_key
        print(f"  {key}: {count} samples -> {chunks} chunks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to preprocessed .pt samples.")
    parser.add_argument("--samples-per-key", type=int, default=4)
    parser.add_argument("--num-keys-per-batch", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=12, help="How many combined keys to print.")
    args = parser.parse_args()
    diagnose_path_distribution(
        input_path=args.input,
        samples_per_key=args.samples_per_key,
        num_keys_per_batch=args.num_keys_per_batch,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
