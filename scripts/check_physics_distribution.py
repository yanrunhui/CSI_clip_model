import os
import sys
import argparse
import math
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.dataset import PreprocessedCSIDataset


PHYSICS_FIELDS = [
    "k_factor_db",
    "delay_spread_ns",
    "azimuth_spread_deg",
    "first_path_power_dbw",
]


def finite(x):
    try:
        x = float(x)
        return math.isfinite(x)
    except Exception:
        return False


def get_value(sample, field):
    if hasattr(sample, field):
        return getattr(sample, field)

    if field == "delay_spread_ns":
        if hasattr(sample, "delay_spread_ns"):
            return getattr(sample, "delay_spread_ns")
        if hasattr(sample, "delay_spread"):
            return float(getattr(sample, "delay_spread")) * 1e9

    if field == "azimuth_spread_deg":
        if hasattr(sample, "azimuth_spread_deg"):
            return getattr(sample, "azimuth_spread_deg")
        if hasattr(sample, "azimuth_spread"):
            return float(getattr(sample, "azimuth_spread")) * 180.0 / math.pi
        if hasattr(sample, "azimuth_spread_aoa"):
            return float(getattr(sample, "azimuth_spread_aoa")) * 180.0 / math.pi

    return None


def summarize(name, values):
    values = [float(v) for v in values if finite(v)]

    print(f"\n=== {name} ===")
    print("count =", len(values))

    if not values:
        print("No valid values.")
        return

    arr = np.array(values, dtype=np.float64)

    print("min =", float(np.min(arr)))
    print("max =", float(np.max(arr)))
    print("mean =", float(np.mean(arr)))
    print("std =", float(np.std(arr)))
    print("p05 =", float(np.percentile(arr, 5)))
    print("p50 =", float(np.percentile(arr, 50)))
    print("p95 =", float(np.percentile(arr, 95)))
    print("unique_count_rounded_3 =", len(set(round(float(v), 3) for v in arr)))
    print("first_20 =", [round(float(v), 3) for v in arr[:20]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    args = parser.parse_args()

    dataset = PreprocessedCSIDataset.from_pt(args.data_path)
    samples = dataset.samples

    print("data_path =", args.data_path)
    print("num_samples =", len(samples))

    for field in PHYSICS_FIELDS:
        vals = []
        for s in samples:
            v = get_value(s, field)
            if finite(v):
                vals.append(float(v))
        summarize(f"overall raw {field}", vals)

    groups = defaultdict(list)
    for s in samples:
        if not hasattr(s, "semantic_key"):
            continue

        k_bin = getattr(s.semantic_key, "k_factor_bin", "unknown")
        raw_k = get_value(s, "k_factor_db")

        if finite(raw_k):
            groups[k_bin].append(float(raw_k))

    print("\n\n========== k_factor_db grouped by semantic_key.k_factor_bin ==========")
    for k_bin, vals in sorted(groups.items()):
        summarize(f"k_factor_bin={k_bin} raw k_factor_db", vals)

    power_groups = defaultdict(list)
    for s in samples:
        if not hasattr(s, "semantic_key"):
            continue

        p_bin = getattr(s.semantic_key, "first_power_bin", "unknown")
        raw_p = get_value(s, "first_path_power_dbw")

        if finite(raw_p):
            power_groups[p_bin].append(float(raw_p))

    print("\n\n========== first_path_power_dbw grouped by semantic_key.first_power_bin ==========")
    for p_bin, vals in sorted(power_groups.items()):
        summarize(f"first_power_bin={p_bin} raw first_path_power_dbw", vals)

    key_counts = defaultdict(int)
    for s in samples:
        if hasattr(s, "semantic_key"):
            key_counts[s.semantic_key] += 1

    print("\n\n========== semantic_key / coarse_k class sizes ==========")
    print("num_classes =", len(key_counts))
    for key, count in sorted(key_counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{count:6d}  {key}")


if __name__ == "__main__":
    main()
