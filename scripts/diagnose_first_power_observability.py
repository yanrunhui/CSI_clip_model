from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (
    DELAY_POWER_MAP_SHAPE,
    DELAY_POWER_PROFILE_BINS,
    PreprocessedCSIDataset,
)
from scripts.preprocess_all import DELAY_POWER_MAP_POWER_DBW_RANGE

DELAY_NS_RANGE = (0.0, 3000.0)
POWER_DBW_RANGE = DELAY_POWER_MAP_POWER_DBW_RANGE
CANDIDATE_POWER_DBW_RANGES = (
    (-240.0, -40.0),
    (-230.0, -40.0),
    (-220.0, -40.0),
    (-200.0, -40.0),
    (-180.0, -40.0),
)


def _finite_float(value: float | int) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _clip_bin(value: float, lower: float, upper: float, bins: int) -> int:
    clipped = min(max(value, lower), math.nextafter(upper, lower))
    idx = int(math.floor((clipped - lower) / max(upper - lower, 1e-12) * bins))
    return min(max(idx, 0), bins - 1)


def _bin_center(idx: torch.Tensor, lower: float, upper: float, bins: int) -> torch.Tensor:
    width = (upper - lower) / max(bins, 1)
    return lower + (idx.to(dtype=torch.float32) + 0.5) * width


def _safe_log10_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10((num.float() + 1e-12) / (den.float() + 1e-12))


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 2:
        return float("nan")
    x = x[mask].float()
    y = y[mask].float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) == 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def _range_suffix(lower: float, upper: float) -> str:
    return (
        f"{int(abs(lower)) if lower < 0 else int(lower)}"
        f"to"
        f"{int(abs(upper)) if upper < 0 else int(upper)}"
    )


def _print_stats(prefix: str, values: torch.Tensor) -> None:
    values = values[torch.isfinite(values)]
    print(f"{prefix}_count={int(values.numel())}")
    if values.numel() == 0:
        print(f"{prefix}_mean=nan")
        print(f"{prefix}_median=nan")
        print(f"{prefix}_p90=nan")
        print(f"{prefix}_min=nan")
        print(f"{prefix}_max=nan")
        return
    print(f"{prefix}_mean={float(values.mean()):.4f}")
    print(f"{prefix}_median={float(values.median()):.4f}")
    print(f"{prefix}_p90={float(torch.quantile(values.float(), 0.9)):.4f}")
    print(f"{prefix}_min={float(values.min()):.4f}")
    print(f"{prefix}_max={float(values.max()):.4f}")


def _los_matches(sample, los_status: str) -> bool:
    sample_status = getattr(sample.semantic_key, "los_status", "")
    if los_status == "all":
        return True
    if los_status == "los":
        return sample_status == "los"
    return sample_status != "los"


def diagnose(
    data_path: str,
    los_status: str,
    limit_samples: int | None,
    examples: int,
) -> None:
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    samples = [
        (idx, sample)
        for idx, sample in enumerate(dataset.samples)
        if _los_matches(sample, los_status)
    ]
    if limit_samples is not None:
        samples = samples[:limit_samples]
    if not samples:
        raise ValueError(f"No samples matched los_status={los_status!r}.")

    rows: list[dict[str, float | int | str]] = []
    for sample_idx, sample in samples:
        target_power = _finite_float(getattr(sample, "first_path_power_dbw", math.nan))
        first_delay_ns = _finite_float(getattr(sample, "first_path_delay_s", math.nan)) * 1e9
        if not math.isfinite(target_power) or not math.isfinite(first_delay_ns):
            continue

        profile = sample.delay_power_profile.float().clamp(min=0.0)
        profile_sum = float(profile.sum())
        if profile_sum > 0.0:
            profile = profile / profile_sum
        profile_peak_idx = int(profile.argmax().item())
        profile_first_idx = _clip_bin(
            first_delay_ns,
            DELAY_NS_RANGE[0],
            DELAY_NS_RANGE[1],
            DELAY_POWER_PROFILE_BINS,
        )
        profile_first_mass = float(profile[profile_first_idx])
        profile_peak_mass = float(profile[profile_peak_idx])
        profile_first_rank = int((profile > profile[profile_first_idx]).sum().item()) + 1

        delay_map = sample.delay_power_map.float().clamp(min=0.0)
        map_sum = float(delay_map.sum())
        if map_sum > 0.0:
            delay_map = delay_map / map_sum
        map_flat_peak = int(delay_map.flatten().argmax().item())
        map_peak_delay_idx = map_flat_peak // DELAY_POWER_MAP_SHAPE[1]
        map_peak_power_idx = map_flat_peak % DELAY_POWER_MAP_SHAPE[1]
        map_first_delay_idx = _clip_bin(
            first_delay_ns,
            DELAY_NS_RANGE[0],
            DELAY_NS_RANGE[1],
            DELAY_POWER_MAP_SHAPE[0],
        )
        target_power_idx = _clip_bin(
            target_power,
            POWER_DBW_RANGE[0],
            POWER_DBW_RANGE[1],
            DELAY_POWER_MAP_SHAPE[1],
        )
        first_delay_row = delay_map[map_first_delay_idx]
        row_peak_power_idx = int(first_delay_row.argmax().item())
        row_peak_mass = float(first_delay_row[row_peak_power_idx])
        target_cell_mass = float(delay_map[map_first_delay_idx, target_power_idx])

        rows.append(
            {
                "sample_idx": sample_idx,
                "target_power": target_power,
                "first_delay_ns": first_delay_ns,
                "profile_sum": profile_sum,
                "profile_first_idx": profile_first_idx,
                "profile_peak_idx": profile_peak_idx,
                "profile_first_mass": profile_first_mass,
                "profile_peak_mass": profile_peak_mass,
                "profile_first_rank": profile_first_rank,
                "map_sum": map_sum,
                "map_first_delay_idx": map_first_delay_idx,
                "map_peak_delay_idx": map_peak_delay_idx,
                "map_peak_power_idx": map_peak_power_idx,
                "target_power_idx": target_power_idx,
                "row_peak_power_idx": row_peak_power_idx,
                "row_peak_mass": row_peak_mass,
                "target_cell_mass": target_cell_mass,
                "semantic_key": str(sample.semantic_key),
            }
        )

    if not rows:
        raise ValueError("No valid first_path_power_dbw / first_path_delay_s targets found.")

    def tensor(name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor([row[name] for row in rows], dtype=dtype)

    target_power = tensor("target_power")
    first_delay_ns = tensor("first_delay_ns")
    profile_first_idx = tensor("profile_first_idx", dtype=torch.long)
    profile_peak_idx = tensor("profile_peak_idx", dtype=torch.long)
    profile_first_mass = tensor("profile_first_mass")
    profile_peak_mass = tensor("profile_peak_mass")
    profile_first_rank = tensor("profile_first_rank")
    map_first_delay_idx = tensor("map_first_delay_idx", dtype=torch.long)
    map_peak_delay_idx = tensor("map_peak_delay_idx", dtype=torch.long)
    map_peak_power_idx = tensor("map_peak_power_idx", dtype=torch.long)
    target_power_idx = tensor("target_power_idx", dtype=torch.long)
    row_peak_power_idx = tensor("row_peak_power_idx", dtype=torch.long)
    row_peak_mass = tensor("row_peak_mass")
    target_cell_mass = tensor("target_cell_mass")

    profile_peak_delay_ns = _bin_center(
        profile_peak_idx,
        DELAY_NS_RANGE[0],
        DELAY_NS_RANGE[1],
        DELAY_POWER_PROFILE_BINS,
    )
    map_peak_power_dbw = _bin_center(
        map_peak_power_idx,
        POWER_DBW_RANGE[0],
        POWER_DBW_RANGE[1],
        DELAY_POWER_MAP_SHAPE[1],
    )
    row_peak_power_dbw = _bin_center(
        row_peak_power_idx,
        POWER_DBW_RANGE[0],
        POWER_DBW_RANGE[1],
        DELAY_POWER_MAP_SHAPE[1],
    )
    target_power_bin_center = _bin_center(
        target_power_idx,
        POWER_DBW_RANGE[0],
        POWER_DBW_RANGE[1],
        DELAY_POWER_MAP_SHAPE[1],
    )

    profile_delay_bin_abs_diff = (profile_peak_idx - profile_first_idx).abs().float()
    map_delay_bin_abs_diff = (map_peak_delay_idx - map_first_delay_idx).abs().float()
    profile_first_vs_peak_db = _safe_log10_ratio(profile_first_mass, profile_peak_mass)
    target_cell_vs_row_peak_db = _safe_log10_ratio(target_cell_mass, row_peak_mass)

    print(f"data_path={data_path}")
    print(f"los_status={los_status}")
    print(f"samples_requested={len(samples)}")
    print(f"valid_samples={len(rows)}")
    print("delay_power_profile_note=relative_power_distribution_normalized_per_sample")
    print("delay_power_map_note=relative_power_mass_over_delay_bin_and_raw_power_bin")
    _print_stats("target_first_path_power_dbw", target_power)
    _print_stats("first_path_delay_ns", first_delay_ns)
    print(
        "profile_first_delay_bin_is_peak_rate="
        f"{float((profile_first_idx == profile_peak_idx).float().mean()):.4f}"
    )
    print(
        "profile_first_delay_bin_within_1_of_peak_rate="
        f"{float((profile_delay_bin_abs_diff <= 1).float().mean()):.4f}"
    )
    _print_stats("profile_peak_minus_first_delay_bin_abs", profile_delay_bin_abs_diff)
    _print_stats("profile_first_delay_mass", profile_first_mass)
    _print_stats("profile_peak_mass", profile_peak_mass)
    _print_stats("profile_first_delay_mass_vs_peak_db", profile_first_vs_peak_db)
    _print_stats("profile_first_delay_mass_rank", profile_first_rank)
    print(
        "map_first_delay_bin_is_global_peak_delay_rate="
        f"{float((map_first_delay_idx == map_peak_delay_idx).float().mean()):.4f}"
    )
    print(
        "map_first_delay_bin_within_1_of_global_peak_delay_rate="
        f"{float((map_delay_bin_abs_diff <= 1).float().mean()):.4f}"
    )
    _print_stats("map_peak_minus_first_delay_bin_abs", map_delay_bin_abs_diff)
    print(
        "map_target_power_bin_is_global_peak_power_rate="
        f"{float((target_power_idx == map_peak_power_idx).float().mean()):.4f}"
    )
    print(
        "map_target_power_bin_is_first_delay_row_peak_power_rate="
        f"{float((target_power_idx == row_peak_power_idx).float().mean()):.4f}"
    )
    _print_stats("map_target_cell_mass", target_cell_mass)
    _print_stats("map_first_delay_row_peak_mass", row_peak_mass)
    _print_stats("map_target_cell_mass_vs_first_delay_row_peak_db", target_cell_vs_row_peak_db)

    global_peak_power_error = (map_peak_power_dbw - target_power).abs()
    row_peak_power_error = (row_peak_power_dbw - target_power).abs()
    target_bin_center_error = (target_power_bin_center - target_power).abs()
    _print_stats("map_global_peak_power_bin_center_abs_error_db", global_peak_power_error)
    _print_stats("map_first_delay_row_peak_power_bin_center_abs_error_db", row_peak_power_error)
    _print_stats("target_power_bin_quantization_abs_error_db", target_bin_center_error)
    print(f"pearson_target_vs_profile_first_delay_mass={_safe_pearson(target_power, profile_first_mass):.4f}")
    print(f"pearson_target_vs_profile_peak_delay_ns={_safe_pearson(target_power, profile_peak_delay_ns):.4f}")
    print(f"pearson_target_vs_map_global_peak_power_dbw={_safe_pearson(target_power, map_peak_power_dbw):.4f}")
    print(f"pearson_target_vs_map_first_delay_row_peak_power_dbw={_safe_pearson(target_power, row_peak_power_dbw):.4f}")
    print(f"target_power_clipped_low_count={int((target_power < POWER_DBW_RANGE[0]).sum().item())}")
    print(f"target_power_clipped_high_count={int((target_power >= POWER_DBW_RANGE[1]).sum().item())}")
    for candidate_lower, candidate_upper in CANDIDATE_POWER_DBW_RANGES:
        candidate_bins = torch.tensor(
            [
                _clip_bin(
                    float(value),
                    candidate_lower,
                    candidate_upper,
                    DELAY_POWER_MAP_SHAPE[1],
                )
                for value in target_power.tolist()
            ],
            dtype=torch.long,
        )
        candidate_centers = _bin_center(
            candidate_bins,
            candidate_lower,
            candidate_upper,
            DELAY_POWER_MAP_SHAPE[1],
        )
        candidate_error = (candidate_centers - target_power).abs()
        range_prefix = (
            f"candidate_power_range_{_range_suffix(candidate_lower, candidate_upper)}"
        )
        print(
            f"{range_prefix}_clipped_low_count="
            f"{int((target_power < candidate_lower).sum().item())}"
        )
        print(
            f"{range_prefix}_clipped_high_count="
            f"{int((target_power >= candidate_upper).sum().item())}"
        )
        _print_stats(f"{range_prefix}_quantization_abs_error_db", candidate_error)

    if examples > 0:
        worst = torch.argsort(row_peak_power_error, descending=True)[:examples]
        for rank, row_idx_tensor in enumerate(worst, start=1):
            row_idx = int(row_idx_tensor.item())
            row = rows[row_idx]
            print(
                f"worst_row_peak_power_example_{rank}="
                f"sample_idx:{row['sample_idx']} "
                f"target_dbw:{float(row['target_power']):.4f} "
                f"first_delay_ns:{float(row['first_delay_ns']):.4f} "
                f"profile_first_bin:{int(row['profile_first_idx'])} "
                f"profile_peak_bin:{int(row['profile_peak_idx'])} "
                f"profile_first_mass:{float(row['profile_first_mass']):.6f} "
                f"profile_peak_mass:{float(row['profile_peak_mass']):.6f} "
                f"map_first_delay_bin:{int(row['map_first_delay_idx'])} "
                f"target_power_bin:{int(row['target_power_idx'])} "
                f"row_peak_power_bin:{int(row['row_peak_power_idx'])} "
                f"row_peak_power_dbw:{float(row_peak_power_dbw[row_idx]):.4f} "
                f"target_cell_mass:{float(row['target_cell_mass']):.6f} "
                f"row_peak_mass:{float(row['row_peak_mass']):.6f} "
                f"key:{row['semantic_key']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--los-status", choices=("all", "los", "nlos"), default="nlos")
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    diagnose(
        data_path=args.data_path,
        los_status=args.los_status,
        limit_samples=args.limit_samples,
        examples=args.examples,
    )


if __name__ == "__main__":
    main()
