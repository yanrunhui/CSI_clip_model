from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess_all import _load_stacked_pattern, _path_mask


INTERACTION_DIGITS = {
    "1": "reflection",
    "2": "diffraction",
    "3": "scattering",
}


def _stats(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {}
    return {
        "count": float(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def _print_stats(name: str, values: list[float] | np.ndarray, unit: str = "") -> None:
    stats = _stats(values)
    if not stats:
        print(f"{name}: no finite values")
        return
    suffix = f" {unit}" if unit else ""
    print(
        f"{name}: count={int(stats['count'])} "
        f"mean={stats['mean']:.4g}{suffix} std={stats['std']:.4g}{suffix} "
        f"p10={stats['p10']:.4g}{suffix} p50={stats['p50']:.4g}{suffix} "
        f"p90={stats['p90']:.4g}{suffix} min={stats['min']:.4g}{suffix} max={stats['max']:.4g}{suffix}"
    )


def _interaction_counts(code: float) -> dict[str, int]:
    if not np.isfinite(code):
        return {"reflection": 0, "diffraction": 0, "scattering": 0}
    int_code = int(code)
    if int_code == 0:
        return {"reflection": 0, "diffraction": 0, "scattering": 0}
    digits = str(abs(int_code))
    return {
        name: sum(1 for digit in digits if INTERACTION_DIGITS.get(digit) == name)
        for name in INTERACTION_DIGITS.values()
    }


def _select_first_path(delay_s: np.ndarray, valid_mask: np.ndarray, mode: str) -> int | None:
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return None
    if mode == "index":
        return int(valid_indices[0])
    return int(valid_indices[np.argmin(delay_s[valid_indices])])


def analyze_paths(
    scenario_dir: Path,
    max_samples: int | None,
    first_path_mode: str,
    output_csv: Path | None,
) -> None:
    power = _load_stacked_pattern(scenario_dir, "power", max_rows=max_samples)
    n_rows = int(power.shape[0])
    delay = _load_stacked_pattern(scenario_dir, "delay", max_rows=n_rows)
    aoa_az = _load_stacked_pattern(scenario_dir, "aoa_az", max_rows=n_rows)
    aoa_el = _load_stacked_pattern(scenario_dir, "aoa_el", max_rows=n_rows)
    aod_az = _load_stacked_pattern(scenario_dir, "aod_az", max_rows=n_rows)
    aod_el = _load_stacked_pattern(scenario_dir, "aod_el", max_rows=n_rows)
    inter = _load_stacked_pattern(scenario_dir, "inter", max_rows=n_rows)

    rows = []
    path_counts = []
    first_delay_ns = []
    first_power_dbw = []
    first_power_share = []
    first_aoa_az = []
    first_aoa_el = []
    first_aod_az = []
    first_aod_el = []
    los_delay_ns = []
    los_power_dbw = []
    los_power_share = []
    los_aoa_az = []
    los_aoa_el = []
    los_aod_az = []
    los_aod_el = []
    per_sample_reflections = []
    per_sample_diffractions = []
    per_sample_scatterings = []
    first_reflections = []
    first_diffractions = []
    first_scatterings = []
    inter_code_counter: Counter[int] = Counter()
    los_samples = 0
    valid_samples = 0

    for idx in range(n_rows):
        p_dbw = np.asarray(power[idx]).reshape(-1)
        d_s = np.asarray(delay[idx]).reshape(-1)
        aa = np.asarray(aoa_az[idx]).reshape(-1)
        ae = np.asarray(aoa_el[idx]).reshape(-1)
        da = np.asarray(aod_az[idx]).reshape(-1)
        de = np.asarray(aod_el[idx]).reshape(-1)
        inter_code = np.asarray(inter[idx]).reshape(-1)
        valid = _path_mask(p_dbw, d_s)
        if not bool(valid.any()):
            continue

        valid_samples += 1
        path_counts.append(int(valid.sum()))
        power_linear = np.power(10.0, p_dbw[valid] / 10.0)
        power_sum = float(power_linear.sum())

        sample_inter_counts = {"reflection": 0, "diffraction": 0, "scattering": 0}
        for code in inter_code[valid]:
            if np.isfinite(code):
                inter_code_counter[int(code)] += 1
            counts = _interaction_counts(float(code))
            for name, count in counts.items():
                sample_inter_counts[name] += count
        per_sample_reflections.append(sample_inter_counts["reflection"])
        per_sample_diffractions.append(sample_inter_counts["diffraction"])
        per_sample_scatterings.append(sample_inter_counts["scattering"])

        first_idx = _select_first_path(d_s, valid, mode=first_path_mode)
        assert first_idx is not None
        first_counts = _interaction_counts(float(inter_code[first_idx]))
        first_reflections.append(first_counts["reflection"])
        first_diffractions.append(first_counts["diffraction"])
        first_scatterings.append(first_counts["scattering"])
        first_delay_ns.append(float(d_s[first_idx] * 1e9))
        first_power_dbw.append(float(p_dbw[first_idx]))
        first_aoa_az.append(float(aa[first_idx]))
        first_aoa_el.append(float(ae[first_idx]))
        first_aod_az.append(float(da[first_idx]))
        first_aod_el.append(float(de[first_idx]))
        first_power = float(10.0 ** (p_dbw[first_idx] / 10.0))
        first_power_share.append(first_power / max(power_sum, 1e-30))

        los_indices = np.where(valid & (inter_code == 0))[0]
        has_los = bool(los_indices.size)
        if has_los:
            los_samples += 1
            los_idx = int(los_indices[np.argmin(d_s[los_indices])])
            los_delay_ns.append(float(d_s[los_idx] * 1e9))
            los_power_dbw.append(float(p_dbw[los_idx]))
            los_aoa_az.append(float(aa[los_idx]))
            los_aoa_el.append(float(ae[los_idx]))
            los_aod_az.append(float(da[los_idx]))
            los_aod_el.append(float(de[los_idx]))
            los_power = float(10.0 ** (p_dbw[los_idx] / 10.0))
            los_power_share.append(los_power / max(power_sum, 1e-30))

        rows.append(
            {
                "sample_idx": idx,
                "num_paths": int(valid.sum()),
                "has_los_path": int(has_los),
                "first_path_index": first_idx,
                "first_path_delay_ns": first_delay_ns[-1],
                "first_path_power_dbw": first_power_dbw[-1],
                "first_path_power_share": first_power_share[-1],
                "first_path_aoa_az_deg": first_aoa_az[-1],
                "first_path_aoa_el_deg": first_aoa_el[-1],
                "first_path_aod_az_deg": first_aod_az[-1],
                "first_path_aod_el_deg": first_aod_el[-1],
                "first_path_inter_code": int(inter_code[first_idx]) if np.isfinite(inter_code[first_idx]) else "",
                "first_path_reflections": first_counts["reflection"],
                "first_path_diffractions": first_counts["diffraction"],
                "first_path_scatterings": first_counts["scattering"],
                "sample_total_reflections": sample_inter_counts["reflection"],
                "sample_total_diffractions": sample_inter_counts["diffraction"],
                "sample_total_scatterings": sample_inter_counts["scattering"],
            }
        )

    print(f"scenario_dir={scenario_dir}")
    print(f"loaded_samples={n_rows}")
    print(f"valid_samples={valid_samples}")
    print(f"los_samples={los_samples}")
    print(f"los_sample_ratio={los_samples / max(valid_samples, 1):.4f}")
    print("\nPath count")
    _print_stats("num_paths", path_counts)
    print("\nFirst path statistics")
    _print_stats("first_delay", first_delay_ns, "ns")
    _print_stats("first_power", first_power_dbw, "dBW")
    _print_stats("first_power_share", first_power_share)
    _print_stats("first_aoa_az", first_aoa_az, "deg")
    _print_stats("first_aoa_el", first_aoa_el, "deg")
    _print_stats("first_aod_az", first_aod_az, "deg")
    _print_stats("first_aod_el", first_aod_el, "deg")
    print("\nLoS path statistics")
    _print_stats("los_delay", los_delay_ns, "ns")
    _print_stats("los_power", los_power_dbw, "dBW")
    _print_stats("los_power_share", los_power_share)
    _print_stats("los_aoa_az", los_aoa_az, "deg")
    _print_stats("los_aoa_el", los_aoa_el, "deg")
    _print_stats("los_aod_az", los_aod_az, "deg")
    _print_stats("los_aod_el", los_aod_el, "deg")
    print("\nInteraction counts per sample, summed over valid paths")
    _print_stats("sample_total_reflections", per_sample_reflections)
    _print_stats("sample_total_diffractions", per_sample_diffractions)
    _print_stats("sample_total_scatterings", per_sample_scatterings)
    print("\nInteraction counts on first path")
    _print_stats("first_path_reflections", first_reflections)
    _print_stats("first_path_diffractions", first_diffractions)
    _print_stats("first_path_scatterings", first_scatterings)
    print("\nTop interaction codes")
    for code, count in inter_code_counter.most_common(30):
        counts = _interaction_counts(float(code))
        print(
            f"{count} code={code} "
            f"reflection={counts['reflection']} diffraction={counts['diffraction']} "
            f"scattering={counts['scattering']}"
        )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote per-sample CSV to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--scenario-root", type=str, default=str(ROOT / "Raytracing_scenarios"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--first-path-mode", choices=["min-delay", "index"], default="min-delay")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_root) / args.scenario
    analyze_paths(
        scenario_dir=scenario_dir,
        max_samples=args.max_samples,
        first_path_mode=args.first_path_mode,
        output_csv=Path(args.output_csv) if args.output_csv else None,
    )


if __name__ == "__main__":
    main()
