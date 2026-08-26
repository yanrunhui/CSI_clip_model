from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess_all import _iter_d2los_propbin_files, read_rayverse_propbin


def audit_los_delay_relation(
    d2los_root: Path,
    *,
    max_maps: int | None,
    max_sources_per_map: int | None,
    tolerance_ns: float,
) -> dict[str, int | float]:
    files = _iter_d2los_propbin_files(
        d2los_root,
        max_maps=max_maps,
        max_sources_per_map=max_sources_per_map,
    )
    if not files:
        raise FileNotFoundError(f"No RayVerse propbin files found under {d2los_root}")

    valid_links = 0
    los_links = 0
    equal_links = 0
    first_before_direct = 0
    first_after_direct = 0
    max_absolute_difference_ns = 0.0

    for path in files:
        propbin = read_rayverse_propbin(path)
        for rx_record in propbin.rx_records:
            path_count = int(rx_record["path_count"])
            if path_count <= 0:
                continue
            path_offset = int(rx_record["path_offset"])
            paths = propbin.path_records[path_offset : path_offset + path_count]
            delays_ns = paths["delay_ns"].astype(np.float64)
            interactions = paths["interaction_count"].astype(np.int64)
            finite = np.isfinite(delays_ns)
            if not bool(finite.any()):
                continue
            valid_links += 1

            direct_indices = np.flatnonzero(finite & (interactions == 0))
            if direct_indices.size == 0:
                continue
            los_links += 1

            first_delay_ns = float(np.min(delays_ns[finite]))
            direct_delay_ns = float(np.min(delays_ns[direct_indices]))
            difference_ns = first_delay_ns - direct_delay_ns
            max_absolute_difference_ns = max(
                max_absolute_difference_ns,
                abs(difference_ns),
            )
            if math.isclose(
                first_delay_ns,
                direct_delay_ns,
                rel_tol=0.0,
                abs_tol=tolerance_ns,
            ):
                equal_links += 1
            elif difference_ns < 0.0:
                first_before_direct += 1
            else:
                first_after_direct += 1

    return {
        "files": len(files),
        "valid_links": valid_links,
        "los_links": los_links,
        "equal_links": equal_links,
        "first_before_direct": first_before_direct,
        "first_after_direct": first_after_direct,
        "tolerance_ns": tolerance_ns,
        "max_absolute_difference_ns": max_absolute_difference_ns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the minimum valid path delay equals the zero-interaction "
            "direct-path delay in raw RayVerse/D2LoS LoS links."
        )
    )
    parser.add_argument("--d2los-root", type=Path, required=True)
    parser.add_argument("--max-maps", type=int)
    parser.add_argument("--max-sources-per-map", type=int)
    parser.add_argument("--tolerance-ns", type=float, default=1.0e-6)
    args = parser.parse_args()
    if args.tolerance_ns < 0.0:
        raise ValueError("--tolerance-ns must be non-negative")

    report = audit_los_delay_relation(
        args.d2los_root,
        max_maps=args.max_maps,
        max_sources_per_map=args.max_sources_per_map,
        tolerance_ns=args.tolerance_ns,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
