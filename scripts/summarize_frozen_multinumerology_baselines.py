from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    "MAE",
    "RMSE",
    "signed_mean",
    "pearson",
    "accuracy_at_50ns",
)
FINAL_OUTPUT_METHOD = "paired_gate_ab_hard"


def finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--metrics-name", default="final_test_metrics.csv")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()

    if len(args.seeds) < 2:
        raise ValueError("At least two seeds are required for sample SD.")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for seed in args.seeds:
        path = args.root / f"seed_{seed}" / args.metrics_name
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["pair"],
                    row["field"],
                    row["method"],
                    row["target_range"],
                )
                grouped[key].append(row)

    rows = []
    expected_n = len(args.seeds)
    for key, seed_rows in grouped.items():
        if len(seed_rows) != expected_n:
            raise ValueError(
                f"Expected {expected_n} rows for {key}, found {len(seed_rows)}."
            )
        counts = {int(row["count"]) for row in seed_rows}
        if len(counts) != 1:
            raise ValueError(f"Sample counts differ across seeds for {key}: {counts}")
        result: dict[str, str | int | float] = {
            "pair": key[0],
            "target": key[1],
            "method": key[2],
            "target_range": key[3],
            "seeds": ",".join(str(seed) for seed in args.seeds),
            "n": expected_n,
            "count": counts.pop(),
            "is_final_output": int(key[2] == FINAL_OUTPUT_METHOD),
        }
        for metric in METRICS:
            values = [
                value
                for row in seed_rows
                if (value := finite_float(row[metric])) is not None
            ]
            result[f"{metric}_mean"] = mean(values) if values else math.nan
            result[f"{metric}_sample_std"] = (
                stdev(values) if len(values) > 1 else math.nan
            )
        rows.append(result)

    method_order = {
        "single_nf64": 0,
        "single_nf96": 1,
        "single_nf128": 2,
        "single_nf192": 3,
        "single_nf256": 4,
        "fixed_max_period_branch": 5,
        "paired_fused_direct": 6,
        "paired_expert_soft": 7,
        FINAL_OUTPUT_METHOD: 8,
    }
    range_order = {
        "all": 0,
        "0_100": 1,
        "100_300": 2,
        "300_600": 3,
        "600_960": 4,
        "960_1920": 5,
    }
    rows.sort(
        key=lambda row: (
            str(row["pair"]),
            str(row["target"]),
            method_order.get(str(row["method"]), 99),
            range_order.get(str(row["target_range"]), 99),
        )
    )

    output_prefix = args.output_prefix or args.root / "frozen_baseline_3seed_summary"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    all_range_path = Path(f"{output_prefix}_all.csv")
    with all_range_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(row for row in rows if row["target_range"] == "all")
    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "final_output_method": FINAL_OUTPUT_METHOD,
                "standard_deviation": "sample (ddof=1)",
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for row in rows:
        if row["target_range"] != "all":
            continue
        print(
            f"{row['pair']}_{row['target']}_{row['method']}_MAE="
            f"{float(row['MAE_mean']):.4f} +/- "
            f"{float(row['MAE_sample_std']):.4f}"
        )
        print(
            f"{row['pair']}_{row['target']}_{row['method']}_accuracy_at_50ns="
            f"{float(row['accuracy_at_50ns_mean']):.4f} +/- "
            f"{float(row['accuracy_at_50ns_sample_std']):.4f}"
        )
    print(f"saved_frozen_baseline_summary_csv={csv_path}")
    print(f"saved_frozen_baseline_all_summary_csv={all_range_path}")
    print(f"saved_frozen_baseline_summary_json={json_path}")


if __name__ == "__main__":
    main()
