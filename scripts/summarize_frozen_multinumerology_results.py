from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


FINAL_OUTPUT_METHOD = "paired_gate_ab_hard"
TRAINING_NUMEROLOGIES = ("nf64", "nf96", "nf192", "nf256")
HELD_OUT_NUMEROLOGY = "nf128"
PAIRS = ("nf96+nf128", "nf128+nf192")
TARGETS = ("first_path_delay_ns", "los_delay_ns")


def parse_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    metrics = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, raw_value = line.rsplit("=", 1)
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            metrics[key] = value
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/dev_holdout_nf128_expert_gate"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--log-name", default="train_test_output.txt")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    if len(args.seeds) < 2:
        raise ValueError(
            "At least two seeds are required for sample standard deviation."
        )

    seed_metrics = {
        seed: parse_metrics(args.root / f"seed_{seed}" / args.log_name)
        for seed in args.seeds
    }
    rows = []
    for pair in PAIRS:
        for target in TARGETS:
            prefix = f"{pair}_{target}_{FINAL_OUTPUT_METHOD}"
            mae_values = [seed_metrics[seed][f"{prefix}_MAE"] for seed in args.seeds]
            accuracy_values = [
                seed_metrics[seed][f"{prefix}_accuracy_at_50ns"] for seed in args.seeds
            ]
            rows.append(
                {
                    "pair": pair,
                    "target": target,
                    "final_output_method": FINAL_OUTPUT_METHOD,
                    "seeds": ",".join(str(seed) for seed in args.seeds),
                    "n": len(args.seeds),
                    "MAE_mean_ns": mean(mae_values),
                    "MAE_sample_std_ns": stdev(mae_values),
                    "accuracy_at_50ns_mean": mean(accuracy_values),
                    "accuracy_at_50ns_sample_std": stdev(accuracy_values),
                }
            )

    output_prefix = args.output_prefix or args.root / "frozen_final_3seed_summary"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "final_output_method": FINAL_OUTPUT_METHOD,
        "training_numerologies": TRAINING_NUMEROLOGIES,
        "held_out_numerology": HELD_OUT_NUMEROLOGY,
        "standard_deviation": "sample (ddof=1)",
        "rows": rows,
    }
    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"final_output_method={FINAL_OUTPUT_METHOD}")
    print(f"training_numerologies={','.join(TRAINING_NUMEROLOGIES)}")
    print(f"held_out_numerology={HELD_OUT_NUMEROLOGY}")
    for row in rows:
        print(
            f"{row['pair']}_{row['target']}_MAE="
            f"{row['MAE_mean_ns']:.4f} +/- {row['MAE_sample_std_ns']:.4f} ns"
        )
        print(
            f"{row['pair']}_{row['target']}_accuracy_at_50ns="
            f"{row['accuracy_at_50ns_mean']:.4f} +/- "
            f"{row['accuracy_at_50ns_sample_std']:.4f}"
        )
    print(f"saved_frozen_summary_csv={csv_path}")
    print(f"saved_frozen_summary_json={json_path}")


if __name__ == "__main__":
    main()
