from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--expected-repeats", type=int, required=True)
    parser.add_argument("--expected-quality-samples", type=int, required=True)
    parser.add_argument("--require-perfect-parse", action="store_true")
    args = parser.parse_args()
    root = Path(args.benchmark_root)
    manifest = json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for entry in manifest["models"]:
        model = str(entry["model"])
        cost_path = Path(entry["cost_summary"])
        cost = json.loads(cost_path.read_text(encoding="utf-8"))
        if int(cost.get("samples_per_repeat", -1)) != args.expected_samples:
            errors.append(
                f"{model}: samples_per_repeat={cost.get('samples_per_repeat')} "
                f"expected={args.expected_samples}"
            )
        if int(cost.get("repeat_count", -1)) != args.expected_repeats:
            errors.append(
                f"{model}: repeat_count={cost.get('repeat_count')} "
                f"expected={args.expected_repeats}"
            )
        if int(entry.get("quality_sample_count", -1)) != args.expected_quality_samples:
            errors.append(
                f"{model}: quality_sample_count={entry.get('quality_sample_count')} "
                f"expected={args.expected_quality_samples}"
            )
        quality_test_sha256 = str(entry.get("quality_test_data_sha256", ""))
        cost_test_sha256 = str(cost.get("data_sha256", ""))
        if quality_test_sha256 and quality_test_sha256 != cost_test_sha256:
            errors.append(
                f"{model}: cost/quality test SHA256 mismatch "
                f"cost={cost_test_sha256} quality={quality_test_sha256}"
            )
        expected_measurements = args.expected_samples * args.expected_repeats
        if int(cost.get("measured_samples", -1)) != expected_measurements:
            errors.append(
                f"{model}: measured_samples={cost.get('measured_samples')} "
                f"expected={expected_measurements}"
            )
        if bool(cost.get("cpu_offload", False)):
            errors.append(f"{model}: CPU offload enabled")
        if args.require_perfect_parse and "parse_success_rate" in cost:
            if float(cost["parse_success_rate"]) != 1.0:
                errors.append(
                    f"{model}: parse_success_rate={cost['parse_success_rate']} expected=1"
                )
    for filename, score_field in (
        ("panel_a_common_physical_scores.csv", "common_physical_score"),
        ("panel_b_complete_scores.csv", "complete_score"),
    ):
        for row in read_csv(root / filename):
            if row.get("row_type") != "seed":
                continue
            try:
                score = float(row[score_field])
            except (KeyError, TypeError, ValueError):
                score = math.nan
            if not math.isfinite(score):
                errors.append(
                    f"{row.get('model')}: non-finite {score_field} in {filename}"
                )
    if errors:
        raise SystemExit("benchmark_validation_failed:\n" + "\n".join(errors))
    print(
        "benchmark_validation=passed "
        f"cost_samples={args.expected_samples} "
        f"quality_samples={args.expected_quality_samples}"
    )


if __name__ == "__main__":
    main()
