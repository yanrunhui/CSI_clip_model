from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


COMMON_BASELINE_TARGETS = {
    "first_path_delay",
    "first_path_angle",
    "first_path_power",
    "k_factor",
    "reflection_count",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_text_metrics(rows: list[dict[str, str]], expected_samples: int) -> None:
    candidates = [
        row
        for row in rows
        if row.get("metric") == "description_factual_accuracy"
        and row.get("field") == "primary_factual_metrics"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected one primary description_factual_accuracy row, "
            f"found {len(candidates)}."
        )
    actual = int(float(candidates[0]["count"]))
    if actual != expected_samples:
        raise ValueError(
            f"Text quality sample count is {actual}, expected {expected_samples}."
        )


def validate_baseline_metrics(rows: list[dict[str, str]], expected_samples: int) -> None:
    counts = {
        row.get("target", ""): int(float(row.get("count", "0")))
        for row in rows
        if row.get("metric") == "MAE"
    }
    if set(counts) != COMMON_BASELINE_TARGETS:
        raise ValueError(
            "Baseline quality targets do not match the five common tasks: "
            f"{sorted(counts)}"
        )
    mismatched = {
        target: count for target, count in counts.items() if count != expected_samples
    }
    if mismatched:
        raise ValueError(
            "Baseline quality sample counts do not match the requested count: "
            f"{mismatched} expected={expected_samples}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and reuse quality metrics independently of cost timing."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--metric-format", choices=("text", "baseline"), required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--test-data-sha256", required=True)
    parser.add_argument("--provenance-output", required=True)
    args = parser.parse_args()
    if args.expected_samples <= 0:
        raise ValueError("--expected-samples must be positive.")
    if len(args.test_data_sha256) != 64:
        raise ValueError("--test-data-sha256 must contain 64 hexadecimal characters.")

    source = Path(args.source).resolve()
    destination = Path(args.destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = read_rows(source)
    if args.metric_format == "text":
        validate_text_metrics(rows, args.expected_samples)
    else:
        validate_baseline_metrics(rows, args.expected_samples)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != destination.resolve():
        shutil.copy2(source, destination)
    metric_sha256 = sha256_file(destination)
    provenance = {
        "quality_metrics_source": str(source),
        "quality_metrics_destination": str(destination),
        "quality_metrics_sha256": metric_sha256,
        "quality_metric_format": args.metric_format,
        "quality_sample_count": args.expected_samples,
        "test_data_sha256": args.test_data_sha256,
        "quality_reused": True,
    }
    provenance_path = Path(args.provenance_output)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("reused_quality_metrics=" + json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
