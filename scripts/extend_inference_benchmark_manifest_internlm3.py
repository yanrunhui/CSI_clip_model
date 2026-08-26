from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def require(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a completed InternLM3 result to an existing benchmark manifest."
    )
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_manifest)
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    entries = list(payload.get("models", []))
    entries = [
        entry
        for entry in entries
        if str(entry.get("model")) != "InternLM3-8B-Instruct"
    ]
    current = Path(args.benchmark_root) / "internlm3_8b" / "seed_0"
    require(current / "BENCHMARK_COMPLETE")
    metrics_path = current / "text_metrics" / "signal_description_text_metrics.csv"
    provenance_path = current / "quality_provenance.json"
    quality_samples = int(payload.get("quality_sample_count", -1))
    if quality_samples <= 0:
        raise ValueError("Base manifest is missing quality_sample_count.")
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else {}
    )
    metrics_sha256 = sha256_file(metrics_path)
    if provenance and provenance.get("quality_metrics_sha256") != metrics_sha256:
        raise ValueError(f"InternLM3 quality metric SHA256 mismatch: {metrics_path}")
    entries.append(
        {
            "model": "InternLM3-8B-Instruct",
            "seed": 0,
            "cost_summary": require(current / "cost_summary.json"),
            "metrics": require(metrics_path),
            "metric_format": "text",
            "panel_b": True,
            "quality_sample_count": quality_samples,
            "quality_reused": bool(provenance.get("quality_reused", False)),
            "quality_metrics_source": str(
                provenance.get("quality_metrics_source", metrics_path)
            ),
            "quality_metrics_sha256": metrics_sha256,
            "quality_test_data_sha256": str(
                provenance.get("test_data_sha256", "")
            ),
        }
    )
    payload["models"] = entries
    payload["base_manifest"] = str(base_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved_extended_benchmark_manifest={output}")


if __name__ == "__main__":
    main()
