from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BASELINE_NAMES = {
    "csi_encoder_single_task": "CSI encoder + single-task",
    "pdp_ifft_mlp": "PDP/IFFT + MLP",
    "flattened_mlp": "Flattened CSI + MLP",
    "transformer_no_branches": "Transformer without branches",
    "cnn_baseline": "CNN baseline",
}


def require(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def require_complete(path: Path) -> None:
    marker = path / "BENCHMARK_COMPLETE"
    if not marker.is_file():
        raise FileNotFoundError(f"Incomplete benchmark directory: {path} (missing {marker.name})")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def quality_metadata(
    directory: Path,
    metrics: str,
    expected_samples: int,
) -> dict[str, object]:
    metrics_path = Path(metrics)
    provenance_path = directory / "quality_provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if int(provenance.get("quality_sample_count", -1)) != expected_samples:
            raise ValueError(
                f"Quality sample count mismatch in {provenance_path}: "
                f"{provenance.get('quality_sample_count')} != {expected_samples}"
            )
        actual_sha256 = sha256_file(metrics_path)
        if str(provenance.get("quality_metrics_sha256")) != actual_sha256:
            raise ValueError(f"Quality metric SHA256 mismatch: {metrics_path}")
        return {
            "quality_sample_count": expected_samples,
            "quality_reused": bool(provenance.get("quality_reused", False)),
            "quality_metrics_source": str(provenance.get("quality_metrics_source", "")),
            "quality_metrics_sha256": actual_sha256,
            "quality_test_data_sha256": str(provenance.get("test_data_sha256", "")),
        }
    return {
        "quality_sample_count": expected_samples,
        "quality_reused": False,
        "quality_metrics_source": str(metrics_path),
        "quality_metrics_sha256": sha256_file(metrics_path),
        "quality_test_data_sha256": "",
    }


def manifest_entry(
    *,
    model: str,
    seed: int,
    directory: Path,
    metrics_path: Path,
    metric_format: str,
    panel_b: bool,
    quality_samples: int,
) -> dict[str, object]:
    metrics = require(metrics_path)
    return {
        "model": model,
        "seed": seed,
        "cost_summary": require(directory / "cost_summary.json"),
        "metrics": metrics,
        "metric_format": metric_format,
        "panel_b": panel_b,
        **quality_metadata(directory, metrics, quality_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--quality-samples", type=int, required=True)
    args = parser.parse_args()
    if args.quality_samples <= 0:
        raise ValueError("--quality-samples must be positive.")
    root = Path(args.benchmark_root)
    entries = []
    for seed in args.seeds:
        full = root / "full_multitask" / f"seed_{seed}"
        require_complete(full)
        entries.append(
            manifest_entry(
                model="Full model",
                seed=seed,
                directory=full,
                metrics_path=full / "text_metrics" / "signal_description_text_metrics.csv",
                metric_format="text",
                panel_b=True,
                quality_samples=args.quality_samples,
            )
        )
        for directory, display in BASELINE_NAMES.items():
            current = root / directory / f"seed_{seed}"
            require_complete(current)
            entries.append(
                manifest_entry(
                    model=display,
                    seed=seed,
                    directory=current,
                    metrics_path=current / "physics_metrics.csv",
                    metric_format="baseline",
                    panel_b=False,
                    quality_samples=args.quality_samples,
                )
            )
    for directory, display in (
        ("qwen3_1p7b", "Qwen3-1.7B"),
        ("qwen3_5_2b", "Qwen3.5-2B updated direct decoder"),
        ("deepseek_8b", "DeepSeek-R1-Qwen3-8B"),
    ):
        current = root / directory / "seed_0"
        require_complete(current)
        entries.append(
            manifest_entry(
                model=display,
                seed=0,
                directory=current,
                metrics_path=current / "text_metrics" / "signal_description_text_metrics.csv",
                metric_format="text",
                panel_b=True,
                quality_samples=args.quality_samples,
            )
        )
    payload = {
        "benchmark_root": str(root),
        "quality_sample_count": args.quality_samples,
        "models": entries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved_benchmark_manifest={output}")


if __name__ == "__main__":
    main()
