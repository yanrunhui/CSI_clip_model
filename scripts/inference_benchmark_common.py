from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class TimedResult:
    value: Any
    wall_ms: float
    cuda_ms: float


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(
    function: Callable[[], Any],
    device: torch.device,
) -> TimedResult:
    synchronize(device)
    start_event = None
    end_event = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    wall_start = time.perf_counter()
    value = function()
    if end_event is not None:
        end_event.record()
    synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    cuda_ms = (
        float(start_event.elapsed_time(end_event))
        if start_event is not None and end_event is not None
        else math.nan
    )
    return TimedResult(value=value, wall_ms=wall_ms, cuda_ms=cuda_ms)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parameter_counts(*modules: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for module in modules for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_file_sha256(path: str | Path, supplied: str | None = None) -> str:
    if supplied:
        normalized = supplied.strip().lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError(f"Invalid SHA256 value for {path}: {supplied!r}")
        return normalized
    return sha256_file(path)


def sha256_file_map(paths: list[str | Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def summarize_cost(
    *,
    model: str,
    seed: int,
    rows: list[dict[str, Any]],
    total_parameters: int,
    trainable_parameters: int,
    device: torch.device,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"No timed samples were recorded for {model} seed={seed}.")
    wall = [float(row["wall_ms"]) for row in rows]
    cuda = [float(row["cuda_ms"]) for row in rows if math.isfinite(float(row["cuda_ms"]))]
    generated_tokens = [int(row.get("generated_tokens", 0)) for row in rows]
    by_repeat: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_repeat.setdefault(int(row["repeat"]), []).append(row)
    repeat_statistics = []
    for repeat, repeat_rows in sorted(by_repeat.items()):
        repeat_wall = [float(row["wall_ms"]) for row in repeat_rows]
        repeat_cuda = [
            float(row["cuda_ms"])
            for row in repeat_rows
            if math.isfinite(float(row["cuda_ms"]))
        ]
        repeat_seconds = sum(repeat_wall) / 1000.0
        repeat_statistics.append(
            {
                "repeat": repeat,
                "sample_count": len(repeat_rows),
                "median_ms": statistics.median(repeat_wall),
                "p95_ms": percentile(repeat_wall, 0.95),
                "median_cuda_ms": (
                    statistics.median(repeat_cuda) if repeat_cuda else math.nan
                ),
                "p95_cuda_ms": percentile(repeat_cuda, 0.95),
                "throughput_samples_s": (
                    len(repeat_rows) / repeat_seconds if repeat_seconds > 0 else math.nan
                ),
            }
        )
    repeat_ids = [int(row["repeat"]) for row in repeat_statistics]
    if repeat_ids != list(range(len(repeat_ids))):
        raise ValueError(
            f"Repeat ids must be contiguous from zero for {model}: {repeat_ids}"
        )
    repeat_sample_counts = [int(row["sample_count"]) for row in repeat_statistics]
    if len(set(repeat_sample_counts)) != 1:
        raise ValueError(
            f"Incomplete timing repeat for {model}: sample counts={repeat_sample_counts}"
        )
    repeat_medians = [row["median_ms"] for row in repeat_statistics]
    repeat_p95s = [row["p95_ms"] for row in repeat_statistics]
    repeat_cuda_medians = [
        row["median_cuda_ms"]
        for row in repeat_statistics
        if math.isfinite(row["median_cuda_ms"])
    ]
    repeat_cuda_p95s = [
        row["p95_cuda_ms"]
        for row in repeat_statistics
        if math.isfinite(row["p95_cuda_ms"])
    ]
    repeat_throughputs = [row["throughput_samples_s"] for row in repeat_statistics]
    total_wall_seconds = sum(wall) / 1000.0
    summary = {
        "model": model,
        "seed": seed,
        "measured_samples": len(rows),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "repeat_count": len(repeat_statistics),
        "samples_per_repeat": (
            repeat_statistics[0]["sample_count"] if repeat_statistics else 0
        ),
        "median_latency_ms": statistics.mean(repeat_medians) if repeat_medians else math.nan,
        "p95_latency_ms": statistics.mean(repeat_p95s) if repeat_p95s else math.nan,
        "median_cuda_ms": (
            statistics.mean(repeat_cuda_medians) if repeat_cuda_medians else math.nan
        ),
        "p95_cuda_ms": statistics.mean(repeat_cuda_p95s) if repeat_cuda_p95s else math.nan,
        "throughput_samples_s": (
            statistics.mean(repeat_throughputs) if repeat_throughputs else math.nan
        ),
        "pooled_median_latency_ms": statistics.median(wall) if wall else math.nan,
        "pooled_p95_latency_ms": percentile(wall, 0.95),
        "pooled_median_cuda_ms": statistics.median(cuda) if cuda else math.nan,
        "pooled_p95_cuda_ms": percentile(cuda, 0.95),
        "pooled_throughput_samples_s": (
            len(rows) / total_wall_seconds if total_wall_seconds > 0 else math.nan
        ),
        "repeat_median_mean_ms": statistics.mean(repeat_medians) if repeat_medians else math.nan,
        "repeat_median_std_ms": (
            statistics.stdev(repeat_medians) if len(repeat_medians) > 1 else 0.0
        ),
        "repeat_p95_mean_ms": statistics.mean(repeat_p95s) if repeat_p95s else math.nan,
        "repeat_p95_std_ms": (
            statistics.stdev(repeat_p95s) if len(repeat_p95s) > 1 else 0.0
        ),
        "peak_allocated_gb": peak_allocated_bytes / (1024**3),
        "peak_reserved_gb": peak_reserved_bytes / (1024**3),
        "mean_generated_tokens": (
            statistics.mean(generated_tokens) if generated_tokens else 0.0
        ),
        "tokens_per_second": (
            sum(generated_tokens) / total_wall_seconds
            if total_wall_seconds > 0 and sum(generated_tokens) > 0
            else 0.0
        ),
        "device": str(device),
    }
    for repeat_row in repeat_statistics:
        repeat = repeat_row["repeat"]
        summary[f"repeat_{repeat}_median_ms"] = repeat_row["median_ms"]
        summary[f"repeat_{repeat}_p95_ms"] = repeat_row["p95_ms"]
        summary[f"repeat_{repeat}_throughput_samples_s"] = repeat_row[
            "throughput_samples_s"
        ]
    if metadata:
        summary.update(metadata)
    return summary


def write_benchmark_outputs(
    output_dir: str | Path,
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config: dict[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    latency_path = output / "per_sample_latency.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with latency_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "cost_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "cost_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    (output / "benchmark_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def runtime_environment(device: torch.device) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
    }
    for distribution in ("transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            environment[f"{distribution}_version"] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError:
            environment[f"{distribution}_version"] = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        environment.update(
            {
                "gpu_name": torch.cuda.get_device_name(index),
                "gpu_total_memory_gb": torch.cuda.get_device_properties(index).total_memory
                / (1024**3),
                "gpu_compute_capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(index)
                ),
            }
        )
    return environment


def write_environment(output_dir: str | Path, device: torch.device) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment.json").write_text(
        json.dumps(runtime_environment(device), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
