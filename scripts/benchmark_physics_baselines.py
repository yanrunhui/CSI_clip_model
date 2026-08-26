from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset, physics_raw_values  # noqa: E402
from scripts.inference_benchmark_common import (  # noqa: E402
    parameter_counts,
    resolve_file_sha256,
    sha256_file_map,
    summarize_cost,
    timed_call,
    write_benchmark_outputs,
    write_environment,
)
from scripts.train_physics_baselines import (  # noqa: E402
    CNNBaseline,
    CSIEncoderSingleTask,
    FlattenedCSIMLP,
    PDPFeatureMLP,
    TARGET_SPECS,
    TransformerNoBranches,
    make_collate,
    normalized_to_raw,
)


COMMON_TARGETS = (
    "first_path_delay",
    "first_path_angle",
    "first_path_power",
    "k_factor",
    "reflection_count",
)


def checkpoint_argument(checkpoint: dict[str, Any], name: str, default: Any) -> Any:
    return checkpoint.get("args", {}).get(name, default)


def build_model(checkpoint: dict[str, Any]) -> torch.nn.Module:
    model_name = str(checkpoint["model_name"])
    target_name = str(checkpoint["target_name"])
    spec = TARGET_SPECS[target_name]
    d_token = int(checkpoint["d_token"])
    max_tokens = int(checkpoint["max_tokens"])
    n_freq = int(checkpoint["n_freq"])
    hidden_dim = int(checkpoint_argument(checkpoint, "hidden_dim", 1024))
    head_hidden_dim = int(checkpoint_argument(checkpoint, "head_hidden_dim", 256))
    dropout = float(checkpoint_argument(checkpoint, "dropout", 0.1))
    if model_name == "flattened_mlp":
        return FlattenedCSIMLP(
            input_dim=max_tokens * d_token * n_freq,
            output_dim=spec.output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if model_name == "cnn_baseline":
        return CNNBaseline(d_token, spec.output_dim, hidden_dim, dropout)
    if model_name == "pdp_ifft_mlp":
        return PDPFeatureMLP(
            output_dim=spec.output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pdp_bins=int(checkpoint_argument(checkpoint, "pdp_bins", 64)),
        )
    if model_name == "transformer_no_branches":
        return TransformerNoBranches(
            d_token=d_token,
            output_dim=spec.output_dim,
            d_model=int(checkpoint_argument(checkpoint, "encoder_d_model", 384)),
            n_heads=int(checkpoint_argument(checkpoint, "transformer_heads", 6)),
            n_layers=int(checkpoint_argument(checkpoint, "transformer_layers", 4)),
            d_ff=int(checkpoint_argument(checkpoint, "transformer_ff", 1024)),
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
    if model_name == "csi_encoder_single_task":
        return CSIEncoderSingleTask(
            d_token=d_token,
            output_dim=spec.output_dim,
            token_norm_mode=str(checkpoint_argument(checkpoint, "token_norm_mode", "std")),
            d_model=int(checkpoint_argument(checkpoint, "encoder_d_model", 384)),
            d_clip=int(checkpoint_argument(checkpoint, "encoder_d_clip", 256)),
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model_name={model_name!r}")


def load_models(
    checkpoint_dir: Path,
    model_name: str,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    models = {}
    checkpoints = {}
    for target in COMMON_TARGETS:
        path = checkpoint_dir / f"{model_name}_{target}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing baseline checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != model_name or checkpoint.get("target_name") != target:
            raise ValueError(f"Checkpoint identity mismatch: {path}")
        model = build_model(checkpoint)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        models[target] = model
        checkpoints[target] = checkpoint
    return models, checkpoints


def move_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def scalar_prediction(
    prediction: torch.Tensor,
    target: str,
) -> float:
    spec = TARGET_SPECS[target]
    raw = normalized_to_raw(prediction, spec)[0]
    if spec.kind == "angle":
        vector = F.normalize(raw, dim=0, eps=1e-6)
        return math.degrees(math.atan2(float(vector[0]), float(vector[1])))
    return float(raw[0])


def circular_error_degrees(prediction: float, target: float) -> float:
    delta = math.radians(prediction - target)
    return abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-sha256")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--model-name",
        required=True,
        choices=(
            "flattened_mlp",
            "csi_encoder_single_task",
            "cnn_baseline",
            "pdp_ifft_mlp",
            "transformer_no_branches",
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--warmup-samples", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.limit <= 0 or args.warmup_samples < 0 or args.repeats <= 0:
        raise ValueError("limit/repeats must be positive and warmup-samples nonnegative.")
    if not torch.cuda.is_available():
        raise RuntimeError("The cost benchmark requires CUDA.")
    device = torch.device("cuda")
    dataset = PreprocessedCSIDataset.from_pt(args.data_path)
    samples = dataset.samples[: args.limit]
    if len(samples) != args.limit:
        raise ValueError(f"Requested {args.limit} samples, found {len(samples)}.")
    models, checkpoints = load_models(Path(args.checkpoint_dir), args.model_name, device)
    input_shapes = {
        (
            int(checkpoint["max_tokens"]),
            int(checkpoint["d_token"]),
            int(checkpoint["n_freq"]),
        )
        for checkpoint in checkpoints.values()
    }
    if len(input_shapes) != 1:
        raise ValueError(
            "The five task checkpoints do not share one input shape: "
            f"{sorted(input_shapes)}"
        )
    max_tokens, _, _ = next(iter(input_shapes))
    collate = make_collate(max_tokens, TARGET_SPECS["first_path_delay"])

    def predict(sample) -> dict[str, float]:
        batch = move_inputs(collate([sample]), device)
        return {
            target: scalar_prediction(models[target](batch), target)
            for target in COMMON_TARGETS
        }

    with torch.inference_mode():
        for sample in samples[: args.warmup_samples]:
            predict(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        latency_rows: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            for index, sample in enumerate(samples):
                result = timed_call(lambda sample=sample: predict(sample), device)
                latency_rows.append(
                    {
                        "model": args.model_name,
                        "seed": args.seed,
                        "repeat": repeat,
                        "sample_index": index,
                        "group_id": str(getattr(sample, "group_id", "")),
                        "config_key": str(getattr(sample, "config_key", "")),
                        "wall_ms": result.wall_ms,
                        "cuda_ms": result.cuda_ms,
                        "generated_tokens": 0,
                    }
                )
                if repeat == 0:
                    predictions.append(
                        {
                            "index": index,
                            "group_id": str(getattr(sample, "group_id", "")),
                            **result.value,
                        }
                    )

    total_parameters, _ = parameter_counts(*models.values())
    trainable_parameters = sum(
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    )
    peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    checkpoint_paths = [
        Path(args.checkpoint_dir) / f"{args.model_name}_{target}.pt"
        for target in COMMON_TARGETS
    ]
    summary = summarize_cost(
        model=args.model_name,
        seed=args.seed,
        rows=latency_rows,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        device=device,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        metadata={
            "task_count": len(COMMON_TARGETS),
            "repeats": args.repeats,
            "inference_precision": "float32",
            "cpu_offload": False,
            "data_sha256": resolve_file_sha256(args.data_path, args.data_sha256),
            "checkpoint_sha256": json.dumps(
                sha256_file_map(checkpoint_paths), sort_keys=True
            ),
        },
    )
    config = vars(args) | {"targets": list(COMMON_TARGETS)}
    output = Path(args.output_dir)
    write_benchmark_outputs(output, rows=latency_rows, summary=summary, config=config)
    write_environment(output, device)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    target_values: dict[str, list[float]] = {target: [] for target in COMMON_TARGETS}
    for sample in samples:
        raw = physics_raw_values(sample)
        for target in COMMON_TARGETS:
            spec = TARGET_SPECS[target]
            indices = list(spec.indices)
            if spec.kind == "angle":
                target_values[target].append(
                    math.degrees(math.atan2(float(raw[indices[0]]), float(raw[indices[1]])))
                )
            else:
                target_values[target].append(float(raw[indices[0]]))
    metric_rows = []
    for target in COMMON_TARGETS:
        errors = []
        for index, prediction in enumerate(predictions):
            predicted = float(prediction[target])
            expected = target_values[target][index]
            error = (
                circular_error_degrees(predicted, expected)
                if target == "first_path_angle"
                else abs(predicted - expected)
            )
            if math.isfinite(error):
                errors.append(error)
        metric_rows.append(
            {
                "target": target,
                "metric": "MAE",
                "value": sum(errors) / len(errors) if errors else math.nan,
                "count": len(errors),
            }
        )
    with (output / "physics_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("target", "metric", "value", "count"))
        writer.writeheader()
        writer.writerows(metric_rows)
    print("benchmark_cost_summary=" + json.dumps(summary, sort_keys=True))
    print(f"saved_physics_metrics={output / 'physics_metrics.csv'}")


if __name__ == "__main__":
    main()
