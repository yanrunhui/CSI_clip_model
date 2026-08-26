from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    PHYSICS_TARGET_NAMES,
    PHYSICS_TARGET_OFFSETS,
    PHYSICS_TARGET_SCALES,
    PreprocessedCSIDataset,
    PreprocessedSample,
    collate_fn,
)
from data.noise import (  # noqa: E402
    add_awgn_to_preprocessed_sample,
    sample_noise_seed,
)
from scripts.benchmark_full_model import build_model  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    _apply_signal_description_correction,
    _infer_first_path_power_gate_mode,
    _infer_use_power_branch,
    _physics_raw_predictions,
    _render_signal_description,
    _signal_description_record,
    build_tokenizer,
    move_batch,
)
from scripts.evaluate_signal_descriptions import (  # noqa: E402
    DEFAULT_TOLERANCES,
    evaluate_payload,
    numeric_error,
)
from scripts.pretrain import deserialize_prototype_keys  # noqa: E402
from scripts.qwen_csi_text_common import target_response  # noqa: E402
from scripts.summarize_noise_robustness_by_model_seed import (  # noqa: E402
    aggregate_by_model_seed,
)


PRIMARY_METRICS = (
    "first_delay_mae_ns",
    "first_angle_mae_deg",
    "first_power_mae_db",
    "k_factor_mae_db",
    "reflection_mae",
    "numeric_accuracy",
    "los_delay_mae_ns",
    "nlos_delay_mae_ns",
    "los_angle_mae_deg",
    "nlos_angle_mae_deg",
)

ERROR_METRICS = tuple(metric for metric in PRIMARY_METRICS if metric != "numeric_accuracy")


@dataclass(frozen=True)
class CheckpointSpec:
    model_seed: str
    path: Path


class NoiseConditionDataset(Dataset):
    """Apply deterministic per-sample AWGN lazily, without duplicating the split."""

    def __init__(
        self,
        samples: list[PreprocessedSample],
        *,
        snr_db: float | None,
        noise_seed: int | None,
        patch_1d: int,
        patch_2d: tuple[int, int],
    ) -> None:
        if (snr_db is None) != (noise_seed is None):
            raise ValueError("snr_db and noise_seed must either both be set or both be None")
        self.samples = samples
        self.snr_db = snr_db
        self.noise_seed = noise_seed
        self.patch_1d = patch_1d
        self.patch_2d = patch_2d
        self.actual_snr_db: dict[int, float] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreprocessedSample:
        sample = self.samples[index]
        if self.snr_db is None:
            return sample
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_noise_seed(int(self.noise_seed), index))
        noisy_sample, actual_snr_db = add_awgn_to_preprocessed_sample(
            sample,
            self.snr_db,
            generator=generator,
            patch_1d=self.patch_1d,
            patch_2d=self.patch_2d,
        )
        self.actual_snr_db[index] = actual_snr_db
        return noisy_sample


def parse_checkpoint_spec(value: str) -> CheckpointSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--checkpoint must use MODEL_SEED=/path/to/checkpoint.pt"
        )
    model_seed, raw_path = value.split("=", 1)
    model_seed = model_seed.strip()
    raw_path = raw_path.strip()
    if not model_seed or not raw_path:
        raise argparse.ArgumentTypeError(
            "--checkpoint must use non-empty MODEL_SEED and path values"
        )
    return CheckpointSpec(model_seed=model_seed, path=Path(raw_path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else math.nan


def finite_std(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.stdev(finite) if len(finite) > 1 else 0.0 if finite else math.nan


def physics_index(name: str) -> int:
    return PHYSICS_TARGET_NAMES.index(name)


def scalar_mae(
    predicted_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    field: str,
    *,
    los_status: str | None = None,
) -> tuple[float, int]:
    errors: list[float] = []
    for predicted, target in zip(predicted_records, target_records):
        if los_status is not None and str(target.get("los_status")) != los_status:
            continue
        try:
            predicted_value = float(predicted[field])
            target_value = float(target[field])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(predicted_value) and math.isfinite(target_value):
            errors.append(numeric_error(field, predicted_value, target_value))
    return finite_mean(errors), len(errors)


def numeric_slot_accuracy(rows: list[dict[str, Any]]) -> float:
    for row in rows:
        if (
            row.get("metric") == "numerical_slot_accuracy"
            and row.get("field") == "all_numeric_slots"
        ):
            return float(row["value"])
    raise RuntimeError("evaluate_payload did not return numerical_slot_accuracy")


def summarize_prediction_metrics(
    predicted_records: list[dict[str, Any]],
    physical_prediction_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int]]:
    payload = {
        "predicted_signal_records": predicted_records,
        "target_signal_records": target_records,
        "predicted_signal_descriptions": [row["description"] for row in predicted_records],
        "target_signal_descriptions": [row["description"] for row in target_records],
    }
    description_rows, _, _ = evaluate_payload(
        payload,
        tolerances=dict(DEFAULT_TOLERANCES),
        los_delay_tolerance_ns=50.0,
        los_angle_tolerance_deg=30.0,
    )

    metric_fields = {
        "first_delay_mae_ns": ("first_path_delay_ns", None),
        "first_angle_mae_deg": ("first_path_angle_deg", None),
        "first_power_mae_db": ("first_path_power_dbw", None),
        "k_factor_mae_db": ("k_factor_db", None),
        "reflection_mae": ("reflection_count", None),
        "los_delay_mae_ns": ("first_path_delay_ns", "los"),
        "nlos_delay_mae_ns": ("first_path_delay_ns", "nlos"),
        "los_angle_mae_deg": ("first_path_angle_deg", "los"),
        "nlos_angle_mae_deg": ("first_path_angle_deg", "nlos"),
    }
    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metric, (field, los_status) in metric_fields.items():
        metrics[metric], counts[metric] = scalar_mae(
            physical_prediction_records,
            target_records,
            field,
            los_status=los_status,
        )
    metrics["numeric_accuracy"] = numeric_slot_accuracy(description_rows)
    counts["numeric_accuracy"] = sum(
        int(row["count"])
        for row in description_rows
        if row.get("metric") == "numerical_slot_accuracy"
        and row.get("field") == "all_numeric_slots"
    )
    return metrics, counts


@torch.inference_mode()
def predict_condition(
    *,
    samples: list[PreprocessedSample],
    conditioned_dataset: NoiseConditionDataset,
    checkpoint: dict[str, Any],
    batch_size: int,
    device: torch.device,
    signal_description_correction: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = build_tokenizer(samples, checkpoint)
    model = build_model(checkpoint, device)
    prototype_keys = deserialize_prototype_keys(checkpoint.get("prototype_keys"))
    if not prototype_keys:
        raise ValueError("Full-model checkpoint is missing prototype_keys")
    prototype_features = model.encode_prototypes(normalize=True)
    semantic_classifier_enabled = (
        float(checkpoint.get("args", {}).get("semantic_classifier_weight", 0.0)) > 0.0
    )
    use_power_branch = _infer_use_power_branch(checkpoint, None)
    power_gate_mode = _infer_first_path_power_gate_mode(checkpoint, None)
    if power_gate_mode == "predicted_los":
        power_gate_mode = "base"

    loader = DataLoader(
        conditioned_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
    predicted_records: list[dict[str, Any]] = []
    physical_prediction_records: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        features = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
            config_features=batch.get("config_features"),
            antenna_coordinates=batch.get("antenna_coordinates"),
            antenna_mask=batch.get("antenna_mask"),
        )
        delay_context = model.encode_csi_delay_context(
            batch["tokens"],
            batch["token_mask"],
            subcarrier_spacing=batch.get("subcarrier_spacing"),
            config_features=batch.get("config_features"),
        )
        first_delay_context = model.encode_first_path_delay_context(
            batch["tokens"],
            batch["token_mask"],
            beam_positions=batch.get("beam_positions"),
            freq_bin=batch.get("freq_bin"),
            bw_bin=batch.get("bw_bin"),
            subcarrier_spacing=batch.get("subcarrier_spacing"),
            config_features=batch.get("config_features"),
        )
        los_angle_context = model.encode_los_angle_context(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
        )
        first_angle_context = model.encode_first_path_angle_context(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            subcarrier_spacing=batch.get("subcarrier_spacing"),
        )
        power_context = None
        if use_power_branch:
            power_context = model.encode_power_context(
                batch["tokens"],
                batch["token_mask"],
                delay_power_map=batch.get("delay_power_map"),
                delay_power_profile=batch.get("delay_power_profile"),
            )
        outputs = model.predict_physics_components(
            features,
            power_context=power_context,
            delay_context=delay_context,
            first_path_delay_context=first_delay_context,
            los_angle_context=los_angle_context,
            first_path_angle_context=first_angle_context,
        )

        normalized = outputs["final"].clone()
        if power_gate_mode == "base":
            power_idx = physics_index("first_path_power_dbw")
            normalized[:, power_idx] = outputs["base"][:, power_idx]
        raw_predictions = _physics_raw_predictions(normalized)
        # The published first-delay metric uses the dedicated delay context head.
        delay_idx = physics_index("first_path_delay_ns")
        raw_predictions[:, delay_idx] = (
            outputs["first_path_delay_context"]
            * float(PHYSICS_TARGET_SCALES[delay_idx])
            + float(PHYSICS_TARGET_OFFSETS[delay_idx])
        )

        if semantic_classifier_enabled:
            semantic_labels = model.predict_semantic(features).argmax(dim=1)
        else:
            semantic_labels = (
                F.normalize(features, dim=-1) @ prototype_features.T
            ).argmax(dim=1)
        reflection_idx = physics_index("reflection_count")
        reflection_predictions = (
            outputs["reflection_count_prediction"]
            * float(PHYSICS_TARGET_SCALES[reflection_idx])
            + float(PHYSICS_TARGET_OFFSETS[reflection_idx])
        )
        reflection_path_predictions = outputs["reflection_path_count_prediction"] * 10.0

        for row_idx in range(raw_predictions.shape[0]):
            physical_record = _signal_description_record(
                prototype_keys[int(semantic_labels[row_idx])],
                raw_predictions[row_idx],
                los_delay_ns=float(outputs["los_delay_context"][row_idx]) * 3000.0,
                los_angle_sincos=outputs["los_angle_sincos"][row_idx],
                reflection_count=float(reflection_predictions[row_idx]),
                reflection_path_count=float(reflection_path_predictions[row_idx]),
                signal_description_correction="none",
            )
            physical_prediction_records.append(dict(physical_record))
            record = _apply_signal_description_correction(
                dict(physical_record),
                signal_description_correction,
            )
            predicted_records.append(
                {**record, "description": _render_signal_description(record)}
            )

    if len(predicted_records) != len(samples):
        raise RuntimeError(
            f"Prediction count {len(predicted_records)} != sample count {len(samples)}"
        )
    return predicted_records, physical_prediction_records


def actual_snr_summary(
    conditioned_dataset: NoiseConditionDataset,
    *,
    target_snr_db: float | None,
    tolerance_db: float,
) -> dict[str, float | int | None]:
    if target_snr_db is None:
        return {
            "actual_snr_mean_db": None,
            "actual_snr_std_db": None,
            "actual_snr_min_db": None,
            "actual_snr_max_db": None,
            "actual_snr_max_abs_error_db": None,
        }
    if len(conditioned_dataset.actual_snr_db) != len(conditioned_dataset):
        raise RuntimeError("Actual SNR audit is incomplete; not every sample was evaluated")
    values = [conditioned_dataset.actual_snr_db[index] for index in range(len(conditioned_dataset))]
    max_abs_error = max(abs(value - target_snr_db) for value in values)
    if max_abs_error > tolerance_db:
        raise RuntimeError(
            f"Actual SNR differs from target by as much as {max_abs_error:.4f} dB; "
            f"allowed tolerance is {tolerance_db:.4f} dB"
        )
    return {
        "actual_snr_mean_db": finite_mean(values),
        "actual_snr_std_db": finite_std(values),
        "actual_snr_min_db": min(values),
        "actual_snr_max_db": max(values),
        "actual_snr_max_abs_error_db": max_abs_error,
    }


def condition_name(snr_db: float | None) -> str:
    if snr_db is None:
        return "clean"
    value = f"{snr_db:g}".replace("-", "minus_").replace(".", "p")
    return f"snr_{value}_db"


def write_prediction_audit(
    path: Path,
    *,
    samples: list[PreprocessedSample],
    predicted_records: list[dict[str, Any]],
    physical_prediction_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    conditioned_dataset: NoiseConditionDataset,
    model_seed: str,
    noise_seed: int | None,
    target_snr_db: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, (sample, predicted, physical_prediction, target) in enumerate(
            zip(
                samples,
                predicted_records,
                physical_prediction_records,
                target_records,
            )
        ):
            row = {
                "sample_index": index,
                "group_id": str(sample.group_id),
                "config_key": str(sample.config_key),
                "model_seed": model_seed,
                "noise_seed": noise_seed,
                "target_snr_db": target_snr_db,
                "actual_snr_db": conditioned_dataset.actual_snr_db.get(index),
                "predicted_record": predicted,
                "physical_prediction_record": physical_prediction,
                "target_record": target,
            }
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def add_degradation_fields(
    run: dict[str, Any],
    clean_run: dict[str, Any],
) -> None:
    for metric in ERROR_METRICS:
        clean_value = float(clean_run[metric])
        noisy_value = float(run[metric])
        key = f"{metric}_relative_degradation_pct"
        run[key] = (
            (noisy_value - clean_value) / clean_value * 100.0
            if math.isfinite(clean_value)
            and math.isfinite(noisy_value)
            and clean_value != 0.0
            else math.nan
        )
    run["numeric_accuracy_drop_percentage_points"] = (
        float(clean_run["numeric_accuracy"]) - float(run["numeric_accuracy"])
    ) * 100.0


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["condition"]), []).append(run)
    aggregates: list[dict[str, Any]] = []
    degradation_fields = [
        f"{metric}_relative_degradation_pct" for metric in ERROR_METRICS
    ] + ["numeric_accuracy_drop_percentage_points"]
    for condition, rows in grouped.items():
        result: dict[str, Any] = {
            "condition": condition,
            "target_snr_db": rows[0]["target_snr_db"],
            "run_count": len(rows),
        }
        for field in (*PRIMARY_METRICS, *degradation_fields):
            values = [float(row[field]) for row in rows]
            result[f"{field}_mean"] = finite_mean(values)
            result[f"{field}_std"] = finite_std(values)
        actual_values = [
            float(row["actual_snr_mean_db"])
            for row in rows
            if row["actual_snr_mean_db"] is not None
        ]
        result["actual_snr_mean_db"] = finite_mean(actual_values) if actual_values else None
        result["actual_snr_std_across_runs_db"] = (
            finite_std(actual_values) if actual_values else None
        )
        aggregates.append(result)
    return sorted(
        aggregates,
        key=lambda row: (
            row["target_snr_db"] is not None,
            -(float(row["target_snr_db"]) if row["target_snr_db"] is not None else 0.0),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate zero-shot Full-model robustness to per-sample complex AWGN. "
            "Noise is added to unnormalized complex beamspace CSI before model "
            "normalization and all learned CSI feature extraction."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_spec,
        required=True,
        help="Repeat as MODEL_SEED=/path/to/checkpoint.pt (for example 0=seed0.pt).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[30.0, 20.0, 10.0, 0.0])
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--patch-1d", type=int, default=4)
    parser.add_argument("--patch-2d", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--actual-snr-tolerance-db", type=float, default=1.0)
    parser.add_argument(
        "--signal-description-correction",
        choices=("none", "bounds", "relational"),
        default="relational",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exclude-clean", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.actual_snr_tolerance_db <= 0.0:
        raise ValueError("--actual-snr-tolerance-db must be positive")
    if not args.noise_seeds:
        raise ValueError("At least one --noise-seeds value is required")
    if not all(math.isfinite(value) for value in args.snr_db):
        raise ValueError("All --snr-db values must be finite")

    checkpoint_specs: list[CheckpointSpec] = args.checkpoint
    model_seeds = [spec.model_seed for spec in checkpoint_specs]
    if len(model_seeds) != len(set(model_seeds)):
        raise ValueError("Each checkpoint MODEL_SEED must be unique")
    missing_paths = [str(spec.path) for spec in checkpoint_specs if not spec.path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing_paths))
    if not args.data_path.is_file():
        raise FileNotFoundError(args.data_path)

    dataset = PreprocessedCSIDataset.from_pt(str(args.data_path))
    samples = dataset.samples[: args.limit]
    if not samples:
        raise ValueError("The evaluation split contains no samples")
    target_records = [target_response(sample) for sample in samples]
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    conditions: list[tuple[float | None, int | None]] = []
    if not args.exclude_clean:
        conditions.append((None, None))
    conditions.extend(
        (float(snr_db), int(noise_seed))
        for snr_db in args.snr_db
        for noise_seed in args.noise_seeds
    )

    runs: list[dict[str, Any]] = []
    clean_by_model_seed: dict[str, dict[str, Any]] = {}
    for spec in checkpoint_specs:
        checkpoint = torch.load(spec.path, map_location="cpu", weights_only=False)
        for target_snr_db, noise_seed in conditions:
            conditioned_dataset = NoiseConditionDataset(
                samples,
                snr_db=target_snr_db,
                noise_seed=noise_seed,
                patch_1d=args.patch_1d,
                patch_2d=tuple(args.patch_2d),
            )
            predicted_records, physical_prediction_records = predict_condition(
                samples=samples,
                conditioned_dataset=conditioned_dataset,
                checkpoint=checkpoint,
                batch_size=args.batch_size,
                device=device,
                signal_description_correction=args.signal_description_correction,
            )
            metrics, counts = summarize_prediction_metrics(
                predicted_records,
                physical_prediction_records,
                target_records,
            )
            run: dict[str, Any] = {
                "model_seed": spec.model_seed,
                "checkpoint": str(spec.path.resolve()),
                "condition": condition_name(target_snr_db),
                "target_snr_db": target_snr_db,
                "noise_seed": noise_seed,
                "sample_count": len(samples),
                **actual_snr_summary(
                    conditioned_dataset,
                    target_snr_db=target_snr_db,
                    tolerance_db=args.actual_snr_tolerance_db,
                ),
                **metrics,
            }
            for metric, count in counts.items():
                run[f"{metric}_count"] = count

            if target_snr_db is None:
                clean_by_model_seed[spec.model_seed] = run
            elif spec.model_seed in clean_by_model_seed:
                add_degradation_fields(run, clean_by_model_seed[spec.model_seed])
            else:
                for metric in ERROR_METRICS:
                    run[f"{metric}_relative_degradation_pct"] = math.nan
                run["numeric_accuracy_drop_percentage_points"] = math.nan
            if target_snr_db is None:
                for metric in ERROR_METRICS:
                    run[f"{metric}_relative_degradation_pct"] = 0.0
                run["numeric_accuracy_drop_percentage_points"] = 0.0
            runs.append(run)

            if not args.no_save_predictions:
                suffix = condition_name(target_snr_db)
                if noise_seed is not None:
                    suffix += f"_noise_seed_{noise_seed}"
                audit_path = (
                    args.output_dir
                    / "predictions"
                    / f"model_seed_{spec.model_seed}"
                    / f"{suffix}.jsonl"
                )
                write_prediction_audit(
                    audit_path,
                    samples=samples,
                    predicted_records=predicted_records,
                    physical_prediction_records=physical_prediction_records,
                    target_records=target_records,
                    conditioned_dataset=conditioned_dataset,
                    model_seed=spec.model_seed,
                    noise_seed=noise_seed,
                    target_snr_db=target_snr_db,
                )
            print(
                json.dumps(
                    json_safe({
                        "model_seed": spec.model_seed,
                        "condition": run["condition"],
                        "noise_seed": noise_seed,
                        **{metric: run[metric] for metric in PRIMARY_METRICS},
                    }),
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
        del checkpoint

    aggregate = aggregate_runs(runs)
    model_seed_means, model_seed_summary = aggregate_by_model_seed(runs)
    write_csv(args.output_dir / "noise_robustness_runs.csv", runs)
    write_csv(args.output_dir / "noise_robustness_summary.csv", aggregate)
    write_csv(
        args.output_dir / "noise_robustness_model_seed_means.csv",
        model_seed_means,
    )
    write_csv(
        args.output_dir / "noise_robustness_model_seed_summary.csv",
        model_seed_summary,
    )
    manifest = {
        "method": (
            "Per-sample circular complex AWGN on unnormalized beamspace CSI; "
            "padding excluded; noise precedes model normalization and learned encoders."
        ),
        "unchanged_inputs": (
            "Physical labels and the existing simulator-derived delay-power map/profile "
            "are unchanged across SNR conditions."
        ),
        "data_path": str(args.data_path.resolve()),
        "data_sha256": file_sha256(args.data_path),
        "sample_count": len(samples),
        "sample_ids": [str(sample.group_id) for sample in samples],
        "target_snr_db": list(args.snr_db),
        "noise_seeds": list(args.noise_seeds),
        "checkpoint_sha256": {
            spec.model_seed: file_sha256(spec.path) for spec in checkpoint_specs
        },
        "patch_1d": args.patch_1d,
        "patch_2d": list(args.patch_2d),
        "actual_snr_tolerance_db": args.actual_snr_tolerance_db,
        "signal_description_correction": args.signal_description_correction,
        "runs": runs,
        "aggregate": aggregate,
        "model_seed_means": model_seed_means,
        "model_seed_summary": model_seed_summary,
    }
    (args.output_dir / "noise_robustness_results.json").write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"saved_results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
