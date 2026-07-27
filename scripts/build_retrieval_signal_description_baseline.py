from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import PreprocessedCSIDataset, physics_raw_values  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    _render_signal_description,
    _signal_description_record,
)


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def los_delay_ns(sample) -> float:
    value = finite_float(getattr(sample, "los_delay_s", math.nan))
    return float(value * 1e9) if value is not None else math.nan


def los_angle_sincos(sample) -> torch.Tensor | None:
    value = finite_float(getattr(sample, "los_aoa_az_deg", math.nan))
    if value is None:
        return None
    rad = math.radians(value)
    return torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)


def sample_record(sample, *, signal_description_correction: str = "none") -> dict[str, float | str]:
    return _signal_description_record(
        sample.semantic_key,
        physics_raw_values(sample),
        los_delay_ns=los_delay_ns(sample),
        los_angle_sincos=los_angle_sincos(sample),
        reflection_count=float(getattr(sample, "reflection_count", math.nan)),
        reflection_path_count=float(getattr(sample, "reflection_path_count", math.nan)),
        signal_description_correction=signal_description_correction,
    )


def token_stats(sample) -> torch.Tensor:
    tokens = sample.tokens[: sample.n_tokens].float()
    if tokens.numel() == 0:
        return torch.zeros(4, dtype=torch.float32)
    flat = tokens.flatten()
    return torch.tensor(
        [
            float(flat.mean()),
            float(flat.std(unbiased=False)),
            float(flat.abs().mean()),
            float(torch.sqrt(flat.square().mean().clamp(min=1e-12))),
        ],
        dtype=torch.float32,
    )


def pdp_stats_feature(sample) -> torch.Tensor:
    profile = getattr(sample, "delay_power_profile", None)
    if not isinstance(profile, torch.Tensor):
        profile = torch.zeros(64, dtype=torch.float32)
    profile = profile.flatten().float()
    if profile.numel() != 64:
        profile = torch.nn.functional.adaptive_avg_pool1d(
            profile.view(1, 1, -1),
            64,
        ).flatten()
    return torch.cat([profile, token_stats(sample)], dim=0)


def build_features(samples, feature: str) -> torch.Tensor:
    if feature != "pdp_stats":
        raise ValueError(f"Unsupported feature={feature!r}.")
    return torch.stack([pdp_stats_feature(sample) for sample in samples], dim=0)


def standardize(train_features: torch.Tensor, test_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    return (train_features - mean) / std, (test_features - mean) / std


def nearest_neighbor_indices(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_features = train_features.float()
    test_features = test_features.float()
    train_norm = train_features.square().sum(dim=1).unsqueeze(0)
    nearest_indices = []
    nearest_distances = []
    for start in range(0, test_features.shape[0], chunk_size):
        chunk = test_features[start : start + chunk_size]
        distances = (
            chunk.square().sum(dim=1, keepdim=True)
            + train_norm
            - 2.0 * chunk @ train_features.T
        ).clamp(min=0.0)
        values, indices = distances.min(dim=1)
        nearest_indices.append(indices.cpu())
        nearest_distances.append(values.sqrt().cpu())
    return torch.cat(nearest_indices), torch.cat(nearest_distances)


def load_samples(path: str, limit: int | None):
    samples = PreprocessedCSIDataset.from_pt(path).samples
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No samples loaded from {path}.")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CSI-to-text nearest-neighbor retrieval baseline payload. "
            "The predicted description for each test CSI is copied from the closest "
            "training CSI under a CSI/PDP feature distance."
        )
    )
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature", choices=("pdp_stats",), default="pdp_stats")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument(
        "--signal-description-correction",
        choices=("none", "bounds", "relational"),
        default="relational",
    )
    args = parser.parse_args()

    train_samples = load_samples(args.train_data, args.limit_train)
    test_samples = load_samples(args.test_data, args.limit_test)
    train_features = build_features(train_samples, args.feature)
    test_features = build_features(test_samples, args.feature)
    train_features, test_features = standardize(train_features, test_features)
    nearest_indices, nearest_distances = nearest_neighbor_indices(
        train_features,
        test_features,
        chunk_size=args.chunk_size,
    )

    predicted_records = []
    target_records = []
    predicted_texts = []
    target_texts = []
    comparisons = []
    for idx, nearest_idx in enumerate(nearest_indices.tolist()):
        predicted_sample = train_samples[int(nearest_idx)]
        target_sample = test_samples[idx]
        predicted_record = sample_record(
            predicted_sample,
            signal_description_correction=args.signal_description_correction,
        )
        target_record = sample_record(target_sample)
        predicted_text = _render_signal_description(predicted_record)
        target_text = _render_signal_description(target_record)
        predicted_records.append(predicted_record)
        target_records.append(target_record)
        predicted_texts.append(predicted_text)
        target_texts.append(target_text)
        comparisons.append(
            {
                "index": idx,
                "nearest_train_index": int(nearest_idx),
                "nearest_distance": float(nearest_distances[idx]),
                "group_id": getattr(target_sample, "group_id", ""),
                "config_key": getattr(target_sample, "config_key", ""),
                "predicted_signal_description": predicted_text,
                "target_signal_description": target_text,
                "predicted_record": predicted_record,
                "target_record": target_record,
            }
        )

    payload = {
        "predicted_signal_descriptions": predicted_texts,
        "target_signal_descriptions": target_texts,
        "predicted_signal_records": predicted_records,
        "target_signal_records": target_records,
        "comparisons": comparisons,
        "nearest_train_indices": nearest_indices,
        "nearest_distances": nearest_distances,
        "metadata": {
            "baseline": "csi_to_text_nearest_neighbor",
            "feature": args.feature,
            "train_data": args.train_data,
            "test_data": args.test_data,
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "signal_description_correction": args.signal_description_correction,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"saved_retrieval_signal_description_payload={output_path}")
    print("retrieval_signal_description_metadata=" + json.dumps(payload["metadata"], sort_keys=True))
    for idx in range(min(3, len(predicted_texts))):
        print(f"retrieval_signal_description_example_{idx + 1}_pred_text={predicted_texts[idx]}")
        print(f"retrieval_signal_description_example_{idx + 1}_true_text={target_texts[idx]}")


if __name__ == "__main__":
    main()
