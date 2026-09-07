from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from data.dataset import PHYSICS_TARGET_NAMES, PHYSICS_TARGET_OFFSETS, PHYSICS_TARGET_SCALES
from models.encoder import CSIEncoder
from models.model import CSIClip
from models.text_encoder import PhysicsTextEncoder
from scripts.evaluate import (
    _infer_first_path_power_mode,
    _infer_first_path_power_use_internal_gate,
    _infer_shared_physics_token_residual_scale,
    _infer_token_norm_mode,
    _infer_use_array_invariant_delay_encoder,
    _infer_use_continuous_config_encoding,
    _infer_use_delay_specific_encoder,
    _infer_use_delay_spread_head,
    _infer_use_first_path_angle_context_encoder,
    _infer_use_los_angle_context_encoder,
    _infer_use_power_branch,
    _infer_use_shared_physics_token,
    _load_model_state_compatible,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained CrossConfig-CSI checkpoint and run a forward-only smoke "
            "test on compact BUPT NPZ shards. Predictions are diagnostic because the "
            "BUPT input is represented as zero-padded UPA-1x1."
        )
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--freq-bin", type=int, default=0)
    parser.add_argument(
        "--bw-bin",
        type=int,
        default=1,
        help="Training preprocessing bin for 128 selected frequency points (default: 1).",
    )
    parser.add_argument(
        "--bandwidth-hz",
        type=float,
        default=100e6,
        help="Bandwidth used by the checkpoint (default: 100 MHz).",
    )
    parser.add_argument("--antenna-spacing", type=float, default=0.5)
    parser.add_argument("--max-npz", type=int)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no model_state dictionary")
    return state


def infer_text_vocab_size(state: dict[str, torch.Tensor]) -> int:
    for key in (
        "text.embedding.weight",
        "text.token_embedding.weight",
        "text.token_emb.weight",
    ):
        tensor = state.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.shape[0])
    return 300


def build_model(checkpoint: dict, device: torch.device) -> CSIClip:
    state = checkpoint_state(checkpoint)
    token_norm_mode = _infer_token_norm_mode(checkpoint, None)
    continuous = _infer_use_continuous_config_encoding(checkpoint)
    model = CSIClip(
        CSIEncoder(
            d_token=8,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
            use_continuous_config_encoding=continuous,
        ),
        PhysicsTextEncoder(vocab_size=max(infer_text_vocab_size(state), 300)),
        num_prototypes=None,
        semantic_num_classes=None,
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        use_power_branch=_infer_use_power_branch(checkpoint, None),
        first_path_power_mode=_infer_first_path_power_mode(checkpoint),
        first_path_power_use_internal_gate=_infer_first_path_power_use_internal_gate(checkpoint),
        use_delay_spread_head=_infer_use_delay_spread_head(checkpoint),
        use_delay_specific_encoder=_infer_use_delay_specific_encoder(checkpoint),
        use_array_invariant_delay_encoder=_infer_use_array_invariant_delay_encoder(checkpoint),
        use_los_angle_context_encoder=_infer_use_los_angle_context_encoder(checkpoint),
        use_first_path_angle_context_encoder=_infer_use_first_path_angle_context_encoder(checkpoint),
        los_angle_context_token_norm_mode=token_norm_mode,
        use_shared_physics_token=_infer_use_shared_physics_token(checkpoint),
        shared_physics_token_residual_scale=_infer_shared_physics_token_residual_scale(checkpoint),
    ).to(device)
    _load_model_state_compatible(model, state)
    model.eval()
    return model


def configuration_features(
    batch_size: int,
    n_freq: int,
    bandwidth_hz: float,
    spacing: float,
    device: torch.device,
) -> torch.Tensor:
    subcarrier_spacing = bandwidth_hz / n_freq
    values = torch.tensor(
        [
            0.0,
            1.0,
            math.log2(2.0) / 4.0,
            math.log2(2.0) / 4.0,
            math.log2(2.0) / 7.0,
            spacing,
            math.log2(n_freq + 1.0) / 8.0,
            math.log10(bandwidth_hz) / 9.0,
            math.log10(subcarrier_spacing) / 6.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    return values.unsqueeze(0).expand(batch_size, -1)


def load_npz_shards(paths: list[Path]) -> tuple[np.ndarray, list[dict[str, object]]]:
    token_blocks: list[np.ndarray] = []
    provenance: list[dict[str, object]] = []
    for path in paths:
        with np.load(path) as data:
            tokens = np.asarray(data["tokens"], dtype=np.float32)
            snapshot_indexes = np.asarray(data["snapshot_indexes"], dtype=np.int64)
        if tokens.ndim != 4 or tokens.shape[1:] != (1, 8, 128):
            raise ValueError(
                f"Expected tokens [snapshots,1,8,128], got {tokens.shape} in {path}"
            )
        if len(snapshot_indexes) != len(tokens):
            raise ValueError(f"Snapshot index count mismatch in {path}")
        token_blocks.append(tokens)
        for local_index, snapshot_index in enumerate(snapshot_indexes):
            provenance.append(
                {
                    "source_npz": str(path),
                    "timestamp": path.stem,
                    "local_snapshot_index": local_index,
                    "source_snapshot_index": int(snapshot_index),
                }
            )
    return np.concatenate(token_blocks, axis=0), provenance


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    paths = sorted(args.processed_dir.glob("*.npz"))
    if args.max_npz is not None:
        paths = paths[: args.max_npz]
    if not paths:
        raise FileNotFoundError(f"No NPZ files found under {args.processed_dir}")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary")
    model = build_model(checkpoint, device=device)
    tokens_np, provenance = load_npz_shards(paths)
    predictions: list[np.ndarray] = []
    feature_norms: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(tokens_np), args.batch_size):
            stop = min(start + args.batch_size, len(tokens_np))
            tokens = torch.from_numpy(tokens_np[start:stop]).to(device)
            batch_size = len(tokens)
            beam_positions = torch.zeros(batch_size, 1, 2, device=device)
            token_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            freq_bin = torch.full(
                (batch_size,), args.freq_bin, dtype=torch.long, device=device
            )
            bw_bin = torch.full(
                (batch_size,), args.bw_bin, dtype=torch.long, device=device
            )
            spacing = torch.full(
                (batch_size,), args.bandwidth_hz / tokens.shape[-1], device=device
            )
            continuous = bool(getattr(model.csi, "use_continuous_config_encoding", False))
            config_features = (
                configuration_features(
                    batch_size,
                    n_freq=tokens.shape[-1],
                    bandwidth_hz=args.bandwidth_hz,
                    spacing=args.antenna_spacing,
                    device=device,
                )
                if continuous
                else None
            )
            antenna_coordinates = (
                torch.zeros(batch_size, 1, 3, device=device) if continuous else None
            )
            antenna_mask = (
                torch.ones(batch_size, 1, dtype=torch.bool, device=device)
                if continuous
                else None
            )
            features = model.encode_csi(
                tokens,
                beam_positions,
                token_mask,
                freq_bin,
                bw_bin,
                spacing,
                normalize=False,
                config_features=config_features,
                antenna_coordinates=antenna_coordinates,
                antenna_mask=antenna_mask,
            )
            components = model.predict_physics_components(features)
            normalized = components["final"]
            raw = normalized * PHYSICS_TARGET_SCALES.to(device) + PHYSICS_TARGET_OFFSETS.to(device)
            predictions.append(raw.cpu().numpy())
            feature_norms.append(features.norm(dim=1).cpu().numpy())

    prediction_array = np.concatenate(predictions, axis=0)
    feature_norm_array = np.concatenate(feature_norms, axis=0)
    if prediction_array.shape != (len(provenance), len(PHYSICS_TARGET_NAMES)):
        raise RuntimeError(f"Unexpected prediction shape: {prediction_array.shape}")
    if not np.isfinite(prediction_array).all() or not np.isfinite(feature_norm_array).all():
        raise RuntimeError("Checkpoint forward pass produced NaN or Inf")

    output_csv = args.output_csv or args.processed_dir / "checkpoint_smoke_predictions.csv"
    rows: list[dict[str, object]] = []
    for index, source in enumerate(provenance):
        row = {**source, "csi_feature_norm": float(feature_norm_array[index])}
        for target_index, target_name in enumerate(PHYSICS_TARGET_NAMES):
            row[f"pred_{target_name}"] = float(prediction_array[index, target_index])
        sin_value = prediction_array[index, PHYSICS_TARGET_NAMES.index("first_path_aoa_az_sin")]
        cos_value = prediction_array[index, PHYSICS_TARGET_NAMES.index("first_path_aoa_az_cos")]
        row["pred_first_path_aoa_az_deg"] = math.degrees(math.atan2(sin_value, cos_value))
        rows.append(row)
    write_csv(output_csv, rows)

    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Device: {device}")
    print(f"NPZ files: {len(paths)}")
    print(f"Forward samples: {len(rows)}")
    print(f"Prediction shape: {prediction_array.shape}")
    print(f"Finite predictions: {np.isfinite(prediction_array).all()}")
    print(f"Output CSV: {output_csv.resolve()}")
    print(
        "Smoke test passed. Predictions are not accuracy metrics: BUPT ground-truth "
        "labels and phased-array beam mapping are not yet available."
    )


if __name__ == "__main__":
    main()
