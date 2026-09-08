from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from scripts.pretrain import deserialize_prototype_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained CrossConfig-CSI checkpoint and run a forward-only smoke "
            "test on compact BUPT NPZ shards. The shard token layout must match the "
            "checkpoint input width."
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
        help=(
            "Training preprocessing bandwidth bin. Defaults to the same automatic "
            "mapping as DeepMIMO preprocessing: <=64 -> 0, <=128 -> 1, >128 -> 2."
        ),
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


def infer_bw_bin(n_freq: int) -> int:
    if n_freq <= 64:
        return 0
    if n_freq <= 128:
        return 1
    return 2


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


def infer_csi_d_token(state: dict[str, torch.Tensor]) -> int:
    weight = state.get("csi.input_proj.spatial_linear.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(
            "Checkpoint is missing csi.input_proj.spatial_linear.weight; "
            "cannot infer the expected CSI token width."
        )
    return int(weight.shape[1])


def infer_attribute_classifier_shapes(
    state: dict[str, torch.Tensor],
) -> dict[str, int]:
    result: dict[str, int] = {}
    pattern = re.compile(r"^attribute_classifiers\.([^.]+)\.3\.weight$")
    for name, value in state.items():
        match = pattern.match(name)
        if match and isinstance(value, torch.Tensor) and value.ndim == 2:
            result[match.group(1)] = int(value.shape[0])
    return result


def build_model(checkpoint: dict, device: torch.device) -> CSIClip:
    state = checkpoint_state(checkpoint)
    prototype_keys = deserialize_prototype_keys(checkpoint.get("prototype_keys"))
    prototype_count = len(prototype_keys) if prototype_keys else None
    has_semantic_classifier = any(
        key.startswith("semantic_classifier.") for key in state
    )
    token_norm_mode = _infer_token_norm_mode(checkpoint, None)
    continuous = _infer_use_continuous_config_encoding(checkpoint)
    d_token = infer_csi_d_token(state)
    model = CSIClip(
        CSIEncoder(
            d_token=d_token,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
            use_continuous_config_encoding=continuous,
        ),
        PhysicsTextEncoder(vocab_size=max(infer_text_vocab_size(state), 300)),
        num_prototypes=prototype_count if "prototypes" in state else None,
        semantic_num_classes=prototype_count if has_semantic_classifier else None,
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
        attribute_num_classes=infer_attribute_classifier_shapes(state),
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
    token_layout: str,
) -> torch.Tensor:
    subcarrier_spacing = bandwidth_hz / n_freq
    is_native_siso = token_layout == "native_siso"
    values = torch.tensor(
        [
            1.0 if is_native_siso else 0.0,
            0.0 if is_native_siso else 1.0,
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


def load_npz_shards(
    paths: list[Path],
    expected_d_token: int,
) -> tuple[np.ndarray, list[dict[str, object]], str]:
    token_blocks: list[np.ndarray] = []
    provenance: list[dict[str, object]] = []
    layouts: set[str] = set()
    for path in paths:
        with np.load(path) as data:
            tokens = np.asarray(data["tokens"], dtype=np.float32)
            snapshot_indexes = np.asarray(data["snapshot_indexes"], dtype=np.int64)
            layout = str(np.asarray(data.get("token_layout", "legacy_upa_2x2_padded")).item())
        if tokens.ndim != 4 or tokens.shape[1] != 1 or tokens.shape[2] != expected_d_token:
            raise ValueError(
                f"Checkpoint expects tokens [snapshots,1,{expected_d_token},Nf], "
                f"got {tokens.shape} in {path}. Regenerate BUPT shards with the "
                "matching --token-layout."
            )
        if len(snapshot_indexes) != len(tokens):
            raise ValueError(f"Snapshot index count mismatch in {path}")
        token_blocks.append(tokens)
        layouts.add(layout)
        for local_index, snapshot_index in enumerate(snapshot_indexes):
            provenance.append(
                {
                    "source_npz": str(path),
                    "timestamp": path.stem,
                    "local_snapshot_index": local_index,
                    "source_snapshot_index": int(snapshot_index),
                }
            )
    if len(layouts) != 1:
        raise ValueError(f"Processed BUPT shards mix token layouts: {sorted(layouts)}")
    return np.concatenate(token_blocks, axis=0), provenance, next(iter(layouts))


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
    expected_d_token = int(model.csi.input_proj.spatial_linear.in_features)
    tokens_np, provenance, token_layout = load_npz_shards(
        paths,
        expected_d_token=expected_d_token,
    )
    bw_bin_value = args.bw_bin if args.bw_bin is not None else infer_bw_bin(tokens_np.shape[-1])
    predictions: list[np.ndarray] = []
    feature_norms: list[np.ndarray] = []
    los_probabilities: list[np.ndarray] = []
    semantic_labels: list[np.ndarray] = []
    attribute_los_probabilities: list[np.ndarray] = []
    attribute_los_labels: list[np.ndarray] = []
    prototype_keys = deserialize_prototype_keys(checkpoint.get("prototype_keys"))
    if prototype_keys:
        los_class_indexes = [
            index
            for index, key in enumerate(prototype_keys)
            if key.los_status.lower() == "los"
        ]
        nlos_class_indexes = [
            index
            for index, key in enumerate(prototype_keys)
            if key.los_status.lower() == "nlos"
        ]
        if not los_class_indexes or not nlos_class_indexes:
            raise ValueError(
                "Checkpoint prototype_keys must contain both LoS and NLoS classes "
                "to produce predicted LoS/NLoS groups."
            )
    else:
        los_class_indexes = []
        nlos_class_indexes = []

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
                (batch_size,), bw_bin_value, dtype=torch.long, device=device
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
                    token_layout=token_layout,
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
            delay_context = model.encode_csi_delay_context(
                tokens,
                token_mask,
                subcarrier_spacing=spacing,
                config_features=config_features,
            )
            first_path_delay_context = model.encode_first_path_delay_context(
                tokens,
                token_mask,
                beam_positions=beam_positions,
                freq_bin=freq_bin,
                bw_bin=bw_bin,
                subcarrier_spacing=spacing,
                config_features=config_features,
            )
            power_context = (
                model.encode_power_context(tokens, token_mask)
                if model.use_power_branch
                else None
            )
            los_angle_context = (
                model.encode_los_angle_context(
                    tokens,
                    beam_positions,
                    token_mask,
                    freq_bin,
                    bw_bin,
                    spacing,
                )
                if model.use_los_angle_context_encoder
                else None
            )
            first_path_angle_context = (
                model.encode_first_path_angle_context(
                    tokens,
                    beam_positions,
                    token_mask,
                    subcarrier_spacing=spacing,
                )
                if model.use_first_path_angle_context_encoder
                else None
            )
            components = model.predict_physics_components(
                features,
                power_context=power_context,
                delay_context=delay_context,
                first_path_delay_context=first_path_delay_context,
                los_angle_context=los_angle_context,
                first_path_angle_context=first_path_angle_context,
            )
            normalized = components["final"]
            raw = normalized * PHYSICS_TARGET_SCALES.to(device) + PHYSICS_TARGET_OFFSETS.to(device)
            predictions.append(raw.cpu().numpy())
            feature_norms.append(features.norm(dim=1).cpu().numpy())
            if prototype_keys:
                if model.prototypes is not None:
                    semantic_logits = (
                        model.logit_scale.exp()
                        * F.normalize(features, dim=-1)
                        @ model.encode_prototypes(normalize=True).T
                    )
                elif model.semantic_classifier is not None:
                    semantic_logits = model.predict_semantic(features)
                else:
                    raise RuntimeError(
                        "Checkpoint has prototype_keys but neither a semantic classifier "
                        "nor learnable prototypes."
                    )
                semantic_probability = torch.softmax(semantic_logits, dim=1)
                los_probabilities.append(
                    semantic_probability[:, los_class_indexes].sum(dim=1).cpu().numpy()
                )
                semantic_labels.append(semantic_logits.argmax(dim=1).cpu().numpy())
            if "los_status" in model.attribute_classifiers:
                attribute_los_logits = model.predict_attributes(features)["los_status"]
                if attribute_los_logits.shape[1] != 2:
                    raise RuntimeError(
                        "los_status attribute classifier must have exactly two outputs "
                        "in sorted label order [los, nlos]."
                    )
                attribute_los_probability = torch.softmax(
                    attribute_los_logits, dim=1
                )[:, 0]
                attribute_los_probabilities.append(
                    attribute_los_probability.cpu().numpy()
                )
                attribute_los_labels.append(
                    attribute_los_logits.argmax(dim=1).cpu().numpy()
                )

    prediction_array = np.concatenate(predictions, axis=0)
    feature_norm_array = np.concatenate(feature_norms, axis=0)
    los_probability_array = (
        np.concatenate(los_probabilities, axis=0) if los_probabilities else None
    )
    semantic_label_array = (
        np.concatenate(semantic_labels, axis=0) if semantic_labels else None
    )
    attribute_los_probability_array = (
        np.concatenate(attribute_los_probabilities, axis=0)
        if attribute_los_probabilities
        else None
    )
    attribute_los_label_array = (
        np.concatenate(attribute_los_labels, axis=0)
        if attribute_los_labels
        else None
    )
    if prediction_array.shape != (len(provenance), len(PHYSICS_TARGET_NAMES)):
        raise RuntimeError(f"Unexpected prediction shape: {prediction_array.shape}")
    if not np.isfinite(prediction_array).all() or not np.isfinite(feature_norm_array).all():
        raise RuntimeError("Checkpoint forward pass produced NaN or Inf")

    output_csv = args.output_csv or args.processed_dir / "checkpoint_smoke_predictions.csv"
    rows: list[dict[str, object]] = []
    for index, source in enumerate(provenance):
        row = {**source, "csi_feature_norm": float(feature_norm_array[index])}
        if los_probability_array is not None and semantic_label_array is not None:
            los_probability = float(los_probability_array[index])
            semantic_label = int(semantic_label_array[index])
            top1_los_status = prototype_keys[semantic_label].los_status.lower()
            row["pred_los_probability"] = los_probability
            row["pred_los_status"] = top1_los_status
            row["pred_los_confidence"] = (
                los_probability
                if top1_los_status == "los"
                else 1.0 - los_probability
            )
            row["pred_semantic_class_index"] = semantic_label
            row["pred_semantic_class_los_status"] = top1_los_status
            row["pred_prototype_los_probability"] = los_probability
            row["pred_prototype_los_status"] = top1_los_status
        if (
            attribute_los_probability_array is not None
            and attribute_los_label_array is not None
        ):
            attribute_los_probability = float(attribute_los_probability_array[index])
            attribute_los_status = (
                "los" if int(attribute_los_label_array[index]) == 0 else "nlos"
            )
            row["pred_attribute_los_probability"] = attribute_los_probability
            row["pred_attribute_los_status"] = attribute_los_status
            row["pred_los_probability"] = attribute_los_probability
            row["pred_los_status"] = attribute_los_status
            row["pred_los_confidence"] = (
                attribute_los_probability
                if attribute_los_status == "los"
                else 1.0 - attribute_los_probability
            )
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
    print(f"Token layout: {token_layout}")
    print(f"d_token: {expected_d_token}")
    print(f"freq_bin: {args.freq_bin}")
    print(f"bw_bin: {bw_bin_value}")
    print(f"Prediction shape: {prediction_array.shape}")
    print(f"Finite predictions: {np.isfinite(prediction_array).all()}")
    if los_probability_array is not None:
        point_count = len({row["timestamp"] for row in rows})
        predicted_los_snapshots = sum(
            1 for row in rows if row["pred_los_status"] == "los"
        )
        print(
            "Predicted LoS/NLoS snapshot groups: "
            f"LoS={predicted_los_snapshots}, "
            f"NLoS={len(los_probability_array) - predicted_los_snapshots} "
            f"across {point_count} MAT points"
        )
        print(
            "LoS decision source: "
            + (
                "attribute_classifier.los_status"
                if attribute_los_probability_array is not None
                else "learnable_prototype_top1"
            )
        )
    print(f"Output CSV: {output_csv.resolve()}")
    print(
        "Smoke test passed. Predictions are not accuracy metrics: BUPT ground-truth "
        "labels and phased-array beam mapping are not yet available."
    )


if __name__ == "__main__":
    main()
