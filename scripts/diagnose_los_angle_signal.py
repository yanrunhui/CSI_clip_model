from __future__ import annotations

import argparse
import math
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    PHYSICS_TARGET_NAMES,
    PreprocessedCSIDataset,
    apply_semantic_key_mode,
    collate_fn,
)
from models.encoder import CSIEncoder  # noqa: E402
from models.model import CSIClip  # noqa: E402
from models.text_encoder import PhysicsTextEncoder  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    _infer_attribute_fields,
    _infer_attribute_remap,
    _infer_semantic_key_mode,
    _infer_token_norm_mode,
    _infer_use_delay_specific_encoder,
    _infer_use_delay_spread_head,
    _infer_use_los_angle_context_encoder,
    _infer_use_power_branch,
    _load_model_state_compatible,
    align_samples_to_checkpoint_prototypes,
    build_attribute_label_maps,
    build_prototype_bank,
    build_tokenizer,
    move_batch,
)
from data.semantic_key import semantic_key_attribute_value  # noqa: E402


def _angle_metrics(prefix: str, predictions: torch.Tensor, targets: torch.Tensor) -> None:
    raw_prediction_norm = torch.linalg.vector_norm(predictions.float(), dim=-1)
    predictions = F.normalize(predictions.float(), dim=-1, eps=1e-6)
    targets = F.normalize(targets.float(), dim=-1, eps=1e-6)
    cosine = (predictions * targets).sum(dim=-1).clamp(-1.0, 1.0)
    pred_angle = torch.atan2(predictions[:, 0], predictions[:, 1])
    target_angle = torch.atan2(targets[:, 0], targets[:, 1])
    signed_error = torch.atan2(
        torch.sin(pred_angle - target_angle),
        torch.cos(pred_angle - target_angle),
    )
    abs_error_deg = signed_error.abs() * (180.0 / math.pi)
    print(f"{prefix}_count={int(abs_error_deg.numel())}")
    print(f"{prefix}_MAE={float(abs_error_deg.mean()):.4f}")
    print(f"{prefix}_median_error={float(abs_error_deg.median()):.4f}")
    print(f"{prefix}_p90_error={float(torch.quantile(abs_error_deg, 0.9)):.4f}")
    for threshold in (5.0, 10.0, 15.0, 30.0):
        threshold_text = str(int(threshold))
        print(
            f"{prefix}_accuracy@{threshold_text}deg="
            f"{float((abs_error_deg <= threshold).float().mean()):.4f}"
        )
    print(f"{prefix}_cosine_mean={float(cosine.mean()):.4f}")
    print(f"{prefix}_signed_mean={float(signed_error.mean() * (180.0 / math.pi)):.4f}")
    print(f"{prefix}_prediction_norm_mean={float(raw_prediction_norm.mean()):.4f}")


def _standardize(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train_features - mean) / std, (test_features - mean) / std


def _ridge_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    train_features, test_features = _standardize(train_features.float(), test_features.float())
    ones_train = torch.ones(train_features.shape[0], 1, dtype=train_features.dtype)
    ones_test = torch.ones(test_features.shape[0], 1, dtype=test_features.dtype)
    train_design = torch.cat([train_features, ones_train], dim=1)
    test_design = torch.cat([test_features, ones_test], dim=1)
    reg = torch.eye(train_design.shape[1], dtype=train_design.dtype) * float(alpha)
    reg[-1, -1] = 0.0
    lhs = train_design.T @ train_design + reg
    rhs = train_design.T @ train_targets.float()
    weights = torch.linalg.solve(lhs, rhs)
    return test_design @ weights


def _knn_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    k: int,
) -> torch.Tensor:
    train_features = F.normalize(train_features.float(), dim=-1, eps=1e-6)
    test_features = F.normalize(test_features.float(), dim=-1, eps=1e-6)
    similarities = test_features @ train_features.T
    topk = similarities.topk(k=min(k, train_features.shape[0]), dim=1)
    weights = F.softmax(topk.values / 0.07, dim=1)
    neighbor_targets = train_targets[topk.indices]
    return (neighbor_targets * weights.unsqueeze(-1)).sum(dim=1)


def _mean_angle_baseline(train_targets: torch.Tensor, count: int) -> torch.Tensor:
    mean_target = train_targets.float().mean(dim=0, keepdim=True)
    return mean_target.expand(count, -1)


def _samples_for_checkpoint(path: str, checkpoint: dict | None):
    dataset = PreprocessedCSIDataset.from_pt(path)
    semantic_key_mode = _infer_semantic_key_mode(checkpoint, None)
    samples = apply_semantic_key_mode(dataset.samples, semantic_key_mode)
    samples, checkpoint_keys = align_samples_to_checkpoint_prototypes(samples, checkpoint)
    return samples, checkpoint_keys


@torch.no_grad()
def _extract_features(
    *,
    samples,
    checkpoint: dict,
    checkpoint_keys,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    tokenizer = build_tokenizer(samples, checkpoint)
    prototype_keys, _, _, _ = build_prototype_bank(
        samples,
        tokenizer,
        prototype_keys_override=checkpoint_keys,
    )
    attribute_remap = _infer_attribute_remap(checkpoint)
    attribute_fields = _infer_attribute_fields(checkpoint)
    attribute_label_maps = build_attribute_label_maps(
        samples,
        attribute_fields,
        attribute_remap=attribute_remap,
    )
    token_norm_mode = _infer_token_norm_mode(checkpoint, None)
    use_power_branch = _infer_use_power_branch(checkpoint, None)
    model = CSIClip(
        CSIEncoder(
            d_token=8,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
        ),
        PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300)),
        num_prototypes=len(prototype_keys),
        semantic_num_classes=len(prototype_keys),
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        use_power_branch=use_power_branch,
        use_delay_spread_head=_infer_use_delay_spread_head(checkpoint),
        use_delay_specific_encoder=_infer_use_delay_specific_encoder(checkpoint),
        use_los_angle_context_encoder=_infer_use_los_angle_context_encoder(checkpoint),
        los_angle_context_token_norm_mode=token_norm_mode,
        attribute_num_classes={
            field: len(label_map)
            for field, label_map in attribute_label_maps.items()
        },
    ).to(device)
    _load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()

    loader = DataLoader(
        PreprocessedCSIDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )

    feature_chunks: dict[str, list[torch.Tensor]] = {
        "csi_raw": [],
        "csi_norm": [],
        "first_path_context": [],
        "los_angle_context": [],
    }
    target_chunks = []
    mask_chunks = []
    head_chunks = []
    semantic_keys = []
    for batch in loader:
        batch = move_batch(batch, device)
        csi_raw = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
        )
        first_path_context = model.encode_first_path_delay_context(
            batch["tokens"],
            batch["token_mask"],
            beam_positions=batch.get("beam_positions"),
            freq_bin=batch.get("freq_bin"),
            bw_bin=batch.get("bw_bin"),
            subcarrier_spacing=batch.get("subcarrier_spacing"),
        )
        los_angle_context = model.encode_los_angle_context(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
        )
        physics_outputs = model.predict_physics_components(
            csi_raw,
            first_path_delay_context=first_path_context,
            los_angle_context=los_angle_context,
        )
        feature_chunks["csi_raw"].append(csi_raw.cpu())
        feature_chunks["csi_norm"].append(F.normalize(csi_raw, dim=-1).cpu())
        feature_chunks["first_path_context"].append(first_path_context.cpu())
        if los_angle_context is not None:
            feature_chunks["los_angle_context"].append(los_angle_context.cpu())
        head_chunks.append(physics_outputs["los_angle_sincos"].cpu())
        target_chunks.append(batch["los_angle_target"].cpu())
        mask_chunks.append(batch["los_angle_target_mask"].cpu())
        semantic_keys.extend(batch["semantic_keys"])

    features = {
        name: torch.cat(chunks, dim=0)
        for name, chunks in feature_chunks.items()
        if chunks
    }
    features["csi_plus_first_path_context"] = torch.cat(
        [features["csi_raw"], features["first_path_context"]],
        dim=1,
    )
    if "los_angle_context" in features:
        features["csi_plus_los_angle_context"] = torch.cat(
            [features["csi_raw"], features["los_angle_context"]],
            dim=1,
        )
    targets = torch.cat(target_chunks, dim=0)
    masks = torch.cat(mask_chunks, dim=0).bool()
    los_mask = torch.tensor(
        [key.los_status == "los" for key in semantic_keys],
        dtype=torch.bool,
    )
    valid_mask = masks & los_mask & torch.isfinite(targets).all(dim=1)
    features = {name: value[valid_mask] for name, value in features.items()}
    features["trained_los_angle_head"] = torch.cat(head_chunks, dim=0)[valid_mask]
    return features, targets[valid_mask]


def diagnose(
    train_data_path: str,
    test_data_path: str,
    checkpoint_path: str,
    batch_size: int,
    device: torch.device,
    ridge_alpha: float,
    knn_k: int,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_samples, train_checkpoint_keys = _samples_for_checkpoint(train_data_path, checkpoint)
    test_samples, test_checkpoint_keys = _samples_for_checkpoint(test_data_path, checkpoint)
    checkpoint_keys = train_checkpoint_keys or test_checkpoint_keys
    train_features, train_targets = _extract_features(
        samples=train_samples,
        checkpoint=checkpoint,
        checkpoint_keys=checkpoint_keys,
        batch_size=batch_size,
        device=device,
    )
    test_features, test_targets = _extract_features(
        samples=test_samples,
        checkpoint=checkpoint,
        checkpoint_keys=checkpoint_keys,
        batch_size=batch_size,
        device=device,
    )

    print(f"train_los_angle_count={train_targets.shape[0]}")
    print(f"test_los_angle_count={test_targets.shape[0]}")
    _angle_metrics(
        "mean_angle_baseline",
        _mean_angle_baseline(train_targets, test_targets.shape[0]),
        test_targets,
    )
    _angle_metrics(
        "trained_los_angle_head",
        test_features["trained_los_angle_head"],
        test_targets,
    )
    for feature_name in (
        "csi_norm",
        "csi_raw",
        "first_path_context",
        "csi_plus_first_path_context",
        "los_angle_context",
        "csi_plus_los_angle_context",
    ):
        if feature_name not in train_features or feature_name not in test_features:
            continue
        ridge_predictions = _ridge_probe(
            train_features[feature_name],
            train_targets,
            test_features[feature_name],
            alpha=ridge_alpha,
        )
        _angle_metrics(
            f"ridge_probe_{feature_name}",
            ridge_predictions,
            test_targets,
        )
        knn_predictions = _knn_probe(
            train_features[feature_name],
            train_targets,
            test_features[feature_name],
            k=knn_k,
        )
        _angle_metrics(
            f"knn_probe_{feature_name}_k{knn_k}",
            knn_predictions,
            test_targets,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--test-data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--knn-k", type=int, default=16)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diagnose(
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=device,
        ridge_alpha=args.ridge_alpha,
        knn_k=args.knn_k,
    )


if __name__ == "__main__":
    main()
