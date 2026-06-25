from __future__ import annotations

import argparse
import math
import sys
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (
    PHYSICS_TARGET_NAMES,
    PHYSICS_TARGET_OFFSETS,
    PHYSICS_TARGET_SCALES,
    PreprocessedCSIDataset,
    apply_semantic_key_mode,
    collate_fn,
)
from models.encoder import CSIEncoder
from models.model import CSIClip
from models.text_encoder import PhysicsTextEncoder
from scripts.evaluate import (
    _infer_max_delay_spread_ns,
    _infer_min_class_size,
    _infer_semantic_key_mode,
    _infer_token_norm_mode,
    _infer_use_delay_specific_encoder,
    _infer_use_delay_spread_head,
    _infer_use_power_branch,
    _load_model_state_compatible,
    align_samples_to_checkpoint_prototypes,
    build_prototype_bank,
    build_tokenizer,
    filter_samples_by_max_delay_spread,
    filter_samples_by_min_class_size,
)
from scripts.pretrain import assert_checkpoint_prototype_compatibility


FIRST_DELAY_BINS_NS = (
    ("0_25", 0.0, 25.0),
    ("25_50", 25.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("100_200", 100.0, 200.0),
    ("200_400", 200.0, 400.0),
    ("400_600", 400.0, 600.0),
    ("600_800", 600.0, 800.0),
    ("800_1040", 800.0, 1040.0),
    ("1040_1280", 1040.0, 1280.0),
    ("1280_plus", 1280.0, float("inf")),
)


class FirstDelayProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            moved[key] = value
    return moved


def _build_loader_and_model(
    *,
    data_path: str,
    checkpoint: dict,
    checkpoint_path: str,
    batch_size: int,
    device: torch.device,
    semantic_key_mode_override: str | None,
    token_norm_mode_override: str | None,
    min_class_size_override: int | None,
    max_delay_spread_ns_override: float | None,
) -> tuple[DataLoader, CSIClip]:
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    semantic_key_mode = _infer_semantic_key_mode(checkpoint, semantic_key_mode_override)
    token_norm_mode = _infer_token_norm_mode(checkpoint, token_norm_mode_override)
    min_class_size = _infer_min_class_size(checkpoint, min_class_size_override)
    max_delay_spread_ns = _infer_max_delay_spread_ns(checkpoint, max_delay_spread_ns_override)
    use_power_branch = _infer_use_power_branch(checkpoint, None)
    use_delay_spread_head = _infer_use_delay_spread_head(checkpoint)
    use_delay_specific_encoder = _infer_use_delay_specific_encoder(checkpoint)

    samples = apply_semantic_key_mode(dataset.samples, semantic_key_mode)
    samples = filter_samples_by_min_class_size(samples, min_class_size=min_class_size)
    samples = filter_samples_by_max_delay_spread(samples, max_delay_spread_ns)
    samples, checkpoint_prototype_keys = align_samples_to_checkpoint_prototypes(samples, checkpoint)
    tokenizer = build_tokenizer(samples, checkpoint)
    prototype_keys, _, _, _ = build_prototype_bank(
        samples,
        tokenizer,
        prototype_keys_override=checkpoint_prototype_keys,
    )

    loader = DataLoader(
        PreprocessedCSIDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
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
        use_delay_spread_head=use_delay_spread_head,
        use_delay_specific_encoder=use_delay_specific_encoder,
    ).to(device)
    assert_checkpoint_prototype_compatibility(
        checkpoint,
        prototype_keys,
        expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
        context=f"probe checkpoint {checkpoint_path}",
    )
    _load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()
    return loader, model


def _print_metrics(prefix: str, predictions: torch.Tensor, targets: torch.Tensor) -> None:
    errors = predictions.float() - targets.float()
    print(f"{prefix}_count={targets.numel()}")
    print(f"{prefix}_MAE={float(errors.abs().mean()):.4f}")
    print(f"{prefix}_RMSE={float(torch.sqrt(errors.square().mean())):.4f}")
    print(f"{prefix}_signed_mean={float(errors.mean()):.4f}")
    print(f"{prefix}_pearson={_pearson(predictions, targets):.4f}")
    print(
        f"{prefix}_pred_range={float(predictions.min()):.4f},"
        f"{float(predictions.max()):.4f}"
    )
    print(
        f"{prefix}_target_range={float(targets.min()):.4f},"
        f"{float(targets.max()):.4f}"
    )


def _print_delay_bin_metrics(
    prefix: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    print(
        f"{prefix}_bin_order="
        + ",".join(
            f"{label}:{lower:g}-{upper:g}"
            if math.isfinite(upper)
            else f"{label}:{lower:g}-inf"
            for label, lower, upper in FIRST_DELAY_BINS_NS
        )
    )
    for bin_idx, (label, lower, upper) in enumerate(FIRST_DELAY_BINS_NS):
        upper_mask = targets <= upper if bin_idx == len(FIRST_DELAY_BINS_NS) - 1 else targets < upper
        mask = (targets >= lower) & upper_mask
        count = int(mask.sum().item())
        print(f"{prefix}_bin_{label}_count={count}")
        if count == 0:
            print(f"{prefix}_bin_{label}_MAE=nan")
            print(f"{prefix}_bin_{label}_signed_mean=nan")
            print(f"{prefix}_bin_{label}_pred_range=nan,nan")
            continue
        bin_predictions = predictions[mask]
        bin_targets = targets[mask]
        bin_errors = bin_predictions - bin_targets
        print(f"{prefix}_bin_{label}_MAE={float(bin_errors.abs().mean()):.4f}")
        print(f"{prefix}_bin_{label}_signed_mean={float(bin_errors.mean()):.4f}")
        print(
            f"{prefix}_bin_{label}_pred_range={float(bin_predictions.min()):.4f},"
            f"{float(bin_predictions.max()):.4f}"
        )


@torch.no_grad()
def _extract_first_delay_features(
    model: CSIClip,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = []
    targets = []
    head_predictions = []
    target_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
    target_scale = PHYSICS_TARGET_SCALES[target_idx].to(device)
    target_offset = PHYSICS_TARGET_OFFSETS[target_idx].to(device)
    for batch in loader:
        batch = _move_batch(batch, device)
        csi_features = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
        )
        first_delay_context = model.encode_first_path_delay_context(
            batch["tokens"],
            batch["token_mask"],
            subcarrier_spacing=batch.get("subcarrier_spacing"),
        )
        physics_outputs = model.predict_physics_components(
            csi_features,
            first_path_delay_context=first_delay_context,
        )
        head_raw = physics_outputs["first_path_delay_context"] * target_scale + target_offset
        raw_target = batch["physics_raw_targets"][:, target_idx]
        mask = batch["physics_target_mask"][:, target_idx].bool() & torch.isfinite(raw_target)
        if bool(mask.any()):
            features.append(first_delay_context[mask].detach().cpu())
            targets.append(raw_target[mask].detach().cpu())
            head_predictions.append(head_raw[mask].detach().cpu())
    if not features:
        raise ValueError("No valid first_path_delay_ns targets were found.")
    return (
        torch.cat(features, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(head_predictions, dim=0),
    )


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


@torch.no_grad()
def _evaluate_probe(
    probe: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    probe.eval()
    x_norm = ((x.to(device) - feature_mean) / feature_std).to(device)
    pred_norm = probe(x_norm)
    pred = (pred_norm * target_std + target_mean).cpu()
    errors = pred - y
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(torch.sqrt(errors.square().mean())),
        "signed_mean": float(errors.mean()),
        "pearson": _pearson(pred, y),
        "pred_min": float(pred.min()),
        "pred_max": float(pred.max()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
    }


@torch.no_grad()
def _predict_probe_raw(
    probe: nn.Module,
    x: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    probe.eval()
    x_norm = ((x.to(device) - feature_mean) / feature_std).to(device)
    pred_norm = probe(x_norm)
    return (pred_norm * target_std + target_mean).cpu()


def _train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
) -> tuple[FirstDelayProbe, dict[str, float], dict[str, torch.Tensor]]:
    feature_mean = train_x.mean(dim=0, keepdim=True).to(device)
    feature_std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6).to(device)
    target_mean = train_y.mean().to(device)
    target_std = train_y.std().clamp(min=1e-6).to(device)
    train_x_norm = ((train_x.to(device) - feature_mean) / feature_std).float()
    train_y_norm = ((train_y.to(device) - target_mean) / target_std).float()

    probe = FirstDelayProbe(train_x.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(train_x_norm, train_y_norm),
        batch_size=batch_size,
        shuffle=True,
    )
    best_state = None
    best_mae = float("inf")
    best_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0
        total_count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = probe(batch_x)
            loss = torch.nn.functional.smooth_l1_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch_x.shape[0]
            total_count += batch_x.shape[0]
        metrics = _evaluate_probe(
            probe,
            test_x,
            test_y,
            feature_mean,
            feature_std,
            target_mean,
            target_std,
            device,
        )
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_metrics = dict(metrics)
            best_metrics["epoch"] = float(epoch)
            best_metrics["train_loss"] = total_loss / max(total_count, 1)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
        if epoch == 1 or epoch == epochs or epoch % max(epochs // 10, 1) == 0:
            print(
                f"epoch={epoch:03d}/{epochs} "
                f"train_loss={total_loss / max(total_count, 1):.4f} "
                f"test_MAE={metrics['mae']:.4f} "
                f"test_signed={metrics['signed_mean']:.4f} "
                f"test_pearson={metrics['pearson']:.4f}"
            )
    if best_state is not None:
        probe.load_state_dict(best_state)
    normalizer = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }
    return probe, best_metrics, normalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--test-data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--semantic-key-mode")
    parser.add_argument("--token-norm-mode", choices=["std", "rms", "none"])
    parser.add_argument("--min-class-size", type=int)
    parser.add_argument("--max-delay-spread-ns", type=float)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_loader, model = _build_loader_and_model(
        data_path=args.train_data_path,
        checkpoint=checkpoint,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=device,
        semantic_key_mode_override=args.semantic_key_mode,
        token_norm_mode_override=args.token_norm_mode,
        min_class_size_override=args.min_class_size,
        max_delay_spread_ns_override=args.max_delay_spread_ns,
    )
    test_loader, _ = _build_loader_and_model(
        data_path=args.test_data_path,
        checkpoint=checkpoint,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device=device,
        semantic_key_mode_override=args.semantic_key_mode,
        token_norm_mode_override=args.token_norm_mode,
        min_class_size_override=args.min_class_size,
        max_delay_spread_ns_override=args.max_delay_spread_ns,
    )
    print("extracting frozen first_path_delay_context features...")
    train_x, train_y, train_head_pred = _extract_first_delay_features(model, train_loader, device)
    test_x, test_y, test_head_pred = _extract_first_delay_features(model, test_loader, device)
    print(
        f"train_examples={train_x.shape[0]} test_examples={test_x.shape[0]} "
        f"context_dim={train_x.shape[1]}"
    )
    _print_metrics("checkpoint_first_path_delay_head", test_head_pred, test_y)
    _print_delay_bin_metrics("checkpoint_first_path_delay_head", test_head_pred, test_y)
    baseline_pred = torch.full_like(test_y, float(train_y.mean()))
    baseline_errors = baseline_pred - test_y
    print(
        f"mean_baseline_MAE={float(baseline_errors.abs().mean()):.4f} "
        f"mean_baseline_signed={float(baseline_errors.mean()):.4f}"
    )
    probe, best, normalizer = _train_probe(
        train_x,
        train_y,
        test_x,
        test_y,
        device=device,
        epochs=args.epochs,
        batch_size=args.probe_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
    )
    probe_predictions = _predict_probe_raw(
        probe,
        test_x,
        normalizer["feature_mean"],
        normalizer["feature_std"],
        normalizer["target_mean"],
        normalizer["target_std"],
        device,
    )
    print(
        f"best_epoch={int(best['epoch'])} "
        f"first_path_delay_context_probe_MAE={best['mae']:.4f} "
        f"first_path_delay_context_probe_RMSE={best['rmse']:.4f} "
        f"first_path_delay_context_probe_signed_mean={best['signed_mean']:.4f} "
        f"first_path_delay_context_probe_pearson={best['pearson']:.4f} "
        f"first_path_delay_context_probe_pred_range={best['pred_min']:.4f},{best['pred_max']:.4f} "
        f"first_path_delay_context_probe_target_range={best['target_min']:.4f},{best['target_max']:.4f}"
    )
    _print_metrics("first_path_delay_context_probe", probe_predictions, test_y)
    _print_delay_bin_metrics("first_path_delay_context_probe", probe_predictions, test_y)


if __name__ == "__main__":
    main()
