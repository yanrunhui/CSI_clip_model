from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    DELAY_POWER_PROFILE_BINS,
    PHYSICS_TARGET_NAMES,
    PHYSICS_TARGET_OFFSETS,
    PHYSICS_TARGET_SCALES,
    PreprocessedCSIDataset,
    apply_semantic_key_mode,
    normalize_physics_targets,
    physics_raw_values,
    semantic_key_mode_choices,
)
from models.encoder import CSIEncoder  # noqa: E402


REFLECTION_COUNT_BINS = (
    ("0_5", 0.0, 6.0),
    ("6_7", 6.0, 8.0),
    ("8_10", 8.0, 11.0),
    ("11_13", 11.0, 14.0),
    ("14_plus", 14.0, float("inf")),
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    indices: tuple[int, ...]
    output_dim: int
    metric_prefix: str
    kind: str = "scalar"


def _target_index(name: str) -> int:
    return PHYSICS_TARGET_NAMES.index(name)


TARGET_SPECS = {
    "first_path_delay": TargetSpec(
        name="first_path_delay",
        indices=(_target_index("first_path_delay_ns"),),
        output_dim=1,
        metric_prefix="first_path_delay",
    ),
    "k_factor": TargetSpec(
        name="k_factor",
        indices=(_target_index("k_factor_db"),),
        output_dim=1,
        metric_prefix="k_factor_db",
    ),
    "reflection_count": TargetSpec(
        name="reflection_count",
        indices=(_target_index("reflection_count"),),
        output_dim=1,
        metric_prefix="reflection_count",
        kind="reflection_count",
    ),
    "first_path_angle": TargetSpec(
        name="first_path_angle",
        indices=(
            _target_index("first_path_aoa_az_sin"),
            _target_index("first_path_aoa_az_cos"),
        ),
        output_dim=2,
        metric_prefix="first_path_angle",
        kind="angle",
    ),
    "first_path_power": TargetSpec(
        name="first_path_power",
        indices=(_target_index("first_path_power_dbw"),),
        output_dim=1,
        metric_prefix="first_path_power",
    ),
}

BASELINE_MODEL_NAMES = (
    "flattened_mlp",
    "csi_encoder_single_task",
    "cnn_baseline",
    "pdp_ifft_mlp",
    "transformer_no_branches",
)

EXTENDED_BASELINE_MODEL_NAMES = (
    "cnn_baseline",
    "pdp_ifft_mlp",
    "transformer_no_branches",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def runtime_metadata(model: nn.Module, device: torch.device, seed: int) -> dict[str, int | str | bool | None]:
    cuda_available = torch.cuda.is_available()
    gpu_name = "none"
    gpu_count = 0
    cuda_device_capability = "none"
    if cuda_available:
        device_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        gpu_count = torch.cuda.device_count()
        cuda_device_capability = ".".join(
            str(part) for part in torch.cuda.get_device_capability(device_index)
        )
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "cuda_device_capability": cuda_device_capability,
        "device": str(device),
        "seed": seed,
        "amp_enabled": False,
        "model_parameters": count_parameters(model),
        "model_trainable_parameters": count_trainable_parameters(model),
    }


def _sample_delay_spread_ns(sample) -> float:
    if hasattr(sample, "delay_spread_ns"):
        value = getattr(sample, "delay_spread_ns")
        scale = 1.0
    elif hasattr(sample, "delay_spread_s"):
        value = getattr(sample, "delay_spread_s")
        scale = 1e9
    elif hasattr(sample, "delay_spread"):
        value = getattr(sample, "delay_spread")
        scale = 1e9
    else:
        return math.nan
    try:
        value = float(value) * scale
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def filter_samples_by_min_class_size(samples, min_class_size: int):
    if min_class_size <= 1:
        return samples
    key_counts = Counter(sample.semantic_key for sample in samples)
    filtered = [sample for sample in samples if key_counts[sample.semantic_key] >= min_class_size]
    if not filtered:
        raise ValueError(
            f"No samples remain after filtering semantic classes with min_class_size={min_class_size}."
        )
    return filtered


def filter_samples_by_max_delay_spread(samples, max_delay_spread_ns: float | None):
    if max_delay_spread_ns is None:
        return samples
    if max_delay_spread_ns <= 0.0:
        raise ValueError("--max-delay-spread-ns must be positive.")
    filtered = [
        sample
        for sample in samples
        if math.isfinite(_sample_delay_spread_ns(sample))
        and _sample_delay_spread_ns(sample) < max_delay_spread_ns
    ]
    if not filtered:
        raise ValueError(
            "No samples remain after applying "
            f"max_delay_spread_ns={max_delay_spread_ns}."
        )
    return filtered


def load_samples(
    path: str,
    limit: int | None = None,
    *,
    semantic_key_mode: str = "full",
    min_class_size: int = 1,
    max_delay_spread_ns: float | None = None,
):
    samples = PreprocessedCSIDataset.from_pt(path).samples
    original_count = len(samples)
    samples = apply_semantic_key_mode(samples, semantic_key_mode)
    samples = filter_samples_by_min_class_size(samples, min_class_size)
    samples = filter_samples_by_max_delay_spread(samples, max_delay_spread_ns)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No samples loaded from {path}.")
    if len(samples) != original_count:
        print(
            "baseline_filtered_samples="
            f"path:{path} original:{original_count} final:{len(samples)} "
            f"semantic_key_mode:{semantic_key_mode} "
            f"min_class_size:{min_class_size} "
            f"max_delay_spread_ns:{max_delay_spread_ns}"
        )
    return samples


def make_collate(max_tokens: int, spec: TargetSpec):
    def collate(batch):
        batch_size = len(batch)
        d_token = batch[0].tokens.shape[1]
        n_freq = batch[0].tokens.shape[2]
        tokens = torch.zeros(batch_size, max_tokens, d_token, n_freq, dtype=batch[0].tokens.dtype)
        beam_positions = torch.zeros(batch_size, max_tokens, 2, dtype=torch.float32)
        token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
        freq_bin = torch.zeros(batch_size, dtype=torch.long)
        bw_bin = torch.zeros(batch_size, dtype=torch.long)
        subcarrier_spacing = torch.zeros(batch_size, dtype=torch.float32)
        delay_power_profile = torch.zeros(batch_size, DELAY_POWER_PROFILE_BINS, dtype=torch.float32)
        target = torch.zeros(batch_size, spec.output_dim, dtype=torch.float32)
        raw_target = torch.zeros(batch_size, spec.output_dim, dtype=torch.float32)
        target_mask = torch.zeros(batch_size, dtype=torch.bool)

        for row, sample in enumerate(batch):
            n_tokens = min(sample.n_tokens, max_tokens)
            tokens[row, :n_tokens] = sample.tokens[:n_tokens]
            beam_positions[row, :n_tokens] = sample.beam_positions[:n_tokens]
            token_mask[row, :n_tokens] = True
            freq_bin[row] = int(sample.freq_bin)
            bw_bin[row] = int(sample.bw_bin)
            subcarrier_spacing[row] = float(sample.subcarrier_spacing_hz)
            sample_profile = getattr(sample, "delay_power_profile", None)
            if isinstance(sample_profile, torch.Tensor):
                profile_width = min(DELAY_POWER_PROFILE_BINS, sample_profile.numel())
                delay_power_profile[row, :profile_width] = sample_profile.flatten()[:profile_width]

            raw_values = physics_raw_values(sample)
            normalized, mask = normalize_physics_targets(raw_values)
            indices = torch.tensor(spec.indices, dtype=torch.long)
            target[row] = normalized[indices]
            raw_target[row] = raw_values[indices]
            target_mask[row] = bool(mask[indices].all() and torch.isfinite(raw_values[indices]).all())

        return {
            "tokens": tokens,
            "beam_positions": beam_positions,
            "token_mask": token_mask,
            "freq_bin": freq_bin,
            "bw_bin": bw_bin,
            "subcarrier_spacing": subcarrier_spacing,
            "delay_power_profile": delay_power_profile,
            "target": target,
            "raw_target": raw_target,
            "target_mask": target_mask,
        }

    return collate


class FlattenedCSIMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch["tokens"].flatten(1))


class CSIEncoderSingleTask(nn.Module):
    def __init__(
        self,
        *,
        d_token: int,
        output_dim: int,
        token_norm_mode: str,
        d_model: int,
        d_clip: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = CSIEncoder(
            d_token=d_token,
            d_model=d_model,
            d_clip=d_clip,
            token_norm_mode=token_norm_mode,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_clip),
            nn.Linear(d_clip, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.encoder(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
        )
        return self.head(features)


class CNNBaseline(nn.Module):
    def __init__(self, d_token: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        channels = max(32, min(hidden_dim // 4, 128))
        self.features = nn.Sequential(
            nn.Conv2d(d_token, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["token_mask"].to(dtype=batch["tokens"].dtype).unsqueeze(1).unsqueeze(-1)
        x = batch["tokens"].permute(0, 2, 1, 3) * mask
        return self.head(self.features(x))


class PDPFeatureMLP(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        pdp_bins: int,
    ):
        super().__init__()
        self.pdp_bins = pdp_bins
        input_dim = pdp_bins * 2 + 4
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def _token_stats(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        mask = token_mask.unsqueeze(-1).unsqueeze(-1)
        weights = mask.to(dtype=tokens.dtype)
        element_count = (token_mask.sum(dim=1).to(dtype=tokens.dtype) * tokens.shape[2] * tokens.shape[3]).clamp(min=1.0)
        mean = (tokens * weights).sum(dim=(1, 2, 3)) / element_count
        centered = (tokens - mean[:, None, None, None]) * weights
        std = torch.sqrt((centered.square().sum(dim=(1, 2, 3)) / element_count).clamp(min=1e-12))
        masked_tokens = tokens.masked_fill(~mask, 0.0)
        abs_mean = (masked_tokens.abs() * weights).sum(dim=(1, 2, 3)) / element_count
        rms = torch.sqrt((masked_tokens.square().sum(dim=(1, 2, 3)) / element_count).clamp(min=1e-12))
        return torch.stack([mean, std, abs_mean, rms], dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = batch["tokens"]
        token_mask = batch["token_mask"]
        token_weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        frequency_power = tokens.square().mean(dim=2)
        delay_profile = torch.fft.ifft(frequency_power.to(torch.complex64), dim=-1).abs().float()
        delay_profile = delay_profile * token_weights
        delay_profile = delay_profile.sum(dim=1) / token_weights.sum(dim=1).clamp(min=1.0)
        delay_profile = F.adaptive_avg_pool1d(
            delay_profile.unsqueeze(1),
            self.pdp_bins,
        ).squeeze(1)
        provided_profile = F.adaptive_avg_pool1d(
            batch["delay_power_profile"].unsqueeze(1),
            self.pdp_bins,
        ).squeeze(1)
        features = torch.cat(
            [delay_profile, provided_profile, self._token_stats(tokens, token_mask)],
            dim=1,
        )
        return self.net(features)


class TransformerNoBranches(nn.Module):
    def __init__(
        self,
        *,
        d_token: int,
        output_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "transformer_no_branches requires encoder_d_model divisible by "
                f"transformer_heads, got {d_model} and {n_heads}."
            )
        self.input_proj = nn.Linear(d_token, d_model)
        self.beam_proj = nn.Linear(2, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        token_summary = batch["tokens"].mean(dim=-1)
        x = self.input_proj(token_summary) + self.beam_proj(batch["beam_positions"])
        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, batch["token_mask"]], dim=1)
        x = self.transformer(x, src_key_padding_mask=~full_mask)
        return self.head(x[:, 0])


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def normalized_to_raw(prediction: torch.Tensor, spec: TargetSpec) -> torch.Tensor:
    indices = torch.tensor(spec.indices, device=prediction.device)
    scale = PHYSICS_TARGET_SCALES.to(prediction.device)[indices]
    offset = PHYSICS_TARGET_OFFSETS.to(prediction.device)[indices]
    return prediction * scale + offset


def safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) == 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def reflection_labels(raw_count: torch.Tensor) -> torch.Tensor:
    labels = torch.full_like(raw_count, -1, dtype=torch.long)
    rounded = raw_count.round()
    finite = torch.isfinite(raw_count)
    for class_idx, (_, lower, upper) in enumerate(REFLECTION_COUNT_BINS):
        upper_mask = rounded <= upper if math.isinf(upper) else rounded < upper
        mask = finite & (rounded >= lower) & upper_mask
        labels = torch.where(mask, torch.full_like(labels, class_idx), labels)
    return labels


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    spec: TargetSpec,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    raw_predictions = []
    raw_targets = []
    masks = []
    for batch in loader:
        batch = move_batch(batch, device)
        prediction = model(batch)
        raw_prediction = normalized_to_raw(prediction, spec)
        raw_predictions.append(raw_prediction.detach().cpu())
        raw_targets.append(batch["raw_target"].detach().cpu())
        masks.append(batch["target_mask"].detach().cpu())

    pred = torch.cat(raw_predictions, dim=0)
    target = torch.cat(raw_targets, dim=0)
    mask = torch.cat(masks, dim=0).bool()
    result: dict[str, float] = {"count": float(mask.sum().item())}
    if not bool(mask.any()):
        return result

    if spec.kind == "angle":
        pred_vec = F.normalize(pred[mask], dim=-1, eps=1e-6)
        target_vec = F.normalize(target[mask], dim=-1, eps=1e-6)
        pred_angle = torch.atan2(pred_vec[:, 0], pred_vec[:, 1])
        target_angle = torch.atan2(target_vec[:, 0], target_vec[:, 1])
        signed_error = torch.atan2(
            torch.sin(pred_angle - target_angle),
            torch.cos(pred_angle - target_angle),
        )
        abs_error_deg = signed_error.abs() * (180.0 / math.pi)
        result["MAE"] = float(abs_error_deg.mean())
        result["RMSE"] = float(torch.sqrt(abs_error_deg.square().mean()))
        result["pearson"] = safe_pearson(pred_angle, target_angle)
        return result

    pred_scalar = pred[mask, 0]
    target_scalar = target[mask, 0]
    errors = pred_scalar - target_scalar
    abs_errors = errors.abs()
    result["MAE"] = float(abs_errors.mean())
    result["RMSE"] = float(torch.sqrt(errors.square().mean()))
    result["signed_mean"] = float(errors.mean())
    result["pearson"] = safe_pearson(pred_scalar, target_scalar)

    if spec.kind == "reflection_count":
        true_labels = reflection_labels(target_scalar)
        pred_labels = reflection_labels(pred_scalar)
        valid_labels = (true_labels >= 0) & (pred_labels >= 0)
        if bool(valid_labels.any()):
            result["accuracy"] = float((pred_labels[valid_labels] == true_labels[valid_labels]).float().mean())
            result["adjacent_accuracy"] = float((pred_labels[valid_labels] - true_labels[valid_labels]).abs().le(1).float().mean())
        else:
            result["accuracy"] = float("nan")
            result["adjacent_accuracy"] = float("nan")
    return result


def train_one(
    *,
    model_name: str,
    target_name: str,
    train_samples,
    test_samples,
    args,
    device: torch.device,
) -> dict[str, float | str]:
    spec = TARGET_SPECS[target_name]
    max_tokens = max(max(sample.n_tokens for sample in train_samples), max(sample.n_tokens for sample in test_samples))
    d_token = train_samples[0].tokens.shape[1]
    n_freq = train_samples[0].tokens.shape[2]
    collate = make_collate(max_tokens, spec)
    train_loader = DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_samples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    if model_name == "flattened_mlp":
        model = FlattenedCSIMLP(
            input_dim=max_tokens * d_token * n_freq,
            output_dim=spec.output_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    elif model_name == "cnn_baseline":
        model = CNNBaseline(
            d_token=d_token,
            output_dim=spec.output_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    elif model_name == "pdp_ifft_mlp":
        model = PDPFeatureMLP(
            output_dim=spec.output_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            pdp_bins=args.pdp_bins,
        )
    elif model_name == "transformer_no_branches":
        model = TransformerNoBranches(
            d_token=d_token,
            output_dim=spec.output_dim,
            d_model=args.encoder_d_model,
            n_heads=args.transformer_heads,
            n_layers=args.transformer_layers,
            d_ff=args.transformer_ff,
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
        )
    elif model_name == "csi_encoder_single_task":
        model = CSIEncoderSingleTask(
            d_token=d_token,
            output_dim=spec.output_dim,
            token_norm_mode=args.token_norm_mode,
            d_model=args.encoder_d_model,
            d_clip=args.encoder_d_clip,
            hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    model = model.to(device)
    metadata = runtime_metadata(model, device, args.seed)
    training_start_time = datetime.now().astimezone().isoformat(timespec="seconds")
    training_start_perf = time.perf_counter()
    for key, value in metadata.items():
        print(f"baseline_{key}={value}")
    print(f"baseline_training_start_time={training_start_time}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            mask = batch["target_mask"]
            if not bool(mask.any()):
                continue
            loss = F.smooth_l1_loss(prediction[mask], batch["target"][mask], reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_count = int(mask.sum().item())
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            avg_loss = total_loss / max(total_count, 1)
            print(
                f"baseline_train model={model_name} target={target_name} "
                f"epoch={epoch} loss={avg_loss:.6f} count={total_count}"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_elapsed_seconds = time.perf_counter() - training_start_perf
    training_end_time = datetime.now().astimezone().isoformat(timespec="seconds")
    metrics = evaluate_model(model, test_loader, spec, device)
    result: dict[str, float | str] = {
        "model": model_name,
        "target": target_name,
        "train_count": len(train_samples),
        "test_count": len(test_samples),
        "training_start_time": training_start_time,
        "training_end_time": training_end_time,
        "training_elapsed_seconds": training_elapsed_seconds,
        "training_elapsed_hms": format_duration(training_elapsed_seconds),
    }
    result.update(metadata)
    result.update(metrics)

    print(f"baseline_model={model_name}")
    print(f"baseline_target={target_name}")
    print(f"baseline_training_end_time={training_end_time}")
    print(f"baseline_training_elapsed_seconds={training_elapsed_seconds:.2f}")
    print(f"baseline_training_elapsed_hms={format_duration(training_elapsed_seconds)}")
    print(f"baseline_count={int(metrics.get('count', 0.0))}")
    for key in ("MAE", "RMSE", "signed_mean", "pearson", "accuracy", "adjacent_accuracy"):
        if key in metrics:
            print(f"baseline_{target_name}_{key}={metrics[key]:.4f}")
    print("baseline_summary_json=" + json.dumps(result, sort_keys=True))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_name}_{target_name}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "model_name": model_name,
                "target_name": target_name,
                "metrics": metrics,
                "max_tokens": max_tokens,
                "d_token": d_token,
                "n_freq": n_freq,
            },
            model_path,
        )
        print(f"saved_baseline_checkpoint={model_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument(
        "--model",
        choices=(*BASELINE_MODEL_NAMES, "extended", "all"),
        default="flattened_mlp",
    )
    parser.add_argument(
        "--target",
        choices=(*TARGET_SPECS.keys(), "all"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-d-model", type=int, default=384)
    parser.add_argument("--encoder-d-clip", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=6)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-ff", type=int, default=1024)
    parser.add_argument("--pdp-bins", type=int, default=64)
    parser.add_argument("--token-norm-mode", choices=("std", "rms", "none"), default="std")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument(
        "--semantic-key-mode",
        choices=semantic_key_mode_choices(),
        default="full",
        help="Semantic key granularity before min-class filtering.",
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        default=1,
        help="Drop semantic classes with fewer than this many samples.",
    )
    parser.add_argument(
        "--max-delay-spread-ns",
        type=float,
        help="Drop samples whose delay_spread_ns is greater than or equal to this value.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--results-json",
        help="Optional path for the aggregate baseline results JSON. Defaults to output_dir/baseline_results.json.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples = load_samples(
        args.train_data,
        args.limit_train,
        semantic_key_mode=args.semantic_key_mode,
        min_class_size=args.min_class_size,
        max_delay_spread_ns=args.max_delay_spread_ns,
    )
    test_samples = load_samples(
        args.test_data,
        args.limit_test,
        semantic_key_mode=args.semantic_key_mode,
        min_class_size=args.min_class_size,
        max_delay_spread_ns=args.max_delay_spread_ns,
    )
    if args.model == "all":
        model_names = BASELINE_MODEL_NAMES
    elif args.model == "extended":
        model_names = EXTENDED_BASELINE_MODEL_NAMES
    else:
        model_names = (args.model,)
    target_names = tuple(TARGET_SPECS) if args.target == "all" else (args.target,)
    print(f"baseline_device={device}")
    print(f"baseline_train_samples={len(train_samples)}")
    print(f"baseline_test_samples={len(test_samples)}")
    print(f"baseline_models={','.join(model_names)}")
    print(f"baseline_targets={','.join(target_names)}")
    print(f"baseline_semantic_key_mode={args.semantic_key_mode}")
    print(f"baseline_min_class_size={args.min_class_size}")
    print(f"baseline_max_delay_spread_ns={args.max_delay_spread_ns}")

    all_results = []
    for model_name in model_names:
        for target_name in target_names:
            all_results.append(
                train_one(
                    model_name=model_name,
                    target_name=target_name,
                    train_samples=train_samples,
                    test_samples=test_samples,
                    args=args,
                    device=device,
                )
            )
    print("baseline_all_results_json=" + json.dumps(all_results, sort_keys=True))
    results_json_path = (
        Path(args.results_json)
        if args.results_json is not None
        else Path(args.output_dir) / "baseline_results.json"
        if args.output_dir is not None
        else None
    )
    if results_json_path is not None:
        results_json_path.parent.mkdir(parents=True, exist_ok=True)
        results_json_path.write_text(
            json.dumps(all_results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved_baseline_results_json={results_json_path}")
        csv_path = results_json_path.with_suffix(".csv")
        metric_keys = sorted(
            {
                key
                for result in all_results
                for key in result
                if key not in {"model", "target"}
            }
        )
        rows = [",".join(("model", "target", *metric_keys))]
        for result in all_results:
            rows.append(
                ",".join(
                    str(result.get(key, ""))
                    for key in ("model", "target", *metric_keys)
                )
            )
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"saved_baseline_results_csv={csv_path}")


if __name__ == "__main__":
    main()
