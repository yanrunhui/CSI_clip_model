from __future__ import annotations

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    DELAY_POWER_PROFILE_BINS,
    PreprocessedCSIDataset,
    SyntheticCSIDataset,
    collate_fn,
)
from data.tokenizer import CaptionTokenizer  # noqa: E402
from models.encoder import CSIEncoder  # noqa: E402
from models.model import FIRST_PATH_DELAY_POSITION_BINS  # noqa: E402

DELAY_NS_RANGE = (0.0, 3000.0)


def normalize_profile(profile: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    profile = torch.nan_to_num(profile.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    return profile / profile.sum(dim=-1, keepdim=True).clamp(min=eps)


def profile_encoder_input(profile: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(normalize_profile(profile))


class PDPVAE(nn.Module):
    def __init__(
        self,
        bins: int = DELAY_POWER_PROFILE_BINS,
        latent_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.bins = bins
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(bins),
            nn.Linear(bins, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bins),
        )

    def encode(self, profile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(profile_encoder_input(profile))
        return self.mu(hidden), self.logvar(hidden).clamp(min=-8.0, max=8.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, profile: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(profile)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return {"logits": logits, "mu": mu, "logvar": logvar, "z": z}


class CSIPDPLatentRegressor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        token_norm_mode: str = "std",
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.csi = CSIEncoder(
            d_token=8,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        beam_positions: torch.Tensor,
        token_mask: torch.Tensor,
        freq_bin: torch.Tensor,
        bw_bin: torch.Tensor,
        subcarrier_spacing: torch.Tensor,
    ) -> torch.Tensor:
        features = self.csi(
            tokens,
            beam_positions,
            token_mask,
            freq_bin,
            bw_bin,
            subcarrier_spacing,
        )
        return self.head(features)


def pdp_vae_loss(
    logits: torch.Tensor,
    target_profile: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    mse_weight: float,
) -> dict[str, torch.Tensor]:
    target = normalize_profile(target_profile)
    log_prob = F.log_softmax(logits, dim=-1)
    reconstruction_ce = -(target * log_prob).sum(dim=-1).mean()
    reconstruction_mse = F.mse_loss(torch.softmax(logits, dim=-1), target)
    kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1).mean()
    kl = kl / max(int(mu.shape[-1]), 1)
    return {
        "loss": reconstruction_ce + mse_weight * reconstruction_mse + beta * kl,
        "reconstruction_ce": reconstruction_ce,
        "reconstruction_mse": reconstruction_mse,
        "kl": kl,
    }


def profile_features(
    profile: torch.Tensor,
    first_path_peak_ratio: float,
    min_first_path_mass: float,
) -> dict[str, torch.Tensor]:
    profile = normalize_profile(profile)
    device = profile.device
    dtype = profile.dtype
    bins = profile.shape[-1]
    centers = torch.linspace(
        DELAY_NS_RANGE[0],
        DELAY_NS_RANGE[1],
        bins + 1,
        device=device,
        dtype=dtype,
    )
    centers = 0.5 * (centers[:-1] + centers[1:])
    max_mass = profile.max(dim=-1).values
    valid_profile = profile.sum(dim=-1) > 0.0
    threshold = torch.maximum(
        max_mass * float(first_path_peak_ratio),
        torch.full_like(max_mass, float(min_first_path_mass)),
    )
    first_mask = profile >= threshold.unsqueeze(-1)
    large_index = torch.full((bins,), bins, device=device, dtype=torch.long)
    indices = torch.arange(bins, device=device, dtype=torch.long)
    first_idx = torch.where(first_mask, indices.unsqueeze(0), large_index.unsqueeze(0)).min(dim=-1).values
    has_first = valid_profile & (first_idx < bins)
    safe_first_idx = first_idx.clamp(max=bins - 1)
    first_delay_ns = centers[safe_first_idx]
    first_delay_ns = torch.where(has_first, first_delay_ns, torch.full_like(first_delay_ns, float("nan")))
    first_power_rel_db = 10.0 * profile.gather(1, safe_first_idx.unsqueeze(1)).squeeze(1).clamp(min=1e-12).log10()
    first_power_rel_db = torch.where(
        has_first,
        first_power_rel_db,
        torch.full_like(first_power_rel_db, float("nan")),
    )
    mean_delay = (profile * centers.unsqueeze(0)).sum(dim=-1)
    delay_spread_ns = torch.sqrt(
        (profile * (centers.unsqueeze(0) - mean_delay.unsqueeze(-1)).square()).sum(dim=-1).clamp(min=0.0)
    )
    delay_spread_ns = torch.where(
        valid_profile,
        delay_spread_ns,
        torch.full_like(delay_spread_ns, float("nan")),
    )
    return {
        "first_delay_ns": first_delay_ns,
        "delay_spread_ns": delay_spread_ns,
        "first_power_rel_db": first_power_rel_db,
    }


def first_delay_bins(delay_ns: torch.Tensor) -> torch.Tensor:
    targets = torch.full_like(delay_ns, fill_value=-1, dtype=torch.long)
    for idx, (_, lower, upper) in enumerate(FIRST_PATH_DELAY_POSITION_BINS):
        upper_mask = delay_ns <= upper if idx == len(FIRST_PATH_DELAY_POSITION_BINS) - 1 else delay_ns < upper
        mask = torch.isfinite(delay_ns) & (delay_ns >= lower) & upper_mask
        targets = torch.where(mask, torch.full_like(targets, idx), targets)
    return targets


def mean_abs_error(predicted: torch.Tensor, target: torch.Tensor) -> float:
    mask = torch.isfinite(predicted) & torch.isfinite(target)
    if not bool(mask.any()):
        return float("nan")
    return float((predicted[mask] - target[mask]).abs().mean().detach().cpu())


def bin_accuracy(predicted_delay_ns: torch.Tensor, target_delay_ns: torch.Tensor) -> float:
    predicted = first_delay_bins(predicted_delay_ns)
    target = first_delay_bins(target_delay_ns)
    mask = (predicted >= 0) & (target >= 0)
    if not bool(mask.any()):
        return float("nan")
    return float((predicted[mask] == target[mask]).float().mean().detach().cpu())


@torch.no_grad()
def evaluate_profile_reconstruction(
    vae: PDPVAE,
    loader: DataLoader,
    device: torch.device,
    first_path_peak_ratio: float,
    min_first_path_mass: float,
) -> dict[str, float]:
    vae.eval()
    original_features = []
    reconstructed_features = []
    raw_first_delay = []
    raw_first_power = []
    losses = []
    for profile, first_delay_ns, first_power_dbw in loader:
        profile = profile.to(device)
        outputs = vae(profile)
        loss_parts = pdp_vae_loss(
            outputs["logits"],
            profile,
            outputs["mu"],
            outputs["logvar"],
            beta=0.0,
            mse_weight=0.0,
        )
        losses.append(loss_parts["reconstruction_ce"].detach().cpu())
        reconstruction = torch.softmax(outputs["logits"], dim=-1)
        original_features.append(
            {
                key: value.detach().cpu()
                for key, value in profile_features(
                    profile,
                    first_path_peak_ratio=first_path_peak_ratio,
                    min_first_path_mass=min_first_path_mass,
                ).items()
            }
        )
        reconstructed_features.append(
            {
                key: value.detach().cpu()
                for key, value in profile_features(
                    reconstruction,
                    first_path_peak_ratio=first_path_peak_ratio,
                    min_first_path_mass=min_first_path_mass,
                ).items()
            }
        )
        raw_first_delay.append(first_delay_ns)
        raw_first_power.append(first_power_dbw)

    original = {
        key: torch.cat([features[key] for features in original_features])
        for key in original_features[0]
    }
    reconstructed = {
        key: torch.cat([features[key] for features in reconstructed_features])
        for key in reconstructed_features[0]
    }
    raw_delay = torch.cat(raw_first_delay)
    raw_power = torch.cat(raw_first_power)
    return {
        "reconstruction_ce": float(torch.stack(losses).mean()),
        "recon_vs_pdp_first_delay_mae_ns": mean_abs_error(
            reconstructed["first_delay_ns"],
            original["first_delay_ns"],
        ),
        "recon_vs_pdp_delay_spread_mae_ns": mean_abs_error(
            reconstructed["delay_spread_ns"],
            original["delay_spread_ns"],
        ),
        "recon_vs_pdp_first_power_rel_mae_db": mean_abs_error(
            reconstructed["first_power_rel_db"],
            original["first_power_rel_db"],
        ),
        "recon_vs_pdp_first_delay_bin_accuracy": bin_accuracy(
            reconstructed["first_delay_ns"],
            original["first_delay_ns"],
        ),
        "pdp_vs_label_first_delay_mae_ns": mean_abs_error(original["first_delay_ns"], raw_delay),
        "recon_vs_label_first_delay_mae_ns": mean_abs_error(reconstructed["first_delay_ns"], raw_delay),
        "pdp_vs_label_first_delay_bin_accuracy": bin_accuracy(original["first_delay_ns"], raw_delay),
        "recon_vs_label_first_delay_bin_accuracy": bin_accuracy(reconstructed["first_delay_ns"], raw_delay),
        "pdp_relative_first_power_vs_label_power_mae_db": mean_abs_error(
            original["first_power_rel_db"],
            raw_power,
        ),
    }


@torch.no_grad()
def evaluate_csi_to_latent(
    model: CSIPDPLatentRegressor,
    vae: PDPVAE,
    loader: DataLoader,
    device: torch.device,
    first_path_peak_ratio: float,
    min_first_path_mass: float,
) -> dict[str, float]:
    model.eval()
    vae.eval()
    original_features = []
    reconstructed_features = []
    raw_first_delay = []
    latent_mse_values = []
    for batch in loader:
        batch = move_batch(batch, device)
        target_profile = batch["delay_power_profile"]
        target_mu, _ = vae.encode(target_profile)
        predicted_z = model(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
        )
        latent_mse_values.append(F.mse_loss(predicted_z, target_mu).detach().cpu())
        reconstruction = torch.softmax(vae.decode(predicted_z), dim=-1)
        original_features.append(
            {
                key: value.detach().cpu()
                for key, value in profile_features(
                    target_profile,
                    first_path_peak_ratio=first_path_peak_ratio,
                    min_first_path_mass=min_first_path_mass,
                ).items()
            }
        )
        reconstructed_features.append(
            {
                key: value.detach().cpu()
                for key, value in profile_features(
                    reconstruction,
                    first_path_peak_ratio=first_path_peak_ratio,
                    min_first_path_mass=min_first_path_mass,
                ).items()
            }
        )
        raw_first_delay.append(batch["physics_raw_targets"][:, 4].detach().cpu())

    original = {
        key: torch.cat([features[key] for features in original_features])
        for key in original_features[0]
    }
    reconstructed = {
        key: torch.cat([features[key] for features in reconstructed_features])
        for key in reconstructed_features[0]
    }
    raw_delay = torch.cat(raw_first_delay)
    return {
        "csi_latent_mse": float(torch.stack(latent_mse_values).mean()),
        "csi_recon_vs_pdp_first_delay_mae_ns": mean_abs_error(
            reconstructed["first_delay_ns"],
            original["first_delay_ns"],
        ),
        "csi_recon_vs_pdp_delay_spread_mae_ns": mean_abs_error(
            reconstructed["delay_spread_ns"],
            original["delay_spread_ns"],
        ),
        "csi_recon_vs_pdp_first_power_rel_mae_db": mean_abs_error(
            reconstructed["first_power_rel_db"],
            original["first_power_rel_db"],
        ),
        "csi_recon_vs_pdp_first_delay_bin_accuracy": bin_accuracy(
            reconstructed["first_delay_ns"],
            original["first_delay_ns"],
        ),
        "csi_recon_vs_label_first_delay_mae_ns": mean_abs_error(
            reconstructed["first_delay_ns"],
            raw_delay,
        ),
        "csi_recon_vs_label_first_delay_bin_accuracy": bin_accuracy(
            reconstructed["first_delay_ns"],
            raw_delay,
        ),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def load_profile_tensors(
    data_path: str,
    limit_samples: int | None,
    min_profile_mass: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    samples = dataset.samples[:limit_samples] if limit_samples is not None else dataset.samples
    profiles = []
    first_delay_ns = []
    first_power_dbw = []
    kept_samples = []
    for sample in samples:
        profile = sample.delay_power_profile.float()
        if float(profile.sum()) < min_profile_mass:
            continue
        profiles.append(profile)
        kept_samples.append(sample)
        first_delay_ns.append(
            float(sample.first_path_delay_s) * 1e9
            if math.isfinite(float(sample.first_path_delay_s))
            else float("nan")
        )
        first_power_dbw.append(float(sample.first_path_power_dbw))
    if not profiles:
        raise ValueError("No samples with non-empty delay_power_profile were found.")
    return (
        torch.stack(profiles),
        torch.tensor(first_delay_ns, dtype=torch.float32),
        torch.tensor(first_power_dbw, dtype=torch.float32),
        kept_samples,
    )


def split_indices(count: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    val_count = max(1, int(round(count * val_fraction)))
    val_count = min(val_count, count - 1) if count > 1 else 1
    return permutation[val_count:], permutation[:val_count]


def make_profile_loader(
    profiles: torch.Tensor,
    first_delay_ns: torch.Tensor,
    first_power_dbw: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(profiles[indices], first_delay_ns[indices], first_power_dbw[indices])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_csi_loader(
    samples: list,
    indices: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    selected = [samples[int(index)] for index in indices.tolist()]
    tokenizer = CaptionTokenizer()
    return DataLoader(
        SyntheticCSIDataset(selected),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )


def train_pdp_vae(
    args: argparse.Namespace,
    profiles: torch.Tensor,
    first_delay_ns: torch.Tensor,
    first_power_dbw: torch.Tensor,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    device: torch.device,
    output_dir: Path,
) -> PDPVAE:
    model = PDPVAE(
        bins=profiles.shape[-1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_profile_loader(
        profiles,
        first_delay_ns,
        first_power_dbw,
        train_indices,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_profile_loader(
        profiles,
        first_delay_ns,
        first_power_dbw,
        val_indices,
        batch_size=args.batch_size,
        shuffle=False,
    )
    log_path = output_dir / "pdp_vae_log.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for profile, _, _ in train_loader:
            profile = profile.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(profile)
            losses = pdp_vae_loss(
                outputs["logits"],
                profile,
                outputs["mu"],
                outputs["logvar"],
                beta=args.beta,
                mse_weight=args.mse_weight,
            )
            losses["loss"].backward()
            optimizer.step()
            epoch_losses.append({key: float(value.detach().cpu()) for key, value in losses.items()})
        val_metrics = evaluate_profile_reconstruction(
            model,
            val_loader,
            device=device,
            first_path_peak_ratio=args.first_path_peak_ratio,
            min_first_path_mass=args.min_first_path_mass,
        )
        train_loss = sum(item["loss"] for item in epoch_losses) / max(len(epoch_losses), 1)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        print(
            f"pdp_vae epoch={epoch:03d}/{args.epochs} "
            f"loss={train_loss:.4f} "
            f"recon_first_delay_mae={val_metrics['recon_vs_pdp_first_delay_mae_ns']:.2f}ns "
            f"recon_bin_acc={val_metrics['recon_vs_pdp_first_delay_bin_accuracy']:.4f}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    checkpoint = {
        "model_state": model.state_dict(),
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "bins": profiles.shape[-1],
        "delay_ns_range": DELAY_NS_RANGE,
        "first_path_peak_ratio": args.first_path_peak_ratio,
        "min_first_path_mass": args.min_first_path_mass,
    }
    torch.save(checkpoint, output_dir / "pdp_vae.pt")
    metrics = evaluate_profile_reconstruction(
        model,
        val_loader,
        device=device,
        first_path_peak_ratio=args.first_path_peak_ratio,
        min_first_path_mass=args.min_first_path_mass,
    )
    with (output_dir / "pdp_vae_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return model


def train_csi_to_latent(
    args: argparse.Namespace,
    vae: PDPVAE,
    samples: list,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    device: torch.device,
    output_dir: Path,
) -> None:
    for parameter in vae.parameters():
        parameter.requires_grad = False
    vae.eval()
    model = CSIPDPLatentRegressor(
        latent_dim=vae.latent_dim,
        token_norm_mode=args.token_norm_mode,
        hidden_dim=args.csi_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.csi_lr, weight_decay=args.weight_decay)
    train_loader = make_csi_loader(samples, train_indices, args.batch_size, shuffle=True)
    val_loader = make_csi_loader(samples, val_indices, args.batch_size, shuffle=False)
    log_path = output_dir / "csi_to_pdp_latent_log.jsonl"
    best_metric = float("inf")
    best_metrics = None
    best_epoch = 0
    for epoch in range(1, args.csi_epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            profile = batch["delay_power_profile"]
            with torch.no_grad():
                target_mu, _ = vae.encode(profile)
            predicted_z = model(
                batch["tokens"],
                batch["beam_positions"],
                batch["token_mask"],
                batch["freq_bin"],
                batch["bw_bin"],
                batch["subcarrier_spacing"],
            )
            logits = vae.decode(predicted_z)
            target = normalize_profile(profile)
            reconstruction_ce = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            latent_mse = F.mse_loss(predicted_z, target_mu)
            loss = args.csi_reconstruction_weight * reconstruction_ce + args.csi_latent_weight * latent_mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "reconstruction_ce": float(reconstruction_ce.detach().cpu()),
                    "latent_mse": float(latent_mse.detach().cpu()),
                }
            )
        val_metrics = evaluate_csi_to_latent(
            model,
            vae,
            val_loader,
            device=device,
            first_path_peak_ratio=args.first_path_peak_ratio,
            min_first_path_mass=args.min_first_path_mass,
        )
        train_loss = sum(item["loss"] for item in epoch_losses) / max(len(epoch_losses), 1)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        current_metric = float(val_metrics["csi_recon_vs_pdp_first_delay_mae_ns"])
        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_metrics = dict(row)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vae_checkpoint": str(output_dir / "pdp_vae.pt"),
                    "latent_dim": vae.latent_dim,
                    "token_norm_mode": args.token_norm_mode,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "best_metric_name": "csi_recon_vs_pdp_first_delay_mae_ns",
                },
                output_dir / "csi_to_pdp_latent_best.pt",
            )
            with (output_dir / "csi_to_pdp_latent_best_metrics.json").open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(best_metrics, handle, indent=2, sort_keys=True)
        print(
            f"csi_to_latent epoch={epoch:03d}/{args.csi_epochs} "
            f"loss={train_loss:.4f} "
            f"csi_recon_first_delay_mae={val_metrics['csi_recon_vs_pdp_first_delay_mae_ns']:.2f}ns "
            f"csi_recon_bin_acc={val_metrics['csi_recon_vs_pdp_first_delay_bin_accuracy']:.4f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save(
        {
            "model_state": model.state_dict(),
            "vae_checkpoint": str(output_dir / "pdp_vae.pt"),
            "latent_dim": vae.latent_dim,
            "token_norm_mode": args.token_norm_mode,
        },
        output_dir / "csi_to_pdp_latent.pt",
    )
    metrics = evaluate_csi_to_latent(
        model,
        vae,
        val_loader,
        device=device,
        first_path_peak_ratio=args.first_path_peak_ratio,
        min_first_path_mass=args.min_first_path_mass,
    )
    with (output_dir / "csi_to_pdp_latent_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PDP VAE and optionally a CSI-to-PDP-latent regressor.",
    )
    parser.add_argument("--data-path", default="artifacts/d2los_400k_coarse_k_cap5000_drop8_23421_train.pt")
    parser.add_argument("--output-dir", default="artifacts/pdp_vae_experiment")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-profile-mass", type=float, default=1e-8)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--mse-weight", type=float, default=10.0)
    parser.add_argument("--first-path-peak-ratio", type=float, default=0.02)
    parser.add_argument("--min-first-path-mass", type=float, default=1e-5)
    parser.add_argument("--run-csi-to-latent", action="store_true")
    parser.add_argument("--csi-epochs", type=int, default=20)
    parser.add_argument("--csi-lr", type=float, default=3e-4)
    parser.add_argument("--csi-hidden-dim", type=int, default=256)
    parser.add_argument("--csi-latent-weight", type=float, default=1.0)
    parser.add_argument("--csi-reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--token-norm-mode", choices=("none", "std", "l2"), default="std")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    profiles, first_delay_ns, first_power_dbw, samples = load_profile_tensors(
        args.data_path,
        limit_samples=args.limit_samples,
        min_profile_mass=args.min_profile_mass,
    )
    train_indices, val_indices = split_indices(len(samples), args.val_fraction, args.seed)
    print(
        f"loaded_samples={len(samples)} train={len(train_indices)} val={len(val_indices)} "
        f"profile_bins={profiles.shape[-1]} device={device}",
        flush=True,
    )
    vae = train_pdp_vae(
        args,
        profiles=profiles,
        first_delay_ns=first_delay_ns,
        first_power_dbw=first_power_dbw,
        train_indices=train_indices,
        val_indices=val_indices,
        device=device,
        output_dir=output_dir,
    )
    if args.run_csi_to_latent:
        train_csi_to_latent(
            args,
            vae=vae,
            samples=samples,
            train_indices=train_indices,
            val_indices=val_indices,
            device=device,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
