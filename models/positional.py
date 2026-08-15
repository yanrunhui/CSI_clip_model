from __future__ import annotations

import math

import torch
from torch import nn


class BeamPositionEncoding(nn.Module):
    def __init__(self, d_model: int, n_freq_bands: int = 32):
        super().__init__()
        self.freq_bands = nn.Parameter(torch.randn(n_freq_bands, 2) * 0.1)
        self.proj = nn.Linear(n_freq_bands * 2, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        proj = positions @ self.freq_bands.T
        features = torch.cat([proj.sin(), proj.cos()], dim=-1)
        return self.proj(features)


class FrequencyBandEncoding(nn.Module):
    def __init__(self, d_model: int, n_freq_bins: int = 3, n_bw_bins: int = 3):
        super().__init__()
        self.freq_embed = nn.Embedding(n_freq_bins, d_model)
        self.bw_embed = nn.Embedding(n_bw_bins, d_model)

    def forward(self, freq_bin: torch.Tensor, bw_bin: torch.Tensor) -> torch.Tensor:
        return (self.freq_embed(freq_bin) + self.bw_embed(bw_bin)).unsqueeze(1)

    def frequency_only(self, freq_bin: torch.Tensor) -> torch.Tensor:
        return self.freq_embed(freq_bin).unsqueeze(1)


class ContinuousConfigurationEncoding(nn.Module):
    def __init__(self, d_model: int, input_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        geometry_dim = max(d_model // 2, 32)
        self.geometry_net = nn.Sequential(
            nn.Linear(3, geometry_dim),
            nn.GELU(),
            nn.Linear(geometry_dim, geometry_dim),
        )
        self.geometry_proj = nn.Sequential(
            nn.LayerNorm(geometry_dim * 2),
            nn.Linear(geometry_dim * 2, d_model),
        )

    def forward(
        self,
        features: torch.Tensor,
        antenna_coordinates: torch.Tensor | None = None,
        antenna_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = self.net(features)
        if (
            antenna_coordinates is not None
            and antenna_mask is not None
            and antenna_coordinates.shape[1] > 0
        ):
            geometry = self.geometry_net(
                antenna_coordinates.to(
                    device=features.device,
                    dtype=features.dtype,
                )
            )
            mask = antenna_mask.to(device=features.device).bool()
            weights = mask.unsqueeze(-1).to(dtype=geometry.dtype)
            geometry_mean = (geometry * weights).sum(dim=1)
            geometry_mean = geometry_mean / weights.sum(dim=1).clamp(min=1.0)
            geometry_max = geometry.masked_fill(
                ~mask.unsqueeze(-1),
                torch.finfo(geometry.dtype).min,
            ).amax(dim=1)
            geometry_max = torch.where(
                torch.isfinite(geometry_max),
                geometry_max,
                torch.zeros_like(geometry_max),
            )
            encoded = encoded + self.geometry_proj(
                torch.cat([geometry_mean, geometry_max], dim=-1)
            )
        return encoded.unsqueeze(1)


def normalized_subcarrier_spacing(
    subcarrier_spacing: torch.Tensor,
    *,
    continuous: bool,
) -> torch.Tensor:
    spacing = subcarrier_spacing.clamp(min=0.0)
    if not continuous:
        return (spacing / 480e3).clamp(0.0, 1.0)
    return torch.log1p(spacing) / math.log1p(1e6)
