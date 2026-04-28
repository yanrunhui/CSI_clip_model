from __future__ import annotations

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
