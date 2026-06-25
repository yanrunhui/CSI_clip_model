from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .positional import BeamPositionEncoding, FrequencyBandEncoding


class InputProjection(nn.Module):
    def __init__(
        self,
        d_token: int = 8,
        d_spatial: int = 64,
        d_freq_pool: int = 16,
        d_model: int = 384,
    ):
        super().__init__()
        self.d_freq_pool = d_freq_pool
        self.spatial_linear = nn.Linear(d_token, d_spatial)
        self.film_gamma = nn.Linear(1, d_spatial)
        self.film_beta = nn.Linear(1, d_spatial)
        self.freq_backbone = nn.Sequential(
            nn.Conv1d(d_spatial, d_spatial, kernel_size=7, padding=3, groups=d_spatial),
            nn.GELU(),
            nn.Conv1d(d_spatial, d_spatial, kernel_size=1),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(d_spatial * d_freq_pool * 2, d_model)

    def forward(self, x: torch.Tensor, subcarrier_spacing: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "CSI tokens must have shape (batch, beams, d_token, n_freq), "
                f"got {tuple(x.shape)}."
            )
        expected_d_token = self.spatial_linear.in_features
        if x.shape[2] != expected_d_token:
            raise ValueError(
                "CSI token feature dimension mismatch: "
                f"expected d_token={expected_d_token}, got {x.shape[2]} "
                f"for tokens shape {tuple(x.shape)}. "
                "Use an eval .pt generated with the same preprocessing as training, "
                "or rebuild the model with the matching d_token."
            )
        if subcarrier_spacing.ndim != 1 or subcarrier_spacing.shape[0] != x.shape[0]:
            raise ValueError(
                "subcarrier_spacing must have shape (batch,), "
                f"got {tuple(subcarrier_spacing.shape)} for batch={x.shape[0]}."
            )
        B, K, D, Nf = x.shape
        x = x.reshape(B * K, D, Nf).transpose(-1, -2)
        x = self.spatial_linear(x)

        sc_norm = (subcarrier_spacing / 480e3).clamp(0, 1)
        gamma = self.film_gamma(sc_norm.unsqueeze(-1)).unsqueeze(1).expand(-1, K, -1)
        beta = self.film_beta(sc_norm.unsqueeze(-1)).unsqueeze(1).expand(-1, K, -1)
        gamma = gamma.reshape(B * K, 1, -1)
        beta = beta.reshape(B * K, 1, -1)
        x = x * (1.0 + gamma) + beta

        x = self.freq_backbone(x.transpose(-1, -2))
        x_avg = F.adaptive_avg_pool1d(x, self.d_freq_pool)
        x_max = F.adaptive_max_pool1d(x, self.d_freq_pool)
        x = torch.cat([x_avg, x_max], dim=1).flatten(1)
        x = self.out_proj(x)
        return x.reshape(B, K, -1)


class CSIEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        d_ff: int = 1536,
        d_clip: int = 256,
        dropout: float = 0.1,
        d_token: int = 8,
        n_freq_bins: int = 3,
        n_bw_bins: int = 3,
        token_norm_mode: str = "std",
        token_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if token_norm_mode not in {"none", "rms", "std"}:
            raise ValueError(
                "token_norm_mode must be one of: none, rms, std, "
                f"got {token_norm_mode!r}."
            )
        self.input_proj = InputProjection(d_token=d_token, d_model=d_model)
        self.token_norm_mode = token_norm_mode
        self.token_norm_eps = token_norm_eps
        self.beam_pe = BeamPositionEncoding(d_model)
        self.freq_enc = FrequencyBandEncoding(d_model, n_freq_bins=n_freq_bins, n_bw_bins=n_bw_bins)
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
        self.final_norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_clip),
        )

    def _normalize_tokens(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.token_norm_mode == "none":
            return tokens

        weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        element_count = token_mask.sum(dim=1, keepdim=True).to(dtype=tokens.dtype)
        element_count = (element_count * (tokens.shape[2] * tokens.shape[3])).clamp(min=1.0)

        if self.token_norm_mode == "rms":
            rms = torch.sqrt(
                ((tokens.square() * weights).sum(dim=(1, 2, 3), keepdim=True) / element_count[:, :, None, None])
                .clamp(min=self.token_norm_eps ** 2)
            )
            return tokens * weights / rms

        mean = (tokens * weights).sum(dim=(1, 2, 3), keepdim=True) / element_count[:, :, None, None]
        centered = (tokens - mean) * weights
        std = torch.sqrt(
            (centered.square().sum(dim=(1, 2, 3), keepdim=True) / element_count[:, :, None, None])
            .clamp(min=self.token_norm_eps ** 2)
        )
        return centered / std

    def forward(
        self,
        tokens: torch.Tensor,
        beam_positions: torch.Tensor,
        token_mask: torch.Tensor,
        freq_bin: torch.Tensor,
        bw_bin: torch.Tensor,
        subcarrier_spacing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._normalize_tokens(tokens, token_mask)
        x = self.input_proj(tokens, subcarrier_spacing)
        x = x + self.beam_pe(beam_positions) + self.freq_enc(freq_bin, bw_bin)

        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, token_mask], dim=1)
        x = self.transformer(x, src_key_padding_mask=~full_mask)
        x = self.final_norm(x)
        cls_out = x[:, 0]
        token_out = x[:, 1:]
        token_weights = token_mask.unsqueeze(-1).to(x.dtype)
        token_mean = (token_out * token_weights).sum(dim=1) / token_weights.sum(dim=1).clamp(min=1.0)
        return self.proj(cls_out + token_mean)
