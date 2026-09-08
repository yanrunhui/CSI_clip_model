from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from data.semantic_key import FIRST_POWER_DBW_BIN_LABELS
from .encoder import CSIEncoder
from .positional import normalized_subcarrier_spacing

K_FACTOR_STRONG_BIN_LABELS = ("low", "mid", "high", "very_high")
DELAY_SPREAD_BIN_LABELS = ("0_25", "25_50", "50_100", "100_200", "200_400", "400_plus")
DELAY_SPREAD_TAIL_LABELS = ("ge100", "ge200")
DELAY_SPREAD_TAIL_THRESHOLDS_NS = (100.0, 200.0)
REFLECTION_COUNT_BIN_LABELS = ("0_5", "6_7", "8_10", "11_13", "14_plus")
CSI_DELAY_CONTEXT_DIM = 64
DELAY_SPREAD_POSITION_BINS = (
    ("0_25", 0.0, 25.0),
    ("25_50", 25.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("100_200", 100.0, 200.0),
    ("200_400", 200.0, 400.0),
)
FIRST_PATH_DELAY_POSITION_BINS = (
    ("0_25", 0.0, 25.0),
    ("25_50", 25.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("100_200", 100.0, 200.0),
    ("200_400", 200.0, 400.0),
    ("400_600", 400.0, 600.0),
    ("600_800", 600.0, 800.0),
    ("800_1040", 800.0, 1040.0),
    ("1040_1280", 1040.0, 1280.0),
    ("1280_plus", 1280.0, 2560.0),
)
FIRST_PATH_DELAY_BIN_LABELS = tuple(label for label, _, _ in FIRST_PATH_DELAY_POSITION_BINS)


def fuse_first_path_delay_from_bin_position(
    bin_logits: torch.Tensor,
    bin_position: torch.Tensor,
) -> torch.Tensor:
    """Convert the argmax delay bin and in-bin position into raw delay in ns."""
    labels = tuple(label for label, _, _ in FIRST_PATH_DELAY_POSITION_BINS)
    if bin_logits.shape[-1] != len(labels):
        raise ValueError(
            "first-path-delay bin logit size mismatch: "
            f"got {bin_logits.shape[-1]}, expected {len(labels)}."
        )
    bounds = torch.tensor(
        [(lower, upper) for _, lower, upper in FIRST_PATH_DELAY_POSITION_BINS],
        device=bin_logits.device,
        dtype=bin_logits.dtype,
    )
    predicted_bins = bin_logits.argmax(dim=-1)
    lower = bounds[:, 0][predicted_bins]
    upper = bounds[:, 1][predicted_bins]
    return lower + bin_position.to(dtype=bin_logits.dtype).clamp(0.0, 1.0) * (upper - lower)


def fuse_first_path_delay_soft_from_bin_position(
    bin_logits: torch.Tensor,
    bin_position: torch.Tensor,
) -> torch.Tensor:
    """Differentiably fuse bin probabilities and in-bin position into raw delay in ns."""
    labels = tuple(label for label, _, _ in FIRST_PATH_DELAY_POSITION_BINS)
    if bin_logits.shape[-1] != len(labels):
        raise ValueError(
            "first-path-delay bin logit size mismatch: "
            f"got {bin_logits.shape[-1]}, expected {len(labels)}."
        )
    bounds = torch.tensor(
        [(lower, upper) for _, lower, upper in FIRST_PATH_DELAY_POSITION_BINS],
        device=bin_logits.device,
        dtype=bin_logits.dtype,
    )
    position = bin_position.to(dtype=bin_logits.dtype).clamp(0.0, 1.0).unsqueeze(-1)
    candidates = bounds[:, 0] + position * (bounds[:, 1] - bounds[:, 0])
    probabilities = torch.softmax(bin_logits, dim=-1)
    return (probabilities * candidates).sum(dim=-1)


class PowerFeatureEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 32,
        hidden_dim: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.stats_scale = 4.0
        self.proj = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        delay_power_map: torch.Tensor | None = None,
        delay_power_profile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del delay_power_map
        del delay_power_profile
        weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        valid_beam_count = token_mask.sum(dim=1).to(dtype=tokens.dtype).clamp(min=1.0)
        feature_count = (
            valid_beam_count * (tokens.shape[2] * tokens.shape[3])
        ).clamp(min=1.0)

        raw = tokens * weights
        raw_power = raw.square()
        raw_abs_mean = raw.abs().sum(dim=(1, 2, 3)) / feature_count
        raw_abs_max = raw.abs().amax(dim=(1, 2, 3))
        raw_power_mean = raw_power.sum(dim=(1, 2, 3)) / feature_count
        raw_power_max = raw_power.amax(dim=(1, 2, 3))
        raw_power_rms = torch.sqrt(raw_power_mean.clamp(min=self.eps ** 2))

        token_weights = token_mask.to(dtype=tokens.dtype)
        beam_power = raw_power.mean(dim=(2, 3)) * token_weights
        beam_peak_power = beam_power.amax(dim=1)

        top3_count = min(max(int(beam_power.shape[1]), 1), 3)
        top5_count = min(max(int(beam_power.shape[1]), 1), 5)
        top3_beam_power, _ = torch.topk(beam_power, k=top3_count, dim=1)
        top5_beam_power, _ = torch.topk(beam_power, k=top5_count, dim=1)
        top3_beam_power_mean = top3_beam_power.mean(dim=1)
        top5_beam_power_mean = top5_beam_power.mean(dim=1)

        freq_element_count = (valid_beam_count[:, None] * tokens.shape[2]).clamp(min=1.0)
        freq_rms = torch.sqrt(
            (raw_power.sum(dim=(1, 2)) / freq_element_count).clamp(min=self.eps ** 2)
        )
        frequency_rms_mean = freq_rms.mean(dim=1)
        frequency_rms_std = freq_rms.std(dim=1, correction=0)

        beam_peak_over_top3 = beam_peak_power / top3_beam_power_mean.clamp(min=self.eps)
        beam_peak_over_top5 = beam_peak_power / top5_beam_power_mean.clamp(min=self.eps)

        delay_power_mean = torch.zeros_like(raw_power_mean)
        delay_power_max = torch.zeros_like(raw_power_mean)
        delay_top3_power_mean = torch.zeros_like(raw_power_mean)
        delay_power_std = torch.zeros_like(raw_power_mean)
        if tokens.shape[2] % 2 == 0 and tokens.shape[2] > 0:
            half = tokens.shape[2] // 2
            complex_tokens = torch.complex(
                raw[:, :, :half, :],
                raw[:, :, half:half * 2, :],
            )
            delay_tokens = torch.fft.ifft(complex_tokens, dim=-1)
            delay_power = delay_tokens.abs().square().mean(dim=2) * token_weights.unsqueeze(-1)
            delay_element_count = (valid_beam_count[:, None] * delay_power.shape[-1]).clamp(min=1.0)
            delay_power_mean = delay_power.sum(dim=(1, 2)) / delay_element_count[:, 0]
            delay_power_max = delay_power.amax(dim=(1, 2))
            flattened_delay_power = delay_power.reshape(delay_power.shape[0], -1)
            top3_delay_count = min(max(int(flattened_delay_power.shape[1]), 1), 3)
            top3_delay_power, _ = torch.topk(flattened_delay_power, k=top3_delay_count, dim=1)
            delay_top3_power_mean = top3_delay_power.mean(dim=1)
            delay_power_std = flattened_delay_power.std(dim=1, correction=0)

        stats = torch.stack(
            [
                raw_abs_mean,
                raw_abs_max,
                raw_power_mean,
                raw_power_max,
                raw_power_rms,
                beam_peak_power,
                top3_beam_power_mean,
                top5_beam_power_mean,
                frequency_rms_mean,
                frequency_rms_std,
                beam_peak_over_top3,
                beam_peak_over_top5,
                delay_power_mean,
                delay_power_max,
                delay_top3_power_mean,
                delay_power_std,
            ],
            dim=1,
        )
        stats = torch.log1p(torch.nan_to_num(stats.clamp(min=0.0), nan=0.0, posinf=0.0, neginf=0.0))
        stats = stats / self.stats_scale
        return self.proj(stats)


class PDPLatentAuxEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        token_norm_mode: str = "std",
        hidden_dim: int = 256,
        d_token: int = 8,
    ):
        super().__init__()
        self.csi = CSIEncoder(
            d_token=d_token,
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


class CSIAngleContextEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = CSI_DELAY_CONTEXT_DIM,
        token_norm_mode: str = "std",
        hidden_dim: int = 256,
        d_token: int = 8,
    ):
        super().__init__()
        self.csi = CSIEncoder(
            d_token=d_token,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
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


class CSIFirstPathAngleContextEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = CSI_DELAY_CONTEXT_DIM,
        hidden_dim: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.beam_feature_proj = nn.Sequential(
            nn.LayerNorm(18),
            nn.Linear(18, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.selector = nn.Linear(hidden_dim, 1)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        beam_positions: torch.Tensor,
        token_mask: torch.Tensor,
        subcarrier_spacing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_weights = token_mask.to(dtype=tokens.dtype)
        raw = tokens * token_weights.unsqueeze(-1).unsqueeze(-1)
        raw_power = raw.square()

        if tokens.shape[2] % 2 == 0 and tokens.shape[2] > 0:
            half = tokens.shape[2] // 2
            complex_tokens = torch.complex(
                raw[:, :, :half, :],
                raw[:, :, half : half * 2, :],
            )
            delay_tokens = torch.fft.ifft(complex_tokens, dim=-1)
            delay_power = delay_tokens.abs().square().mean(dim=2)
        else:
            delay_power = raw_power.mean(dim=2)
        delay_power = delay_power * token_weights.unsqueeze(-1)

        power_sum = delay_power.sum(dim=-1).clamp(min=self.eps)
        delay_distribution = delay_power / power_sum.unsqueeze(-1)
        delay_idx = torch.arange(
            delay_distribution.shape[-1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        delay_den = max(delay_distribution.shape[-1] - 1, 1)
        delay_center = (delay_distribution * delay_idx).sum(dim=-1) / delay_den
        delay_spread = torch.sqrt(
            (
                delay_distribution
                * (delay_idx - delay_center.unsqueeze(-1) * delay_den).square()
            ).sum(dim=-1).clamp(min=0.0)
        ) / delay_den
        delay_entropy = -(
            delay_distribution * (delay_distribution + self.eps).log()
        ).sum(dim=-1) / math.log(max(delay_distribution.shape[-1], 2))

        peak_power, peak_idx = delay_power.max(dim=-1)
        peak_position = peak_idx.to(dtype=tokens.dtype) / delay_den
        thirds = torch.chunk(delay_distribution, 3, dim=-1)
        early = thirds[0]
        early_mass = early.sum(dim=-1)
        mid_mass = thirds[1].sum(dim=-1) if len(thirds) > 1 else torch.zeros_like(early_mass)
        late_mass = thirds[2].sum(dim=-1) if len(thirds) > 2 else torch.zeros_like(early_mass)
        top3_count = min(3, delay_distribution.shape[-1])
        top3_mass = torch.topk(delay_distribution, k=top3_count, dim=-1).values.sum(dim=-1)
        early_peak_mass = early.amax(dim=-1)
        early_peak_ratio = early_peak_mass / delay_distribution.amax(dim=-1).clamp(min=self.eps)

        if beam_positions.ndim == 3 and beam_positions.shape[-1] >= 2:
            beam_xy = beam_positions[..., :2].to(device=tokens.device, dtype=tokens.dtype)
        else:
            beam_xy = torch.zeros(
                tokens.shape[0],
                tokens.shape[1],
                2,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        beam_radius = torch.linalg.vector_norm(beam_xy, dim=-1)
        beam_angle = torch.atan2(beam_xy[..., 1], beam_xy[..., 0])
        beam_sin = torch.sin(beam_angle)
        beam_cos = torch.cos(beam_angle)

        if subcarrier_spacing is None:
            spacing = torch.zeros(tokens.shape[0], device=tokens.device, dtype=tokens.dtype)
        else:
            spacing = (
                subcarrier_spacing.to(device=tokens.device, dtype=tokens.dtype)
                / 480e3
            ).clamp(0.0, 1.0)
        spacing = spacing.unsqueeze(-1).expand(-1, tokens.shape[1])

        beam_stats = torch.stack(
            [
                torch.log1p(power_sum),
                torch.log1p(peak_power.clamp(min=0.0)),
                peak_position,
                delay_center,
                delay_spread,
                delay_entropy,
                early_mass,
                mid_mass,
                late_mass,
                top3_mass,
                early_peak_mass,
                early_peak_ratio.clamp(max=1.0),
                beam_xy[..., 0],
                beam_xy[..., 1],
                beam_radius,
                beam_sin,
                beam_cos,
                spacing,
            ],
            dim=-1,
        )
        beam_stats = torch.nan_to_num(beam_stats, nan=0.0, posinf=0.0, neginf=0.0)
        beam_features = self.beam_feature_proj(beam_stats)

        valid_mask = token_mask.bool()
        scores = self.selector(beam_features).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1).unsqueeze(-1)
        attention = torch.where(valid_mask.unsqueeze(-1), attention, torch.zeros_like(attention))
        pooled_attention = (beam_features * attention).sum(dim=1)

        valid_count = valid_mask.sum(dim=1, keepdim=True).to(dtype=tokens.dtype).clamp(min=1.0)
        pooled_mean = (beam_features * valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)).sum(dim=1) / valid_count
        pooled_max = beam_features.masked_fill(~valid_mask.unsqueeze(-1), torch.finfo(beam_features.dtype).min).amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        return self.out_proj(torch.cat([pooled_attention, pooled_mean, pooled_max], dim=-1))


class CSIDelayContextEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = CSI_DELAY_CONTEXT_DIM,
        hidden_dim: int = 96,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.beam_feature_proj = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.beam_attention = nn.Linear(hidden_dim, 1)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        subcarrier_spacing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del subcarrier_spacing
        token_weights = token_mask.to(dtype=tokens.dtype)
        raw = tokens * token_weights.unsqueeze(-1).unsqueeze(-1)
        raw_power = raw.square()

        if tokens.shape[2] % 2 == 0 and tokens.shape[2] > 0:
            half = tokens.shape[2] // 2
            complex_tokens = torch.complex(
                raw[:, :, :half, :],
                raw[:, :, half : half * 2, :],
            )
            delay_tokens = torch.fft.ifft(complex_tokens, dim=-1)
            delay_power = delay_tokens.abs().square().mean(dim=2)
        else:
            delay_power = raw_power.mean(dim=2)
        delay_power = delay_power * token_weights.unsqueeze(-1)

        power_sum = delay_power.sum(dim=-1).clamp(min=self.eps)
        delay_distribution = delay_power / power_sum.unsqueeze(-1)
        delay_idx = torch.arange(
            delay_distribution.shape[-1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        delay_den = max(delay_distribution.shape[-1] - 1, 1)
        delay_center = (delay_distribution * delay_idx).sum(dim=-1) / delay_den
        delay_spread = torch.sqrt(
            (
                delay_distribution
                * (delay_idx - delay_center.unsqueeze(-1) * delay_den).square()
            ).sum(dim=-1).clamp(min=0.0)
        ) / delay_den
        delay_entropy = -(
            delay_distribution * (delay_distribution + self.eps).log()
        ).sum(dim=-1) / math.log(max(delay_distribution.shape[-1], 2))

        thirds = torch.chunk(delay_distribution, 3, dim=-1)
        early_mass = thirds[0].sum(dim=-1)
        mid_mass = thirds[1].sum(dim=-1) if len(thirds) > 1 else torch.zeros_like(early_mass)
        late_mass = thirds[2].sum(dim=-1) if len(thirds) > 2 else torch.zeros_like(early_mass)
        top3_count = min(3, delay_distribution.shape[-1])
        top3_mass = torch.topk(delay_distribution, k=top3_count, dim=-1).values.sum(dim=-1)

        raw_power_mean = raw_power.mean(dim=(2, 3))
        raw_power_max = raw_power.amax(dim=(2, 3))
        delay_power_max = delay_power.amax(dim=-1)
        late_over_early = late_mass / early_mass.clamp(min=self.eps)

        beam_stats = torch.stack(
            [
                torch.log1p(power_sum),
                torch.log1p(raw_power_mean.clamp(min=0.0)),
                torch.log1p(raw_power_max.clamp(min=0.0)),
                torch.log1p(delay_power_max.clamp(min=0.0)),
                delay_center,
                delay_spread,
                delay_entropy,
                early_mass,
                mid_mass,
                late_mass,
                top3_mass,
                late_over_early.clamp(max=10.0) / 10.0,
            ],
            dim=-1,
        )
        beam_stats = torch.nan_to_num(beam_stats, nan=0.0, posinf=0.0, neginf=0.0)
        beam_features = self.beam_feature_proj(beam_stats)

        valid_mask = token_mask.bool()
        scores = self.beam_attention(beam_features).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1).unsqueeze(-1)
        attention = torch.where(valid_mask.unsqueeze(-1), attention, torch.zeros_like(attention))
        pooled_attention = (beam_features * attention).sum(dim=1)

        valid_count = valid_mask.sum(dim=1, keepdim=True).to(dtype=tokens.dtype).clamp(min=1.0)
        pooled_mean = (beam_features * valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)).sum(dim=1) / valid_count
        pooled_max = beam_features.masked_fill(~valid_mask.unsqueeze(-1), torch.finfo(beam_features.dtype).min).amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        return self.out_proj(torch.cat([pooled_attention, pooled_mean, pooled_max], dim=-1))


class CSIDelaySpecificEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = CSI_DELAY_CONTEXT_DIM,
        hidden_dim: int = 96,
        eps: float = 1e-6,
        continuous_spacing_encoding: bool = False,
        d_token: int = 8,
    ):
        super().__init__()
        self.eps = eps
        self.continuous_spacing_encoding = bool(continuous_spacing_encoding)
        self.input_norm = nn.LayerNorm(d_token)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(d_token, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.spacing_film_gamma = nn.Linear(1, hidden_dim)
        self.spacing_film_beta = nn.Linear(1, hidden_dim)
        self.multi_scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                )
                for kernel_size in (3, 5, 9)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.attention = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _normalize_tokens(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        element_count = token_mask.sum(dim=1, keepdim=True).to(dtype=tokens.dtype)
        element_count = (element_count * (tokens.shape[2] * tokens.shape[3])).clamp(min=1.0)
        mean = (tokens * weights).sum(dim=(1, 2, 3), keepdim=True) / element_count[:, :, None, None]
        centered = (tokens - mean) * weights
        std = torch.sqrt(
            (centered.square().sum(dim=(1, 2, 3), keepdim=True) / element_count[:, :, None, None])
            .clamp(min=self.eps ** 2)
        )
        return centered / std

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        subcarrier_spacing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, beam_count, channel_count, freq_count = tokens.shape
        tokens = self._normalize_tokens(tokens, token_mask)
        sequence = tokens.permute(0, 1, 3, 2).reshape(batch_size * beam_count, freq_count, channel_count)
        sequence = self.input_norm(sequence).transpose(1, 2)
        features = self.initial_conv(sequence)
        if subcarrier_spacing is not None:
            spacing = normalized_subcarrier_spacing(
                subcarrier_spacing.to(device=tokens.device, dtype=tokens.dtype),
                continuous=self.continuous_spacing_encoding,
            )
            gamma = self.spacing_film_gamma(spacing.unsqueeze(-1)).unsqueeze(-1)
            beta = self.spacing_film_beta(spacing.unsqueeze(-1)).unsqueeze(-1)
            gamma = gamma.repeat_interleave(beam_count, dim=0)
            beta = beta.repeat_interleave(beam_count, dim=0)
            features = features * (1.0 + gamma) + beta
        multi_scale = torch.cat([conv(features) for conv in self.multi_scale_convs], dim=1)
        features = self.fuse(multi_scale)
        features = features.reshape(batch_size, beam_count, -1, freq_count)

        valid_mask = token_mask.bool()
        flat_features = features.permute(0, 1, 3, 2).reshape(batch_size, beam_count * freq_count, -1)
        flat_mask = valid_mask.unsqueeze(-1).expand(-1, -1, freq_count).reshape(batch_size, beam_count * freq_count)
        attention_scores = self.attention(
            features.reshape(batch_size * beam_count, -1, freq_count)
        ).reshape(batch_size, beam_count * freq_count)
        attention_scores = attention_scores.masked_fill(~flat_mask, torch.finfo(attention_scores.dtype).min)
        attention = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        attention = torch.where(flat_mask.unsqueeze(-1), attention, torch.zeros_like(attention))
        pooled_attention = (flat_features * attention).sum(dim=1)

        valid_count = flat_mask.sum(dim=1, keepdim=True).to(dtype=tokens.dtype).clamp(min=1.0)
        pooled_mean = (flat_features * flat_mask.unsqueeze(-1).to(dtype=tokens.dtype)).sum(dim=1) / valid_count
        pooled_max = flat_features.masked_fill(~flat_mask.unsqueeze(-1), torch.finfo(flat_features.dtype).min).amax(dim=1)
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        return self.out_proj(torch.cat([pooled_attention, pooled_mean, pooled_max], dim=-1))


class CSIArrayInvariantDelayEncoder(nn.Module):
    """Delay encoder based on array-aggregated PDP and phase-difference features."""

    def __init__(
        self,
        out_dim: int = CSI_DELAY_CONTEXT_DIM,
        hidden_dim: int = 96,
        config_feature_dim: int = 9,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.input_norm = nn.LayerNorm(6)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(6, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.config_gamma = nn.Linear(config_feature_dim, hidden_dim)
        self.config_beta = nn.Linear(config_feature_dim, hidden_dim)
        self.multi_scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                )
                for kernel_size in (3, 5, 9)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.attention = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _complex_tokens(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.shape[2] % 2 != 0:
            raise ValueError(
                "Array-invariant delay encoding requires paired real/imaginary "
                f"channels, got d_token={tokens.shape[2]}."
            )
        weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        element_count = (
            token_mask.sum(dim=1, keepdim=True).to(dtype=tokens.dtype)
            * tokens.shape[2]
            * tokens.shape[3]
        ).clamp(min=1.0)
        mean = (tokens * weights).sum(dim=(1, 2, 3), keepdim=True)
        mean = mean / element_count[:, :, None, None]
        centered = (tokens - mean) * weights
        std = torch.sqrt(
            (
                centered.square().sum(dim=(1, 2, 3), keepdim=True)
                / element_count[:, :, None, None]
            ).clamp(min=self.eps ** 2)
        )
        normalized = centered / std
        half = normalized.shape[2] // 2
        return torch.complex(
            normalized[:, :, :half, :],
            normalized[:, :, half:, :],
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        subcarrier_spacing: torch.Tensor | None = None,
        config_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        complex_tokens = self._complex_tokens(tokens, token_mask)
        mask = token_mask.bool().unsqueeze(-1).unsqueeze(-1)
        weight = mask.to(dtype=complex_tokens.real.dtype)
        valid_count = (
            token_mask.sum(dim=1, keepdim=True).to(dtype=complex_tokens.real.dtype)
            * complex_tokens.shape[2]
        ).clamp(min=1.0)

        power = complex_tokens.abs().square() * weight
        frequency_power = power.sum(dim=(1, 2)) / valid_count
        masked_power = power.masked_fill(~mask, 0.0)
        frequency_power_max = masked_power.amax(dim=(1, 2))

        delay_response = torch.fft.ifft(
            torch.fft.ifftshift(complex_tokens, dim=-1),
            dim=-1,
        )
        delay_power = (delay_response.abs().square() * weight).sum(dim=(1, 2))
        delay_power = delay_power / valid_count
        delay_distribution = delay_power / delay_power.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=self.eps)

        adjacent = complex_tokens[..., 1:] * complex_tokens[..., :-1].conj()
        adjacent_unit = adjacent / adjacent.abs().clamp(min=self.eps)
        phase_mean = (adjacent_unit * weight).sum(dim=(1, 2)) / valid_count
        phase_real = F.pad(phase_mean.real, (1, 0), value=1.0)
        phase_imag = F.pad(phase_mean.imag, (1, 0), value=0.0)
        phase_confidence = F.pad(phase_mean.abs(), (1, 0), value=1.0)

        frequency_power = frequency_power / frequency_power.mean(
            dim=-1,
            keepdim=True,
        ).clamp(min=self.eps)
        frequency_power_max = frequency_power_max / frequency_power_max.mean(
            dim=-1,
            keepdim=True,
        ).clamp(min=self.eps)
        sequence = torch.stack(
            [
                torch.log1p(frequency_power),
                torch.log1p(frequency_power_max),
                delay_distribution,
                phase_real,
                phase_imag,
                phase_confidence,
            ],
            dim=-1,
        )
        sequence = torch.nan_to_num(
            sequence,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        features = self.initial_conv(self.input_norm(sequence).transpose(1, 2))

        if config_features is None:
            config_features = torch.zeros(
                tokens.shape[0],
                self.config_gamma.in_features,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            if subcarrier_spacing is not None:
                config_features[:, -1] = normalized_subcarrier_spacing(
                    subcarrier_spacing.to(device=tokens.device, dtype=tokens.dtype),
                    continuous=True,
                )
        else:
            config_features = config_features.to(
                device=tokens.device,
                dtype=tokens.dtype,
            )
        gamma = self.config_gamma(config_features).unsqueeze(-1)
        beta = self.config_beta(config_features).unsqueeze(-1)
        features = features * (1.0 + gamma) + beta

        multi_scale = torch.cat(
            [conv(features) for conv in self.multi_scale_convs],
            dim=1,
        )
        features = self.fuse(multi_scale)
        attention = torch.softmax(self.attention(features), dim=-1)
        pooled_attention = (features * attention).sum(dim=-1)
        pooled_mean = features.mean(dim=-1)
        pooled_max = features.amax(dim=-1)
        return self.out_proj(
            torch.cat([pooled_attention, pooled_mean, pooled_max], dim=-1)
        )


class CSIClip(nn.Module):
    def __init__(
        self,
        csi_encoder: nn.Module,
        text_encoder: nn.Module,
        num_prototypes: int | None = None,
        semantic_num_classes: int | None = None,
        embed_dim: int = 256,
        temperature: float = 0.07,
        num_physics_targets: int = 10,
        attribute_num_classes: dict[str, int] | None = None,
        use_power_branch: bool = False,
        first_path_power_mode: str = "residual",
        first_path_power_use_internal_gate: bool = True,
        use_delay_spread_head: bool = False,
        detach_delay_spread_features: bool = False,
        detach_first_path_delay_features: bool = True,
        use_delay_specific_encoder: bool = False,
        use_array_invariant_delay_encoder: bool = False,
        use_los_angle_context_encoder: bool = False,
        use_first_path_angle_context_encoder: bool = False,
        los_angle_context_token_norm_mode: str = "std",
        use_pdp_latent_aux: bool = False,
        pdp_latent_dim: int = 16,
        pdp_latent_hidden_dim: int = 256,
        pdp_latent_token_norm_mode: str = "std",
        freeze_pdp_latent_aux: bool = True,
        pdp_latent_aux_scale: float = 1.0,
        use_shared_physics_token: bool = False,
        shared_physics_token_residual_scale: float = 1.0,
        output_dict: bool = True,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.use_power_branch = use_power_branch
        if first_path_power_mode not in {"residual", "absolute"}:
            raise ValueError(
                "first_path_power_mode must be one of: residual, absolute."
            )
        self.first_path_power_mode = first_path_power_mode
        self.first_path_power_use_internal_gate = bool(first_path_power_use_internal_gate)
        self.use_delay_spread_head = use_delay_spread_head
        self.detach_delay_spread_features = detach_delay_spread_features
        self.detach_first_path_delay_features = detach_first_path_delay_features
        self.use_delay_specific_encoder = use_delay_specific_encoder
        self.use_array_invariant_delay_encoder = bool(
            use_array_invariant_delay_encoder
        )
        self.use_los_angle_context_encoder = use_los_angle_context_encoder
        self.use_first_path_angle_context_encoder = use_first_path_angle_context_encoder
        self.use_pdp_latent_aux = use_pdp_latent_aux
        self.freeze_pdp_latent_aux = freeze_pdp_latent_aux
        self.pdp_latent_aux_scale = float(pdp_latent_aux_scale)
        self.use_shared_physics_token = bool(use_shared_physics_token)
        self.shared_physics_token_residual_scale = float(shared_physics_token_residual_scale)
        self.csi = csi_encoder
        self.text = text_encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        hidden_dim = embed_dim * 2
        self.shared_physics_token = None
        if self.use_shared_physics_token:
            self.shared_physics_token = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            final_linear = self.shared_physics_token[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)
        self.power_feature_encoder = PowerFeatureEncoder()
        d_token = int(csi_encoder.input_proj.spatial_linear.in_features)
        continuous_spacing_encoding = bool(
            getattr(csi_encoder, "use_continuous_config_encoding", False)
        )
        if self.use_array_invariant_delay_encoder:
            self.csi_delay_context_encoder = CSIArrayInvariantDelayEncoder()
            self.first_path_delay_context_encoder = CSIArrayInvariantDelayEncoder()
        elif use_delay_specific_encoder:
            self.csi_delay_context_encoder = CSIDelaySpecificEncoder(
                continuous_spacing_encoding=continuous_spacing_encoding,
                d_token=d_token,
            )
            self.first_path_delay_context_encoder = CSIDelaySpecificEncoder(
                continuous_spacing_encoding=continuous_spacing_encoding,
                d_token=d_token,
            )
        else:
            self.csi_delay_context_encoder = CSIDelayContextEncoder()
            self.first_path_delay_context_encoder = CSIDelayContextEncoder()
        self.los_angle_context_encoder = (
            CSIAngleContextEncoder(
                token_norm_mode=los_angle_context_token_norm_mode,
                d_token=d_token,
            )
            if use_los_angle_context_encoder
            else None
        )
        self.first_path_angle_context_encoder = (
            CSIFirstPathAngleContextEncoder()
            if use_first_path_angle_context_encoder
            else None
        )
        self.csi_delay_context_dim = CSI_DELAY_CONTEXT_DIM
        self.delay_head_input_dim = embed_dim + self.csi_delay_context_dim
        self.pdp_latent_aux_encoder = None
        self.pdp_latent_context_proj = None
        if use_pdp_latent_aux:
            self.pdp_latent_aux_encoder = PDPLatentAuxEncoder(
                latent_dim=pdp_latent_dim,
                token_norm_mode=pdp_latent_token_norm_mode,
                hidden_dim=pdp_latent_hidden_dim,
                d_token=d_token,
            )
            self.pdp_latent_context_proj = nn.Sequential(
                nn.LayerNorm(pdp_latent_dim),
                nn.Linear(pdp_latent_dim, self.csi_delay_context_dim),
            )
            linear = self.pdp_latent_context_proj[-1]
            if isinstance(linear, nn.Linear):
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
            if freeze_pdp_latent_aux:
                for parameter in self.pdp_latent_aux_encoder.parameters():
                    parameter.requires_grad = False
        self.power_context_dim = embed_dim + 32 + 32
        self.delay_profile_stats_dim = 14
        self.delay_profile_context_dim = 32 + 32 + self.delay_profile_stats_dim
        self.interaction_count_context_dim = (
            embed_dim + self.csi_delay_context_dim + 32 + self.delay_profile_stats_dim
        )
        self.delay_spread_index = 1
        self.first_path_delay_index = 4
        self.first_path_power_index = 5
        self.first_path_angle_sin_index = 6
        self.first_path_angle_cos_index = 7
        self.delay_spread_delta_limit = 0.25
        self.delay_spread_fusion_scale = 0.02
        self.first_path_power_delta_limit = 0.5
        self.first_path_power_fusion_scale = 0.1
        self.first_path_power_bin_labels = FIRST_POWER_DBW_BIN_LABELS
        self.k_factor_strong_bin_labels = K_FACTOR_STRONG_BIN_LABELS
        self.delay_spread_bin_labels = DELAY_SPREAD_BIN_LABELS
        self.delay_spread_tail_labels = DELAY_SPREAD_TAIL_LABELS
        self.reflection_count_bin_labels = REFLECTION_COUNT_BIN_LABELS
        self.first_path_delay_bin_labels = FIRST_PATH_DELAY_BIN_LABELS
        self.physics_head = nn.Sequential(
            nn.BatchNorm1d(embed_dim, eps=1e-12, momentum=None),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_physics_targets),
        )
        self.first_path_power_head = nn.Sequential(
            nn.BatchNorm1d(self.power_context_dim, eps=1e-12, momentum=None),
            nn.Linear(self.power_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.first_path_power_gate = nn.Sequential(
            nn.BatchNorm1d(self.power_context_dim, eps=1e-12, momentum=None),
            nn.Linear(self.power_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.delay_spread_head = nn.Sequential(
            nn.BatchNorm1d(self.delay_profile_context_dim, eps=1e-12, momentum=None),
            nn.Linear(self.delay_profile_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.delay_spread_gate = nn.Sequential(
            nn.BatchNorm1d(self.delay_profile_context_dim, eps=1e-12, momentum=None),
            nn.Linear(self.delay_profile_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.delay_spread_direct_head = nn.Sequential(
            nn.BatchNorm1d(self.delay_profile_stats_dim, eps=1e-12, momentum=None),
            nn.Linear(self.delay_profile_stats_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.csi_delay_spread_head = nn.Sequential(
            nn.BatchNorm1d(embed_dim, eps=1e-12, momentum=None),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.delay_spread_bin_classifier = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.delay_spread_bin_labels)),
        )
        self.delay_spread_bin_position_head = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.delay_spread_context_head = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.first_path_delay_context_head = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.first_path_delay_bin_classifier = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.first_path_delay_bin_labels)),
        )
        self.first_path_delay_bin_position_head = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.los_delay_context_head = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.los_angle_head = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.first_path_angle_head = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.first_path_angle_fusion_head = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim * 2),
            nn.Linear(self.delay_head_input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.first_path_angle_selector_fusion_head = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim * 2 + self.csi_delay_context_dim),
            nn.Linear(self.delay_head_input_dim * 2 + self.csi_delay_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.delay_spread_tail_classifier = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.delay_spread_tail_labels)),
        )
        self.reflection_count_classifier = nn.Sequential(
            nn.LayerNorm(self.interaction_count_context_dim),
            nn.Linear(self.interaction_count_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.reflection_count_bin_labels)),
        )
        self.reflection_count_regression_head = nn.Sequential(
            nn.LayerNorm(self.interaction_count_context_dim),
            nn.Linear(self.interaction_count_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reflection_path_count_regression_head = nn.Sequential(
            nn.LayerNorm(self.interaction_count_context_dim),
            nn.Linear(self.interaction_count_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.first_path_power_bin_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.first_path_power_bin_labels)),
        )
        self.first_path_power_bin_position_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.k_factor_strong_bin_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.k_factor_strong_bin_labels)),
        )
        self.k_factor_strong_position_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.k_factor_strong_power_bin_classifier = nn.Sequential(
            nn.LayerNorm(self.power_context_dim),
            nn.Linear(self.power_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.k_factor_strong_bin_labels)),
        )
        self.k_factor_strong_power_position_head = nn.Sequential(
            nn.LayerNorm(self.power_context_dim),
            nn.Linear(self.power_context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_first_path_power_bin_classifier()
        self._init_delay_spread_head_identity()
        self.semantic_classifier = None
        if semantic_num_classes is not None:
            self.semantic_classifier = nn.Sequential(
                nn.BatchNorm1d(embed_dim, eps=1e-12, momentum=None),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, semantic_num_classes),
            )
        self.attribute_classifiers = nn.ModuleDict(
            {
                field: nn.Sequential(
                    nn.BatchNorm1d(embed_dim, eps=1e-12, momentum=None),
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, num_classes),
                )
                for field, num_classes in (attribute_num_classes or {}).items()
            }
        )
        self.prototypes = None
        if num_prototypes is not None:
            self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim) * 0.02)

    def load_pdp_latent_aux_checkpoint(
        self,
        path: str,
        map_location: torch.device | str | None = None,
    ) -> None:
        if self.pdp_latent_aux_encoder is None:
            raise RuntimeError("This CSIClip instance was created without PDP latent aux.")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        state_dict = checkpoint.get("model_state", checkpoint)
        self.pdp_latent_aux_encoder.load_state_dict(state_dict, strict=True)
        if self.freeze_pdp_latent_aux:
            self.pdp_latent_aux_encoder.eval()
            for parameter in self.pdp_latent_aux_encoder.parameters():
                parameter.requires_grad = False

    def _init_first_path_power_bin_classifier(self) -> None:
        for module in (
            *self.first_path_power_bin_classifier.modules(),
            *self.first_path_power_bin_position_head.modules(),
            *self.k_factor_strong_bin_classifier.modules(),
            *self.k_factor_strong_position_head.modules(),
            *self.k_factor_strong_power_bin_classifier.modules(),
            *self.k_factor_strong_power_position_head.modules(),
            *self.delay_spread_bin_classifier.modules(),
            *self.delay_spread_bin_position_head.modules(),
            *self.delay_spread_context_head.modules(),
            *self.first_path_delay_context_head.modules(),
            *self.first_path_delay_bin_classifier.modules(),
            *self.first_path_delay_bin_position_head.modules(),
            *self.los_delay_context_head.modules(),
            *self.los_angle_head.modules(),
            *self.first_path_angle_head.modules(),
            *self.first_path_angle_fusion_head.modules(),
            *self.first_path_angle_selector_fusion_head.modules(),
            *self.delay_spread_tail_classifier.modules(),
            *self.reflection_count_classifier.modules(),
            *self.reflection_count_regression_head.modules(),
            *self.reflection_path_count_regression_head.modules(),
        ):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=1e-3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_delay_spread_head_identity(self) -> None:
        delta_linear = self.delay_spread_head[-1]
        gate_linear = self.delay_spread_gate[-1]
        if isinstance(delta_linear, nn.Linear):
            nn.init.zeros_(delta_linear.weight)
            if delta_linear.bias is not None:
                nn.init.zeros_(delta_linear.bias)
        if isinstance(gate_linear, nn.Linear):
            nn.init.zeros_(gate_linear.weight)
            if gate_linear.bias is not None:
                nn.init.constant_(gate_linear.bias, -4.0)

    @torch.no_grad()
    def initialize_prototypes(
        self,
        prototype_features: torch.Tensor,
        normalize: bool = True,
    ) -> None:
        if self.prototypes is None:
            raise RuntimeError("This CSIClip instance was created without learnable prototypes.")
        if prototype_features.shape != self.prototypes.shape:
            raise ValueError(
                f"Prototype init shape mismatch: expected {tuple(self.prototypes.shape)}, "
                f"got {tuple(prototype_features.shape)}."
            )
        features = prototype_features.to(device=self.prototypes.device, dtype=self.prototypes.dtype)
        if normalize:
            features = F.normalize(features, dim=-1)
        self.prototypes.copy_(features)

    def encode_csi(
        self,
        tokens: torch.Tensor,
        beam_positions: torch.Tensor,
        token_mask: torch.Tensor,
        freq_bin: torch.Tensor,
        bw_bin: torch.Tensor,
        subcarrier_spacing: torch.Tensor,
        normalize: bool = False,
        config_features: torch.Tensor | None = None,
        antenna_coordinates: torch.Tensor | None = None,
        antenna_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.csi(
            tokens,
            beam_positions,
            token_mask,
            freq_bin,
            bw_bin,
            subcarrier_spacing,
            config_features=config_features,
            antenna_coordinates=antenna_coordinates,
            antenna_mask=antenna_mask,
        )
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        features = self.text(token_ids, token_mask)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_prototypes(self, normalize: bool = False) -> torch.Tensor:
        if self.prototypes is None:
            raise RuntimeError("This CSIClip instance was created without learnable prototypes.")
        return F.normalize(self.prototypes, dim=-1) if normalize else self.prototypes

    def encode_shared_physics_token(self, csi_features: torch.Tensor) -> torch.Tensor:
        if self.shared_physics_token is None:
            return csi_features
        return csi_features + self.shared_physics_token_residual_scale * self.shared_physics_token(csi_features)

    def encode_power_context(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        delay_power_map: torch.Tensor | None = None,
        delay_power_profile: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        raw_power_context = self.power_feature_encoder(
            tokens,
            token_mask,
            delay_power_map=delay_power_map,
            delay_power_profile=delay_power_profile,
        )
        delay_map_context = torch.zeros(
            tokens.shape[0],
            32,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        delay_profile_stats = self._delay_profile_statistics(
            delay_power_map=delay_power_map,
            delay_power_profile=delay_power_profile,
            batch_size=tokens.shape[0],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return {
            "raw_power_context": raw_power_context,
            "delay_map_context": delay_map_context,
            "delay_profile_stats": delay_profile_stats,
        }

    def encode_csi_delay_context(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        subcarrier_spacing: torch.Tensor | None = None,
        config_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {"subcarrier_spacing": subcarrier_spacing}
        if self.use_array_invariant_delay_encoder:
            kwargs["config_features"] = config_features
        return self.csi_delay_context_encoder(tokens, token_mask, **kwargs)

    def encode_first_path_delay_context(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        beam_positions: torch.Tensor | None = None,
        freq_bin: torch.Tensor | None = None,
        bw_bin: torch.Tensor | None = None,
        subcarrier_spacing: torch.Tensor | None = None,
        config_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {"subcarrier_spacing": subcarrier_spacing}
        if self.use_array_invariant_delay_encoder:
            kwargs["config_features"] = config_features
        context = self.first_path_delay_context_encoder(tokens, token_mask, **kwargs)
        if not self.use_pdp_latent_aux:
            return context
        if (
            self.pdp_latent_aux_encoder is None
            or self.pdp_latent_context_proj is None
        ):
            raise RuntimeError("PDP latent aux is enabled but not initialized.")
        if beam_positions is None or freq_bin is None or bw_bin is None or subcarrier_spacing is None:
            raise ValueError(
                "PDP latent aux requires beam_positions, freq_bin, bw_bin, and subcarrier_spacing."
            )
        if self.freeze_pdp_latent_aux:
            self.pdp_latent_aux_encoder.eval()
            with torch.no_grad():
                latent = self.pdp_latent_aux_encoder(
                    tokens,
                    beam_positions,
                    token_mask,
                    freq_bin,
                    bw_bin,
                    subcarrier_spacing,
                )
        else:
            latent = self.pdp_latent_aux_encoder(
                tokens,
                beam_positions,
                token_mask,
                freq_bin,
                bw_bin,
                subcarrier_spacing,
            )
        latent_context = self.pdp_latent_context_proj(latent).to(dtype=context.dtype)
        return context + self.pdp_latent_aux_scale * latent_context

    def encode_los_angle_context(
        self,
        tokens: torch.Tensor,
        beam_positions: torch.Tensor,
        token_mask: torch.Tensor,
        freq_bin: torch.Tensor,
        bw_bin: torch.Tensor,
        subcarrier_spacing: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.los_angle_context_encoder is None:
            return None
        return self.los_angle_context_encoder(
            tokens,
            beam_positions,
            token_mask,
            freq_bin,
            bw_bin,
            subcarrier_spacing,
        )

    def encode_first_path_angle_context(
        self,
        tokens: torch.Tensor,
        beam_positions: torch.Tensor,
        token_mask: torch.Tensor,
        subcarrier_spacing: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.first_path_angle_context_encoder is None:
            return None
        return self.first_path_angle_context_encoder(
            tokens,
            beam_positions,
            token_mask,
            subcarrier_spacing=subcarrier_spacing,
        )

    def _delay_profile_statistics(
        self,
        *,
        delay_power_map: torch.Tensor | None,
        delay_power_profile: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if delay_power_profile is None:
            delay_power_profile = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        profile = delay_power_profile.to(device=device, dtype=dtype).flatten(start_dim=1)
        profile_norm = profile.clamp(min=0.0)
        profile_norm = profile_norm / profile_norm.sum(dim=1, keepdim=True).clamp(min=1e-12)
        profile_idx = torch.arange(profile_norm.shape[1], device=device, dtype=dtype)
        profile_den = max(profile_norm.shape[1] - 1, 1)
        profile_center = (profile_norm * profile_idx).sum(dim=1) / profile_den
        profile_spread = torch.sqrt(
            (profile_norm * (profile_idx - profile_center[:, None] * profile_den).square()).sum(dim=1)
        ) / profile_den
        profile_entropy = -(
            profile_norm * (profile_norm + 1e-12).log()
        ).sum(dim=1) / math.log(max(profile_norm.shape[1], 2))
        thirds = torch.chunk(profile_norm, 3, dim=1)
        profile_early = thirds[0].sum(dim=1)
        profile_mid = thirds[1].sum(dim=1) if len(thirds) > 1 else torch.zeros_like(profile_early)
        profile_late = thirds[2].sum(dim=1) if len(thirds) > 2 else torch.zeros_like(profile_early)
        profile_top3 = torch.topk(
            profile_norm,
            k=min(3, profile_norm.shape[1]),
            dim=1,
        ).values.sum(dim=1)

        if delay_power_map is None:
            delay_power_map = torch.zeros(batch_size, 32, 32, device=device, dtype=dtype)
        delay_map = delay_power_map.to(device=device, dtype=dtype).flatten(start_dim=1)
        map_norm = delay_map.clamp(min=0.0)
        map_norm = map_norm / map_norm.sum(dim=1, keepdim=True).clamp(min=1e-12)
        map_shape = delay_power_map.shape[-2:]
        map_2d = map_norm.reshape(batch_size, map_shape[0], map_shape[1])
        delay_marginal = map_2d.sum(dim=2)
        power_marginal = map_2d.sum(dim=1)
        delay_idx = torch.arange(map_shape[0], device=device, dtype=dtype)
        power_idx = torch.arange(map_shape[1], device=device, dtype=dtype)
        delay_den = max(map_shape[0] - 1, 1)
        power_den = max(map_shape[1] - 1, 1)
        map_delay_center = (delay_marginal * delay_idx).sum(dim=1) / delay_den
        map_power_center = (power_marginal * power_idx).sum(dim=1) / power_den
        map_entropy = -(map_norm * (map_norm + 1e-12).log()).sum(dim=1) / math.log(
            max(map_norm.shape[1], 2)
        )

        stats = torch.stack(
            [
                profile_norm.max(dim=1).values,
                profile_top3,
                profile_entropy,
                profile_center,
                profile_spread,
                profile_early,
                profile_mid,
                profile_late,
                map_norm.max(dim=1).values,
                map_entropy,
                map_delay_center,
                map_power_center,
                delay_marginal.max(dim=1).values,
                power_marginal.max(dim=1).values,
            ],
            dim=1,
        )
        return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

    def predict_physics(
        self,
        csi_features: torch.Tensor,
        power_context: dict[str, torch.Tensor] | None = None,
        delay_context: torch.Tensor | None = None,
        first_path_delay_context: torch.Tensor | None = None,
        los_angle_context: torch.Tensor | None = None,
        first_path_angle_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_physics_components(
            csi_features,
            power_context=power_context,
            delay_context=delay_context,
            first_path_delay_context=first_path_delay_context,
            los_angle_context=los_angle_context,
            first_path_angle_context=first_path_angle_context,
        )["final"]

    def predict_physics_components(
        self,
        csi_features: torch.Tensor,
        power_context: dict[str, torch.Tensor] | None = None,
        delay_context: torch.Tensor | None = None,
        first_path_delay_context: torch.Tensor | None = None,
        los_angle_context: torch.Tensor | None = None,
        first_path_angle_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        physics_features = self.encode_shared_physics_token(csi_features)
        base = self.physics_head(physics_features)
        if self.use_delay_spread_head:
            base = base.clone()
            base[:, self.delay_spread_index] = 0.0
        delay_csi_features = (
            physics_features.detach()
            if self.detach_delay_spread_features
            else physics_features
        )
        csi_delay_spread = self.csi_delay_spread_head(delay_csi_features).squeeze(-1)
        if delay_context is None:
            delay_context = torch.zeros(
                csi_features.shape[0],
                self.csi_delay_context_dim,
                device=csi_features.device,
                dtype=csi_features.dtype,
            )
        if first_path_delay_context is None:
            first_path_delay_context = torch.zeros(
                csi_features.shape[0],
                self.csi_delay_context_dim,
                device=csi_features.device,
                dtype=csi_features.dtype,
            )
        delay_head_input = torch.cat([delay_csi_features, delay_context], dim=-1)
        first_path_delay_head_input = torch.cat(
            [delay_csi_features, first_path_delay_context],
            dim=-1,
        )
        delay_spread_bin_logits = self.delay_spread_bin_classifier(delay_head_input)
        delay_spread_bin_position = torch.sigmoid(
            self.delay_spread_bin_position_head(delay_head_input).squeeze(-1)
        )
        delay_spread_context = self.delay_spread_context_head(delay_head_input).squeeze(-1)
        first_path_delay_input = (
            first_path_delay_head_input.detach()
            if self.detach_first_path_delay_features
            else first_path_delay_head_input
        )
        los_delay_input = first_path_delay_input.detach()
        first_path_delay_context = self.first_path_delay_context_head(first_path_delay_input).squeeze(-1)
        first_path_delay_bin_logits = self.first_path_delay_bin_classifier(first_path_delay_input)
        first_path_delay_bin_position = torch.sigmoid(
            self.first_path_delay_bin_position_head(first_path_delay_input).squeeze(-1)
        )
        first_path_delay_bin_fused_raw = fuse_first_path_delay_from_bin_position(
            first_path_delay_bin_logits,
            first_path_delay_bin_position,
        )
        first_path_delay_bin_soft_fused_raw = fuse_first_path_delay_soft_from_bin_position(
            first_path_delay_bin_logits,
            first_path_delay_bin_position,
        )
        los_delay_context = self.los_delay_context_head(los_delay_input).squeeze(-1)
        los_angle_input = first_path_delay_input
        if los_angle_context is not None:
            los_angle_input = torch.cat([delay_csi_features, los_angle_context], dim=-1)
        los_angle_sincos = self.los_angle_head(los_angle_input)
        first_path_angle_input = torch.cat([first_path_delay_input, los_angle_input], dim=-1)
        if first_path_angle_context is None:
            first_path_angle_sincos = self.first_path_angle_fusion_head(first_path_angle_input)
        else:
            first_path_angle_sincos = self.first_path_angle_selector_fusion_head(
                torch.cat([first_path_angle_input, first_path_angle_context], dim=-1)
            )
        delay_spread_tail_logits = self.delay_spread_tail_classifier(delay_head_input)
        if power_context is None:
            raw_power_context = torch.zeros(
                csi_features.shape[0],
                32,
                device=csi_features.device,
                dtype=csi_features.dtype,
            )
            delay_profile_stats = torch.zeros(
                csi_features.shape[0],
                self.delay_profile_stats_dim,
                device=csi_features.device,
                dtype=csi_features.dtype,
            )
        else:
            raw_power_context = power_context["raw_power_context"]
            delay_profile_stats = power_context["delay_profile_stats"]
        interaction_count_input = torch.cat(
            [
                physics_features,
                delay_context,
                raw_power_context,
                delay_profile_stats,
            ],
            dim=-1,
        )
        reflection_count_logits = self.reflection_count_classifier(interaction_count_input)
        reflection_count_prediction = self.reflection_count_regression_head(
            interaction_count_input
        ).squeeze(-1)
        reflection_path_count_prediction = self.reflection_path_count_regression_head(
            interaction_count_input
        ).squeeze(-1)
        first_path_power_bin_logits = self.first_path_power_bin_classifier(
            physics_features
        )
        first_path_power_bin_position = torch.sigmoid(
            self.first_path_power_bin_position_head(physics_features).squeeze(-1)
        )
        k_factor_strong_bin_logits = self.k_factor_strong_bin_classifier(physics_features)
        k_factor_strong_position = torch.sigmoid(
            self.k_factor_strong_position_head(physics_features).squeeze(-1)
        )
        if not self.use_power_branch or power_context is None:
            zeros = torch.zeros(
                base.shape[0],
                device=base.device,
                dtype=base.dtype,
            )
            final = base.clone()
            if self.use_delay_spread_head:
                final[:, self.delay_spread_index] = delay_spread_context
            final[:, self.first_path_delay_index] = (
                first_path_delay_bin_soft_fused_raw / 3000.0
            )
            final[:, self.first_path_angle_sin_index] = first_path_angle_sincos[:, 0]
            final[:, self.first_path_angle_cos_index] = first_path_angle_sincos[:, 1]
            return {
                "base": base,
                "enhanced_first_path_power": base[:, self.first_path_power_index],
                "enhanced_delay_spread": base[:, self.delay_spread_index],
                "csi_delay_spread": csi_delay_spread,
                "profile_delay_spread": base[:, self.delay_spread_index],
                "profile_direct_delay_spread": base[:, self.delay_spread_index],
                "delay_spread_context": delay_spread_context,
                "first_path_delay_context": first_path_delay_context,
                "first_path_delay_bin_logits": first_path_delay_bin_logits,
                "first_path_delay_bin_position": first_path_delay_bin_position,
                "first_path_delay_bin_fused_raw": first_path_delay_bin_fused_raw,
                "first_path_delay_bin_soft_fused_raw": first_path_delay_bin_soft_fused_raw,
                "los_delay_context": los_delay_context,
                "los_angle_sincos": los_angle_sincos,
                "first_path_angle_sincos": first_path_angle_sincos,
                "enhanced_delay_spread_gate": zeros,
                "enhanced_delay_spread_delta": zeros,
                "enhanced_gate": zeros,
                "enhanced_delta": zeros,
                "delay_spread_bin_logits": delay_spread_bin_logits,
                "delay_spread_bin_position": delay_spread_bin_position,
                "delay_spread_tail_logits": delay_spread_tail_logits,
                "reflection_count_logits": reflection_count_logits,
                "reflection_count_prediction": reflection_count_prediction,
                "reflection_path_count_prediction": reflection_path_count_prediction,
                "first_path_power_bin_logits": first_path_power_bin_logits,
                "first_path_power_bin_position": first_path_power_bin_position,
                "k_factor_strong_bin_logits": k_factor_strong_bin_logits,
                "k_factor_strong_position": k_factor_strong_position,
                "final": final,
            }
        enhanced_input = torch.cat(
            [
                physics_features,
                power_context["delay_map_context"],
                power_context["raw_power_context"],
            ],
            dim=-1,
        )
        delay_profile_input = torch.cat(
            [
                power_context["delay_map_context"],
                power_context["raw_power_context"],
                power_context["delay_profile_stats"],
            ],
            dim=-1,
        )
        profile_direct_delay_spread = self.delay_spread_direct_head(
            power_context["delay_profile_stats"]
        ).squeeze(-1)
        k_factor_strong_bin_logits = self.k_factor_strong_power_bin_classifier(enhanced_input)
        k_factor_strong_position = torch.sigmoid(
            self.k_factor_strong_power_position_head(enhanced_input).squeeze(-1)
        )
        enhanced_delta = self.first_path_power_head(enhanced_input).squeeze(-1)
        enhanced_delta = self.first_path_power_delta_limit * torch.tanh(
            enhanced_delta / max(self.first_path_power_delta_limit, 1e-6)
        )
        enhanced_gate = torch.sigmoid(
            self.first_path_power_gate(enhanced_input).squeeze(-1)
        )
        delay_spread_delta = self.delay_spread_head(delay_profile_input).squeeze(-1)
        delay_spread_delta = self.delay_spread_delta_limit * torch.tanh(
            delay_spread_delta / max(self.delay_spread_delta_limit, 1e-6)
        )
        delay_spread_gate = torch.sigmoid(
            self.delay_spread_gate(delay_profile_input).squeeze(-1)
        )
        base_delay_spread = base[:, self.delay_spread_index]
        profile_delay_spread = base_delay_spread + delay_spread_delta
        enhanced_delay_spread = (
            base_delay_spread
            + self.delay_spread_fusion_scale * delay_spread_gate * delay_spread_delta
        )
        base_first_path_power = base[:, self.first_path_power_index]
        gated_delta = (
            enhanced_gate * enhanced_delta
            if self.first_path_power_use_internal_gate
            else enhanced_delta
        )
        if self.first_path_power_mode == "absolute":
            enhanced_first_path_power = enhanced_delta
        else:
            enhanced_first_path_power = base_first_path_power + gated_delta
        final = base.clone()
        if self.use_delay_spread_head:
            final[:, self.delay_spread_index] = delay_spread_context
        # Keep enhanced power available for diagnostics, but do not route it into final.
        final[:, self.first_path_power_index] = base_first_path_power
        final[:, self.first_path_delay_index] = (
            first_path_delay_bin_soft_fused_raw / 3000.0
        )
        final[:, self.first_path_angle_sin_index] = first_path_angle_sincos[:, 0]
        final[:, self.first_path_angle_cos_index] = first_path_angle_sincos[:, 1]
        return {
            "base": base,
            "enhanced_first_path_power": enhanced_first_path_power,
            "enhanced_delay_spread": enhanced_delay_spread,
            "csi_delay_spread": csi_delay_spread,
            "profile_delay_spread": profile_delay_spread,
            "profile_direct_delay_spread": profile_direct_delay_spread,
            "delay_spread_context": delay_spread_context,
            "first_path_delay_context": first_path_delay_context,
            "first_path_delay_bin_logits": first_path_delay_bin_logits,
            "first_path_delay_bin_position": first_path_delay_bin_position,
            "first_path_delay_bin_fused_raw": first_path_delay_bin_fused_raw,
            "first_path_delay_bin_soft_fused_raw": first_path_delay_bin_soft_fused_raw,
            "los_delay_context": los_delay_context,
            "los_angle_sincos": los_angle_sincos,
            "first_path_angle_sincos": first_path_angle_sincos,
            "enhanced_delay_spread_gate": delay_spread_gate,
            "enhanced_delay_spread_delta": delay_spread_delta,
            "enhanced_gate": enhanced_gate,
            "enhanced_delta": enhanced_delta,
            "delay_spread_bin_logits": delay_spread_bin_logits,
            "delay_spread_bin_position": delay_spread_bin_position,
            "delay_spread_tail_logits": delay_spread_tail_logits,
            "reflection_count_logits": reflection_count_logits,
            "reflection_count_prediction": reflection_count_prediction,
            "reflection_path_count_prediction": reflection_path_count_prediction,
            "first_path_power_bin_logits": first_path_power_bin_logits,
            "first_path_power_bin_position": first_path_power_bin_position,
            "k_factor_strong_bin_logits": k_factor_strong_bin_logits,
            "k_factor_strong_position": k_factor_strong_position,
            "final": final,
        }

    def predict_semantic(self, csi_features: torch.Tensor) -> torch.Tensor:
        if self.semantic_classifier is None:
            raise RuntimeError("This CSIClip instance was created without a semantic classifier.")
        return self.semantic_classifier(csi_features)

    def predict_attributes(self, csi_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.attribute_classifiers:
            raise RuntimeError("This CSIClip instance was created without attribute classifiers.")
        return {
            field: classifier(csi_features)
            for field, classifier in self.attribute_classifiers.items()
        }

    def forward(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor] | list[str]]):
        csi_features = self.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=True,
            config_features=batch.get("config_features"),
            antenna_coordinates=batch.get("antenna_coordinates"),
            antenna_mask=batch.get("antenna_mask"),
        )
        text_features = self.encode_text(
            batch["t_prop_ids"],
            batch["t_prop_mask"],
            normalize=True,
        )
        prototype_features = self.encode_prototypes(normalize=True) if self.prototypes is not None else None
        if self.output_dict:
            out = {
                "csi_features": csi_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if prototype_features is not None:
                out["prototype_features"] = prototype_features
            return out
        if prototype_features is not None:
            return csi_features, text_features, prototype_features, self.logit_scale.exp()
        return csi_features, text_features, self.logit_scale.exp()


CrossConfigCSI = CSIClip
