from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from data.semantic_key import FIRST_POWER_DBW_BIN_LABELS

K_FACTOR_STRONG_BIN_LABELS = ("low", "mid", "high", "very_high")
DELAY_SPREAD_BIN_LABELS = ("0_25", "25_50", "50_100", "100_200", "200_400", "400_plus")
DELAY_SPREAD_TAIL_LABELS = ("ge100", "ge200")
DELAY_SPREAD_TAIL_THRESHOLDS_NS = (100.0, 200.0)
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
    ("400_800", 400.0, 800.0),
    ("800_1600", 800.0, 1600.0),
)
FIRST_PATH_DELAY_BIN_LABELS = (
    *(label for label, _, _ in FIRST_PATH_DELAY_POSITION_BINS),
    "1600_plus",
)


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
    ):
        super().__init__()
        self.eps = eps
        self.input_norm = nn.LayerNorm(8)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(8, hidden_dim, kernel_size=3, padding=1),
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
            spacing = (subcarrier_spacing.to(device=tokens.device, dtype=tokens.dtype) / 480e3).clamp(0.0, 1.0)
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
        use_delay_spread_head: bool = False,
        detach_delay_spread_features: bool = False,
        detach_first_path_delay_features: bool = True,
        use_delay_specific_encoder: bool = False,
        output_dict: bool = True,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.use_power_branch = use_power_branch
        self.use_delay_spread_head = use_delay_spread_head
        self.detach_delay_spread_features = detach_delay_spread_features
        self.detach_first_path_delay_features = detach_first_path_delay_features
        self.use_delay_specific_encoder = use_delay_specific_encoder
        self.csi = csi_encoder
        self.text = text_encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        hidden_dim = embed_dim * 2
        self.power_feature_encoder = PowerFeatureEncoder()
        self.csi_delay_context_encoder = (
            CSIDelaySpecificEncoder()
            if use_delay_specific_encoder
            else CSIDelayContextEncoder()
        )
        self.first_path_delay_context_encoder = (
            CSIDelaySpecificEncoder()
            if use_delay_specific_encoder
            else CSIDelayContextEncoder()
        )
        self.csi_delay_context_dim = CSI_DELAY_CONTEXT_DIM
        self.delay_head_input_dim = embed_dim + self.csi_delay_context_dim
        self.power_context_dim = embed_dim + 32 + 32
        self.delay_profile_stats_dim = 14
        self.delay_profile_context_dim = 32 + 32 + self.delay_profile_stats_dim
        self.delay_spread_index = 1
        self.first_path_power_index = 5
        self.delay_spread_delta_limit = 0.25
        self.delay_spread_fusion_scale = 0.02
        self.first_path_power_delta_limit = 0.5
        self.first_path_power_fusion_scale = 0.1
        self.first_path_power_bin_labels = FIRST_POWER_DBW_BIN_LABELS
        self.k_factor_strong_bin_labels = K_FACTOR_STRONG_BIN_LABELS
        self.delay_spread_bin_labels = DELAY_SPREAD_BIN_LABELS
        self.delay_spread_tail_labels = DELAY_SPREAD_TAIL_LABELS
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
        self.delay_spread_tail_classifier = nn.Sequential(
            nn.LayerNorm(self.delay_head_input_dim),
            nn.Linear(self.delay_head_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.delay_spread_tail_labels)),
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
            *self.delay_spread_tail_classifier.modules(),
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
    ) -> torch.Tensor:
        features = self.csi(
            tokens,
            beam_positions,
            token_mask,
            freq_bin,
            bw_bin,
            subcarrier_spacing,
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
    ) -> torch.Tensor:
        return self.csi_delay_context_encoder(
            tokens,
            token_mask,
            subcarrier_spacing=subcarrier_spacing,
        )

    def encode_first_path_delay_context(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        subcarrier_spacing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.first_path_delay_context_encoder(
            tokens,
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
    ) -> torch.Tensor:
        return self.predict_physics_components(
            csi_features,
            power_context=power_context,
            delay_context=delay_context,
            first_path_delay_context=first_path_delay_context,
        )["final"]

    def predict_physics_components(
        self,
        csi_features: torch.Tensor,
        power_context: dict[str, torch.Tensor] | None = None,
        delay_context: torch.Tensor | None = None,
        first_path_delay_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base = self.physics_head(csi_features)
        if self.use_delay_spread_head:
            base = base.clone()
            base[:, self.delay_spread_index] = 0.0
        delay_csi_features = (
            csi_features.detach()
            if self.detach_delay_spread_features
            else csi_features
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
        los_delay_context = self.los_delay_context_head(los_delay_input).squeeze(-1)
        delay_spread_tail_logits = self.delay_spread_tail_classifier(delay_head_input)
        first_path_power_bin_logits = self.first_path_power_bin_classifier(
            csi_features
        )
        first_path_power_bin_position = torch.sigmoid(
            self.first_path_power_bin_position_head(csi_features).squeeze(-1)
        )
        k_factor_strong_bin_logits = self.k_factor_strong_bin_classifier(csi_features)
        k_factor_strong_position = torch.sigmoid(
            self.k_factor_strong_position_head(csi_features).squeeze(-1)
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
                "los_delay_context": los_delay_context,
                "enhanced_delay_spread_gate": zeros,
                "enhanced_delay_spread_delta": zeros,
                "enhanced_gate": zeros,
                "enhanced_delta": zeros,
                "delay_spread_bin_logits": delay_spread_bin_logits,
                "delay_spread_bin_position": delay_spread_bin_position,
                "delay_spread_tail_logits": delay_spread_tail_logits,
                "first_path_power_bin_logits": first_path_power_bin_logits,
                "first_path_power_bin_position": first_path_power_bin_position,
                "k_factor_strong_bin_logits": k_factor_strong_bin_logits,
                "k_factor_strong_position": k_factor_strong_position,
                "final": final,
            }
        enhanced_input = torch.cat(
            [
                csi_features,
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
        gated_delta = enhanced_gate * enhanced_delta
        enhanced_first_path_power = base_first_path_power + gated_delta
        final = base.clone()
        if self.use_delay_spread_head:
            final[:, self.delay_spread_index] = delay_spread_context
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
            "los_delay_context": los_delay_context,
            "enhanced_delay_spread_gate": delay_spread_gate,
            "enhanced_delay_spread_delta": delay_spread_delta,
            "enhanced_gate": enhanced_gate,
            "enhanced_delta": enhanced_delta,
            "delay_spread_bin_logits": delay_spread_bin_logits,
            "delay_spread_bin_position": delay_spread_bin_position,
            "delay_spread_tail_logits": delay_spread_tail_logits,
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
