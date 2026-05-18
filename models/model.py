from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PowerFeatureEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 32,
        hidden_dim: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.proj = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
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
            ],
            dim=1,
        )
        return self.proj(stats)


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
        output_dict: bool = True,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.use_power_branch = use_power_branch
        self.csi = csi_encoder
        self.text = text_encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        hidden_dim = embed_dim * 2
        self.power_feature_encoder = PowerFeatureEncoder()
        self.first_path_power_index = 5
        self.first_path_power_residual_scale = nn.Parameter(torch.tensor(0.0))
        self.physics_head = nn.Sequential(
            nn.BatchNorm1d(embed_dim, eps=1e-12, momentum=None),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_physics_targets),
        )
        self.first_path_power_head = nn.Sequential(
            nn.BatchNorm1d(embed_dim + 32, eps=1e-12, momentum=None),
            nn.Linear(embed_dim + 32, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
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
    ) -> torch.Tensor:
        return self.power_feature_encoder(tokens, token_mask)

    def predict_physics(
        self,
        csi_features: torch.Tensor,
        power_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_physics_components(
            csi_features,
            power_context=power_context,
        )["final"]

    def predict_physics_components(
        self,
        csi_features: torch.Tensor,
        power_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base = self.physics_head(csi_features)
        if not self.use_power_branch or power_context is None:
            zeros = torch.zeros(
                base.shape[0],
                device=base.device,
                dtype=base.dtype,
            )
            return {
                "base": base,
                "direct_first_path_power": base[:, self.first_path_power_index],
                "residual_scaled": zeros,
                "final": base,
            }
        power_input = torch.cat([csi_features.detach(), power_context], dim=-1)
        direct_first_path_power = self.first_path_power_head(power_input).squeeze(-1)
        base_first_path_power = base[:, self.first_path_power_index]
        residual_scaled = self.first_path_power_residual_scale * (
            direct_first_path_power - base_first_path_power
        )
        final = base.clone()
        final[:, self.first_path_power_index] = (
            base_first_path_power
            + residual_scaled
        )
        return {
            "base": base,
            "direct_first_path_power": direct_first_path_power,
            "residual_scaled": residual_scaled,
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
