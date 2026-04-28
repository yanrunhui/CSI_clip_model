from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CSIClip(nn.Module):
    def __init__(
        self,
        csi_encoder: nn.Module,
        text_encoder: nn.Module,
        num_prototypes: int | None = None,
        embed_dim: int = 256,
        temperature: float = 0.07,
        output_dict: bool = True,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.csi = csi_encoder
        self.text = text_encoder
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        self.prototypes = None
        if num_prototypes is not None:
            self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim) * 0.02)

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
