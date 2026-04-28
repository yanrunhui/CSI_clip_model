from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def semantic_classification_loss(
    query_features: torch.Tensor,
    class_features: torch.Tensor,
    logit_scale: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = logit_scale * query_features @ class_features.T
    return F.cross_entropy(logits, labels)


def cosine_alignment_loss(
    left_features: torch.Tensor,
    right_features: torch.Tensor,
) -> torch.Tensor:
    return (1.0 - (left_features * right_features).sum(dim=-1)).mean()


class SemanticPrototypeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        csi_features: torch.Tensor,
        text_features: torch.Tensor,
        prototype_features: torch.Tensor,
        logit_scale: torch.Tensor,
        labels: torch.Tensor,
        output_dict: bool = False,
    ):
        loss_csi_to_text = semantic_classification_loss(
            csi_features, text_features, logit_scale, labels
        )
        loss_csi_to_prototype = semantic_classification_loss(
            csi_features, prototype_features, logit_scale, labels
        )
        loss_text_to_prototype = cosine_alignment_loss(
            text_features, prototype_features
        )
        if output_dict:
            return {
                "loss_csi_to_text": loss_csi_to_text,
                "loss_csi_to_prototype": loss_csi_to_prototype,
                "loss_text_to_prototype": loss_text_to_prototype,
            }
        return loss_csi_to_text + loss_csi_to_prototype + loss_text_to_prototype


PrototypeClipLoss = SemanticPrototypeLoss
ClipLoss = SemanticPrototypeLoss
