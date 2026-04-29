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


def paired_contrastive_loss(
    left_features: torch.Tensor,
    right_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    logits = logit_scale * left_features @ right_features.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_left_to_right = F.cross_entropy(logits, labels)
    loss_right_to_left = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_left_to_right + loss_right_to_left)


def instance_contrastive_loss(
    csi_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    return paired_contrastive_loss(csi_features, text_features, logit_scale)


def multipositive_contrastive_loss(
    csi_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    logits = logit_scale * csi_features @ text_features.T
    positive_mask = positive_mask.to(dtype=logits.dtype)
    positive_count = positive_mask.sum(dim=1).clamp(min=1.0)
    loss_csi_to_text = -(
        positive_mask * F.log_softmax(logits, dim=1)
    ).sum(dim=1) / positive_count

    positive_mask_t = positive_mask.T
    positive_count_t = positive_mask_t.sum(dim=1).clamp(min=1.0)
    loss_text_to_csi = -(
        positive_mask_t * F.log_softmax(logits.T, dim=1)
    ).sum(dim=1) / positive_count_t
    return 0.5 * (loss_csi_to_text.mean() + loss_text_to_csi.mean())


def masked_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    errors = F.smooth_l1_loss(predictions, targets, reduction="none")
    errors = errors * mask.to(dtype=errors.dtype)
    return errors.sum() / mask.sum().clamp(min=1).to(dtype=errors.dtype)


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
        loss_text_to_prototype = paired_contrastive_loss(
            text_features, prototype_features, logit_scale
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
