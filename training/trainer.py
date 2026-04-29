from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .losses import (
    PrototypeClipLoss,
    cosine_alignment_loss,
    instance_contrastive_loss,
    masked_regression_loss,
    multipositive_contrastive_loss,
    paired_contrastive_loss,
    semantic_classification_loss,
)


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 100
    prototype_weight: float = 1.0
    text_prototype_weight: float = 1.0
    text_mode: str = "prototype"
    aux_regression_weight: float = 0.0
    multipositive_distance_threshold: float = 0.25
    multipositive_positive_mode: str = "semantic_and_physics"
    min_class_size_for_multipositive: int = 2


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        prototype_token_ids: torch.Tensor,
        prototype_token_mask: torch.Tensor,
        prototype_label_map: dict[object, int],
        prototype_class_counts: torch.Tensor | None = None,
        loss: PrototypeClipLoss | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.prototype_token_ids = prototype_token_ids.to(device)
        self.prototype_token_mask = prototype_token_mask.to(device)
        self.prototype_label_map = prototype_label_map
        if prototype_class_counts is None:
            prototype_class_counts = torch.zeros(len(prototype_label_map), dtype=torch.long)
        self.prototype_class_counts = prototype_class_counts.to(device)
        self.loss = loss or PrototypeClipLoss()

    @staticmethod
    def _multipositive_mask(
        labels: torch.Tensor,
        physics_targets: torch.Tensor,
        physics_mask: torch.Tensor,
        class_counts: torch.Tensor,
        distance_threshold: float,
        positive_mode: str,
        min_class_size: int,
    ) -> torch.Tensor:
        semantic_has_enough_samples = class_counts[labels] >= min_class_size
        same_semantic = (labels[:, None] == labels[None, :]) & semantic_has_enough_samples[:, None]
        common_mask = physics_mask[:, None, :] & physics_mask[None, :, :]
        common_count = common_mask.sum(dim=-1)
        diffs = (physics_targets[:, None, :] - physics_targets[None, :, :]).abs()
        distances = (diffs * common_mask.to(dtype=diffs.dtype)).sum(dim=-1)
        distances = distances / common_count.clamp(min=1).to(dtype=diffs.dtype)
        near_physics = (common_count >= 3) & (distances <= distance_threshold)
        eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
        if positive_mode == "semantic_or_physics":
            positive_mask = same_semantic | near_physics
        elif positive_mode == "semantic_and_physics":
            positive_mask = same_semantic & near_physics
        elif positive_mode == "semantic":
            positive_mask = same_semantic
        elif positive_mode == "physics":
            positive_mask = near_physics
        else:
            raise ValueError(f"Unsupported multipositive_positive_mode={positive_mode!r}")
        return positive_mask | eye

    def _move_batch(self, batch: dict):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
            else:
                moved[key] = value
        return moved

    def train_step(self, batch: dict, epoch: int, cfg: TrainConfig) -> dict[str, float]:
        del epoch
        batch = self._move_batch(batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        csi_features = self.model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=True,
        )
        prototype_features = self.model.encode_prototypes(normalize=True)
        labels = torch.tensor(
            [self.prototype_label_map[key] for key in batch["semantic_keys"]],
            device=self.device,
            dtype=torch.long,
        )
        logit_scale = self.model.logit_scale.exp()
        physics_predictions = None
        if cfg.aux_regression_weight > 0:
            physics_predictions = self.model.predict_physics(csi_features)

        if cfg.text_mode == "prototype":
            text_features = self.model.encode_text(
                self.prototype_token_ids,
                self.prototype_token_mask,
                normalize=True,
            )
            losses = self.loss(
                csi_features=csi_features,
                text_features=text_features,
                prototype_features=prototype_features,
                logit_scale=logit_scale,
                labels=labels,
                output_dict=True,
            )
        elif cfg.text_mode in ("instance", "multipositive"):
            instance_text_features = self.model.encode_text(
                batch["t_instance_ids"],
                batch["t_instance_mask"],
                normalize=True,
            )
            prototype_text_features = self.model.encode_text(
                self.prototype_token_ids,
                self.prototype_token_mask,
                normalize=True,
            )
            if cfg.text_mode == "multipositive":
                positive_mask = self._multipositive_mask(
                    labels=labels,
                    physics_targets=batch["physics_targets"],
                    physics_mask=batch["physics_target_mask"],
                    class_counts=self.prototype_class_counts,
                    distance_threshold=cfg.multipositive_distance_threshold,
                    positive_mode=cfg.multipositive_positive_mode,
                    min_class_size=cfg.min_class_size_for_multipositive,
                )
                csi_to_text_loss = multipositive_contrastive_loss(
                    csi_features,
                    instance_text_features,
                    logit_scale,
                    positive_mask=positive_mask,
                )
            else:
                csi_to_text_loss = instance_contrastive_loss(
                    csi_features, instance_text_features, logit_scale
                )
            losses = {
                "loss_csi_to_text": csi_to_text_loss,
                "loss_csi_to_prototype": semantic_classification_loss(
                    csi_features, prototype_features, logit_scale, labels
                ),
                "loss_text_to_prototype": 0.5 * (
                    cosine_alignment_loss(instance_text_features, prototype_features[labels]) +
                    paired_contrastive_loss(prototype_text_features, prototype_features, logit_scale)
                ),
            }
        else:
            raise ValueError(f"Unsupported text_mode={cfg.text_mode!r}")
        if physics_predictions is not None:
            losses["loss_aux_regression"] = masked_regression_loss(
                physics_predictions,
                batch["physics_targets"],
                batch["physics_target_mask"],
            )
        total_loss = (
            losses["loss_csi_to_text"] +
            cfg.prototype_weight * losses["loss_csi_to_prototype"] +
            cfg.text_prototype_weight * losses["loss_text_to_prototype"] +
            cfg.aux_regression_weight * losses.get("loss_aux_regression", torch.zeros((), device=self.device))
        )
        total_loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.model.logit_scale.clamp_(0, math.log(100))
        metrics = {name: float(value.detach()) for name, value in losses.items()}
        metrics["prototype_weight"] = float(cfg.prototype_weight)
        metrics["text_prototype_weight"] = float(cfg.text_prototype_weight)
        metrics["text_mode_instance"] = float(cfg.text_mode == "instance")
        metrics["text_mode_multipositive"] = float(cfg.text_mode == "multipositive")
        metrics["aux_regression_weight"] = float(cfg.aux_regression_weight)
        metrics["min_class_size_for_multipositive"] = float(cfg.min_class_size_for_multipositive)
        metrics["multipositive_positive_count_mean"] = (
            float(positive_mask.sum(dim=1).float().mean().detach())
            if cfg.text_mode == "multipositive"
            else 0.0
        )
        metrics["contrastive_loss"] = float(total_loss.detach())
        metrics["loss_total"] = float(total_loss.detach())
        metrics["logit_scale"] = float(logit_scale.detach())
        return metrics
