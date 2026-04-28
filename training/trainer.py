from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .losses import PrototypeClipLoss


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 100
    prototype_weight: float = 1.0
    text_prototype_weight: float = 1.0


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        prototype_token_ids: torch.Tensor,
        prototype_token_mask: torch.Tensor,
        prototype_label_map: dict[object, int],
        loss: PrototypeClipLoss | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.prototype_token_ids = prototype_token_ids.to(device)
        self.prototype_token_mask = prototype_token_mask.to(device)
        self.prototype_label_map = prototype_label_map
        self.loss = loss or PrototypeClipLoss()

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
        text_features = self.model.encode_text(
            self.prototype_token_ids,
            self.prototype_token_mask,
            normalize=True,
        )
        prototype_features = self.model.encode_prototypes(normalize=True)
        labels = torch.tensor(
            [self.prototype_label_map[key] for key in batch["semantic_keys"]],
            device=self.device,
            dtype=torch.long,
        )
        logit_scale = self.model.logit_scale.exp()
        losses = self.loss(
            csi_features=csi_features,
            text_features=text_features,
            prototype_features=prototype_features,
            logit_scale=logit_scale,
            labels=labels,
            output_dict=True,
        )
        total_loss = (
            losses["loss_csi_to_text"] +
            cfg.prototype_weight * losses["loss_csi_to_prototype"] +
            cfg.text_prototype_weight * losses["loss_text_to_prototype"]
        )
        total_loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.model.logit_scale.clamp_(0, math.log(100))
        metrics = {name: float(value.detach()) for name, value in losses.items()}
        metrics["prototype_weight"] = float(cfg.prototype_weight)
        metrics["text_prototype_weight"] = float(cfg.text_prototype_weight)
        metrics["contrastive_loss"] = float(total_loss.detach())
        metrics["loss_total"] = float(total_loss.detach())
        metrics["logit_scale"] = float(logit_scale.detach())
        return metrics
