from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from data.semantic_key import AttributeRemap, semantic_key_attribute_value

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
    csi_to_text_weight: float = 1.0
    prototype_weight: float = 1.0
    text_prototype_weight: float = 1.0
    text_mode: str = "prototype"
    semantic_classifier_weight: float = 0.0
    semantic_classifier_class_weight: str = "none"
    semantic_classifier_logit_adjustment: float = 0.0
    attribute_classifier_weight: float = 0.0
    attribute_classifier_class_weight: str = "none"
    attribute_classifier_logit_adjustment: float = 0.0
    aux_regression_weight: float = 0.0
    aux_regression_indices: tuple[int, ...] | None = None
    direct_power_weight: float = 0.0
    freeze_csi: bool = False
    freeze_text_prototypes: bool = False
    prototype_warmup_epochs: int = 0
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
        attribute_label_maps: dict[str, dict[str, int]] | None = None,
        attribute_class_counts: dict[str, torch.Tensor] | None = None,
        attribute_remap: AttributeRemap | None = None,
        loss: PrototypeClipLoss | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.prototype_token_ids = prototype_token_ids.to(device)
        self.prototype_token_mask = prototype_token_mask.to(device)
        self.prototype_label_map = prototype_label_map
        expected_indices = tuple(range(len(prototype_label_map)))
        actual_indices = tuple(sorted(int(index) for index in prototype_label_map.values()))
        if actual_indices != expected_indices:
            raise ValueError(
                "prototype_label_map indices must be contiguous and start at 0, "
                f"got {actual_indices}."
            )
        self.prototype_keys_by_index = tuple(
            key for key, _ in sorted(prototype_label_map.items(), key=lambda item: item[1])
        )
        if prototype_class_counts is None:
            prototype_class_counts = torch.zeros(len(prototype_label_map), dtype=torch.long)
        self.prototype_class_counts = prototype_class_counts.to(device)
        self.attribute_label_maps = attribute_label_maps or {}
        self.attribute_class_counts = {
            field: counts.to(device)
            for field, counts in (attribute_class_counts or {}).items()
        }
        self.attribute_remap = attribute_remap or {}
        self.loss = loss or PrototypeClipLoss()

    def _attribute_targets(self, semantic_keys: list[object]) -> dict[str, torch.Tensor]:
        targets = {}
        for field, label_map in self.attribute_label_maps.items():
            targets[field] = torch.tensor(
                [
                    label_map[semantic_key_attribute_value(key, field, self.attribute_remap)]
                    for key in semantic_keys
                ],
                device=self.device,
                dtype=torch.long,
            )
        return targets

    def _attribute_class_weight(
        self,
        field: str,
        dtype: torch.dtype,
        mode: str,
    ) -> torch.Tensor | None:
        counts = self.attribute_class_counts.get(field)
        if counts is None:
            return None
        exponent = 0.25 if mode == "mild" else 0.5
        weights = (counts.float().mean() / counts.float().clamp(min=1.0)).pow(exponent)
        return weights.clamp(0.25, 4.0).to(device=self.device, dtype=dtype)

    def _attribute_logit_adjustment(
        self,
        field: str,
        dtype: torch.dtype,
        strength: float,
    ) -> torch.Tensor | None:
        counts = self.attribute_class_counts.get(field)
        if counts is None or strength <= 0.0:
            return None
        priors = counts.float() / counts.float().sum().clamp(min=1.0)
        return (strength * priors.clamp(min=1e-6).log()).to(
            device=self.device,
            dtype=dtype,
        )

    def _semantic_class_weight(
        self,
        dtype: torch.dtype,
        mode: str,
    ) -> torch.Tensor | None:
        if self.prototype_class_counts.numel() == 0:
            return None
        exponent = 0.25 if mode == "mild" else 0.5
        weights = (
            self.prototype_class_counts.float().mean()
            / self.prototype_class_counts.float().clamp(min=1.0)
        ).pow(exponent)
        return weights.clamp(0.25, 4.0).to(device=self.device, dtype=dtype)

    def _semantic_logit_adjustment(
        self,
        dtype: torch.dtype,
        strength: float,
    ) -> torch.Tensor | None:
        if strength <= 0.0 or self.prototype_class_counts.numel() == 0:
            return None
        priors = self.prototype_class_counts.float() / self.prototype_class_counts.float().sum().clamp(min=1.0)
        return (strength * priors.clamp(min=1e-6).log()).to(
            device=self.device,
            dtype=dtype,
        )

    @staticmethod
    def _grad_norm(module: torch.nn.Module) -> float:
        squared_norm = 0.0
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            parameter_norm = parameter.grad.detach().float().norm(2)
            squared_norm += float(parameter_norm * parameter_norm)
        return math.sqrt(squared_norm)

    @staticmethod
    def _histogram(values: torch.Tensor, num_classes: int) -> torch.Tensor:
        return torch.bincount(values.detach().cpu(), minlength=num_classes)

    @staticmethod
    def _format_histogram(histogram: torch.Tensor) -> str:
        return ",".join(str(int(value)) for value in histogram.tolist())

    @staticmethod
    def _format_float_vector(values: torch.Tensor, precision: int = 4) -> str:
        return ",".join(f"{float(value):.{precision}f}" for value in values.detach().cpu().tolist())

    @staticmethod
    def _last_linear(module: torch.nn.Module | None) -> torch.nn.Linear | None:
        if module is None:
            return None
        if isinstance(module, torch.nn.Linear):
            return module
        if isinstance(module, torch.nn.Sequential):
            for layer in reversed(module):
                if isinstance(layer, torch.nn.Linear):
                    return layer
        for child in reversed(tuple(module.children())):
            linear = Trainer._last_linear(child)
            if linear is not None:
                return linear
        return None

    def _assert_semantic_label_roundtrip(
        self,
        semantic_keys: list[object],
        labels: torch.Tensor,
    ) -> None:
        for sample_idx, (key, label) in enumerate(zip(semantic_keys, labels.detach().cpu().tolist())):
            if label < 0 or label >= len(self.prototype_keys_by_index):
                raise ValueError(
                    f"Semantic label index out of range for sample {sample_idx}: "
                    f"label={label} num_prototypes={len(self.prototype_keys_by_index)}."
                )
            roundtrip_key = self.prototype_keys_by_index[label]
            if roundtrip_key != key:
                raise ValueError(
                    f"Semantic label roundtrip mismatch at sample {sample_idx}: "
                    f"key={key} label={label} roundtrip_key={roundtrip_key}."
                )

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
        batch = self._move_batch(batch)
        self.model.train()
        if cfg.freeze_csi:
            self.model.csi.eval()
        if cfg.freeze_text_prototypes:
            self.model.text.eval()
        warmup_active = epoch <= cfg.prototype_warmup_epochs
        effective_csi_to_text_weight = 0.0 if warmup_active else cfg.csi_to_text_weight
        effective_prototype_weight = 0.0 if warmup_active else cfg.prototype_weight
        effective_semantic_classifier_weight = 0.0 if warmup_active else cfg.semantic_classifier_weight
        effective_attribute_classifier_weight = 0.0 if warmup_active else cfg.attribute_classifier_weight
        effective_aux_regression_weight = 0.0 if warmup_active else cfg.aux_regression_weight
        self.optimizer.zero_grad(set_to_none=True)
        csi_features_raw = self.model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
        )
        csi_features = torch.nn.functional.normalize(csi_features_raw, dim=-1)
        prototype_features = self.model.encode_prototypes(normalize=True)
        prototype_text_features = self.model.encode_text(
            self.prototype_token_ids,
            self.prototype_token_mask,
            normalize=True,
        )
        labels = torch.tensor(
            [self.prototype_label_map[key] for key in batch["semantic_keys"]],
            device=self.device,
            dtype=torch.long,
        )
        self._assert_semantic_label_roundtrip(batch["semantic_keys"], labels)
        label_histogram = self._histogram(labels, len(self.prototype_keys_by_index))
        logit_scale = self.model.logit_scale.exp()
        positive_mask = None
        physics_predictions = None
        semantic_predictions = None
        semantic_prediction_histogram = None
        semantic_head_bias = None
        physics_outputs = None
        if effective_aux_regression_weight > 0 or cfg.direct_power_weight > 0.0:
            power_context = None
            if bool(getattr(self.model, "use_power_branch", False)):
                power_context = self.model.encode_power_context(
                    batch["tokens"],
                    batch["token_mask"],
                    delay_power_map=batch.get("delay_power_map"),
                    delay_power_profile=batch.get("delay_power_profile"),
                )
            physics_outputs = self.model.predict_physics_components(
                csi_features_raw,
                power_context=power_context,
            )
            physics_predictions = physics_outputs["final"]

        if cfg.text_mode == "prototype":
            losses = self.loss(
                csi_features=csi_features,
                text_features=prototype_text_features,
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
        if warmup_active and cfg.text_mode in {"instance", "multipositive"}:
            losses["loss_text_to_prototype"] = paired_contrastive_loss(
                prototype_text_features,
                prototype_features,
                logit_scale,
            )
        if effective_semantic_classifier_weight > 0:
            semantic_logits = self.model.predict_semantic(csi_features_raw)
            semantic_predictions = semantic_logits.argmax(dim=1)
            semantic_prediction_histogram = self._histogram(
                semantic_predictions,
                len(self.prototype_keys_by_index),
            )
            semantic_head_linear = self._last_linear(self.model.semantic_classifier)
            if semantic_head_linear is not None and semantic_head_linear.bias is not None:
                semantic_head_bias = semantic_head_linear.bias.detach()
            class_weight = (
                self._semantic_class_weight(
                    semantic_logits.dtype,
                    cfg.semantic_classifier_class_weight,
                )
                if cfg.semantic_classifier_class_weight in {"mild", "balanced"}
                else None
            )
            semantic_loss = torch.nn.functional.cross_entropy(
                semantic_logits + (
                    self._semantic_logit_adjustment(
                        semantic_logits.dtype,
                        cfg.semantic_classifier_logit_adjustment,
                    )
                    if cfg.semantic_classifier_logit_adjustment > 0.0
                    else 0.0
                ),
                labels,
                weight=class_weight,
            )
            losses["loss_semantic_classifier"] = semantic_loss
            losses["accuracy_semantic_classifier"] = (
                (semantic_predictions == labels).float().mean()
            )
            losses["logit_std_semantic_classifier"] = semantic_logits.detach().float().std()
        if effective_attribute_classifier_weight > 0:
            attribute_logits = self.model.predict_attributes(csi_features_raw)
            attribute_targets = self._attribute_targets(batch["semantic_keys"])
            attribute_losses = []
            attribute_accuracies = []
            attribute_logit_stds = []
            for field, targets in attribute_targets.items():
                logits = attribute_logits[field]
                class_weight = (
                    self._attribute_class_weight(
                        field,
                        logits.dtype,
                        cfg.attribute_classifier_class_weight,
                    )
                    if cfg.attribute_classifier_class_weight in {"mild", "balanced"}
                    else None
                )
                field_loss = torch.nn.functional.cross_entropy(
                    logits + (
                        self._attribute_logit_adjustment(
                            field,
                            logits.dtype,
                            cfg.attribute_classifier_logit_adjustment,
                        )
                        if cfg.attribute_classifier_logit_adjustment > 0.0
                        else 0.0
                    ),
                    targets,
                    weight=class_weight,
                )
                losses[f"loss_attribute_{field}"] = field_loss
                field_accuracy = (logits.argmax(dim=1) == targets).float().mean()
                losses[f"accuracy_attribute_{field}"] = field_accuracy
                losses[f"logit_std_attribute_{field}"] = logits.detach().float().std()
                attribute_losses.append(field_loss)
                attribute_accuracies.append(field_accuracy)
                attribute_logit_stds.append(logits.detach().float().std())
            if attribute_losses:
                losses["loss_attribute_classifier"] = torch.stack(attribute_losses).mean()
                losses["accuracy_attribute_classifier"] = torch.stack(attribute_accuracies).mean()
                losses["logit_std_attribute_classifier"] = torch.stack(attribute_logit_stds).mean()
        if physics_predictions is not None:
            regression_predictions = physics_predictions
            regression_targets = batch["physics_targets"]
            regression_mask = batch["physics_target_mask"]
            if cfg.aux_regression_indices:
                indices = torch.tensor(cfg.aux_regression_indices, device=self.device, dtype=torch.long)
                regression_predictions = regression_predictions.index_select(dim=1, index=indices)
                regression_targets = regression_targets.index_select(dim=1, index=indices)
                regression_mask = regression_mask.index_select(dim=1, index=indices)
            losses["loss_aux_regression"] = masked_regression_loss(
                regression_predictions,
                regression_targets,
                regression_mask,
            )
            if (
                physics_outputs is not None
                and bool(getattr(self.model, "use_power_branch", False))
                and cfg.direct_power_weight > 0.0
            ):
                first_path_power_idx = 5
                enhanced_first_path_power = physics_outputs["enhanced_first_path_power"]
                direct_power_target = batch["physics_targets"][:, first_path_power_idx]
                direct_power_mask = batch["physics_target_mask"][:, first_path_power_idx]
                direct_power_errors = torch.nn.functional.smooth_l1_loss(
                    enhanced_first_path_power,
                    direct_power_target,
                    reduction="none",
                )
                direct_power_errors = direct_power_errors * direct_power_mask.to(
                    dtype=direct_power_errors.dtype
                )
                losses["loss_direct_power"] = (
                    direct_power_errors.sum()
                    / direct_power_mask.sum().clamp(min=1).to(dtype=direct_power_errors.dtype)
                )
        total_loss = (
            effective_csi_to_text_weight * losses["loss_csi_to_text"] +
            effective_prototype_weight * losses["loss_csi_to_prototype"] +
            cfg.text_prototype_weight * losses["loss_text_to_prototype"] +
            effective_semantic_classifier_weight * losses.get("loss_semantic_classifier", torch.zeros((), device=self.device)) +
            effective_attribute_classifier_weight * losses.get("loss_attribute_classifier", torch.zeros((), device=self.device)) +
            effective_aux_regression_weight * losses.get("loss_aux_regression", torch.zeros((), device=self.device)) +
            cfg.direct_power_weight * losses.get("loss_direct_power", torch.zeros((), device=self.device))
        )
        total_loss.backward()
        grad_metrics = {
            "grad_norm_csi_encoder": self._grad_norm(self.model.csi),
            "grad_norm_semantic_classifier": (
                self._grad_norm(self.model.semantic_classifier)
                if hasattr(self.model, "semantic_classifier") and self.model.semantic_classifier is not None
                else 0.0
            ),
            "grad_norm_attribute_classifiers": (
                self._grad_norm(self.model.attribute_classifiers)
                if hasattr(self.model, "attribute_classifiers")
                else 0.0
            ),
        }
        self.optimizer.step()
        with torch.no_grad():
            self.model.logit_scale.clamp_(0, math.log(100))
        metrics = {name: float(value.detach()) for name, value in losses.items()}
        metrics.update(grad_metrics)
        metrics["csi_feature_raw_std"] = float(csi_features_raw.detach().float().std())
        metrics["csi_feature_normalized_std"] = float(csi_features.detach().float().std())
        metrics["csi_to_text_weight"] = float(effective_csi_to_text_weight)
        metrics["prototype_weight"] = float(effective_prototype_weight)
        metrics["text_prototype_weight"] = float(cfg.text_prototype_weight)
        metrics["text_mode_instance"] = float(cfg.text_mode == "instance")
        metrics["text_mode_multipositive"] = float(cfg.text_mode == "multipositive")
        metrics["semantic_classifier_weight"] = float(effective_semantic_classifier_weight)
        metrics["semantic_classifier_logit_adjustment"] = float(
            cfg.semantic_classifier_logit_adjustment
        )
        metrics["attribute_classifier_weight"] = float(effective_attribute_classifier_weight)
        metrics["attribute_classifier_logit_adjustment"] = float(
            cfg.attribute_classifier_logit_adjustment
        )
        metrics["aux_regression_weight"] = float(effective_aux_regression_weight)
        metrics["direct_power_weight"] = float(cfg.direct_power_weight)
        metrics["prototype_warmup_active"] = float(warmup_active)
        metrics["prototype_warmup_epochs"] = float(cfg.prototype_warmup_epochs)
        metrics["min_class_size_for_multipositive"] = float(cfg.min_class_size_for_multipositive)
        metrics["batch_label_unique_classes"] = float((label_histogram > 0).sum().item())
        metrics["batch_label_majority_fraction"] = float(
            label_histogram.max().item() / max(int(label_histogram.sum().item()), 1)
        )
        metrics["batch_label_histogram"] = self._format_histogram(label_histogram)
        metrics["multipositive_positive_count_mean"] = (
            float(positive_mask.sum(dim=1).float().mean().detach())
            if positive_mask is not None
            else 0.0
        )
        if semantic_predictions is not None and semantic_prediction_histogram is not None:
            metrics["batch_semantic_prediction_unique_classes"] = float(
                (semantic_prediction_histogram > 0).sum().item()
            )
            metrics["batch_semantic_prediction_majority_fraction"] = float(
                semantic_prediction_histogram.max().item()
                / max(int(semantic_prediction_histogram.sum().item()), 1)
            )
            metrics["batch_semantic_argmax_histogram"] = self._format_histogram(
                semantic_prediction_histogram
            )
            metrics["semantic_logit_mean"] = float(semantic_logits.detach().float().mean())
            metrics["semantic_logit_max_mean"] = float(
                semantic_logits.detach().float().max(dim=1).values.mean()
            )
        if semantic_head_bias is not None:
            metrics["semantic_head_bias_mean"] = float(semantic_head_bias.float().mean())
            metrics["semantic_head_bias_std"] = float(semantic_head_bias.float().std())
            metrics["semantic_head_bias_max"] = float(semantic_head_bias.float().max())
            metrics["semantic_head_bias_argmax"] = float(semantic_head_bias.argmax().item())
            metrics["semantic_head_bias_values"] = self._format_float_vector(semantic_head_bias)
        metrics["contrastive_loss"] = float(total_loss.detach())
        metrics["loss_total"] = float(total_loss.detach())
        metrics["logit_scale"] = float(logit_scale.detach())
        return metrics
