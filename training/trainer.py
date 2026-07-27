from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from data.dataset import PHYSICS_TARGET_NAMES, PHYSICS_TARGET_OFFSETS, PHYSICS_TARGET_SCALES
from data.semantic_key import (
    FIRST_POWER_DBW_BIN_LABELS,
    FIRST_POWER_DBW_BINS,
    FIRST_POWER_DBW_POSITION_BINS,
    AttributeRemap,
    semantic_key_attribute_value,
)
from models.model import (
    DELAY_SPREAD_BIN_LABELS,
    DELAY_SPREAD_POSITION_BINS,
    DELAY_SPREAD_TAIL_LABELS,
    DELAY_SPREAD_TAIL_THRESHOLDS_NS,
    FIRST_PATH_DELAY_BIN_LABELS,
    FIRST_PATH_DELAY_POSITION_BINS,
    REFLECTION_COUNT_BIN_LABELS,
)

from .losses import (
    PrototypeClipLoss,
    cosine_alignment_loss,
    instance_contrastive_loss,
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
    k_factor_loss_weights: dict[str, float] | None = None
    strong_k_bin_classifier_weight: float = 0.0
    strong_k_position_weight: float = 0.0
    strong_k_bin_weights: dict[str, float] | None = None
    direct_power_weight: float = 0.0
    delay_spread_weight: float = 0.0
    delay_spread_raw_weight: float = 0.0
    delay_spread_raw_beta_ns: float = 20.0
    first_path_delay_weight: float = 0.0
    first_path_delay_raw_weight: float = 0.0
    first_path_delay_fused_raw_weight: float = 0.0
    first_path_delay_raw_beta_ns: float = 20.0
    first_path_delay_bin_classifier_weight: float = 0.0
    first_path_delay_bin_position_weight: float = 0.0
    first_path_delay_bin_consistency_weight: float = 0.0
    first_path_delay_bin_weights: dict[str, float] | None = None
    estimated_pdp_tail_bin_weight: float = 0.0
    estimated_pdp_tail_labels: tuple[str, ...] = ("1040_1280",)
    estimated_pdp_tail_gate_mode: str = "target"
    first_path_delay_tail_underestimate_weight: float = 0.0
    los_delay_weight: float = 0.0
    los_delay_nonnegative_weight: float = 0.0
    use_physics_calibration_loss: bool = False
    los_delay_consistency_weight: float = 0.0
    los_angle_weight: float = 0.0
    first_path_angle_weight: float = 0.0
    first_path_angle_nlos_weight: float = 0.0
    delay_spread_teacher_weight: float = 0.1
    delay_spread_bin_weights: dict[str, float] | None = None
    delay_spread_bin_classifier_weight: float = 0.0
    delay_spread_bin_position_weight: float = 0.0
    delay_spread_tail_classifier_weight: float = 0.0
    interaction_count_classifier_weight: float = 0.0
    interaction_count_regression_weight: float = 0.0
    reflection_count_classifier_weight: float = 0.0
    reflection_count_regression_weight: float = 0.0
    reflection_path_count_regression_weight: float = 0.0
    reflection_count_nlos_weight: float = 1.0
    interaction_count_soft_labels: bool = False
    physics_relational_weight: float = 0.0
    first_path_power_bin_classifier_weight: float = 0.0
    first_path_power_bin_position_weight: float = 0.0
    first_path_power_bin_weights: dict[str, float] | None = None
    first_path_power_nlos_weight: float = 1.0
    first_path_power_gate_mode: str = "none"
    first_path_power_mode: str = "residual"
    first_path_power_use_internal_gate: bool = True
    nlos_enhanced_power_loss: bool = False
    freeze_csi: bool = False
    freeze_text_prototypes: bool = False
    prototype_warmup_epochs: int = 0
    multipositive_distance_threshold: float = 0.25
    multipositive_positive_mode: str = "semantic_and_physics"
    min_class_size_for_multipositive: int = 2


class Trainer:
    FIRST_PATH_POWER_BINS = FIRST_POWER_DBW_BINS
    FIRST_PATH_POWER_POSITION_BINS = FIRST_POWER_DBW_POSITION_BINS
    FIRST_PATH_POWER_BIN_LABELS = FIRST_POWER_DBW_BIN_LABELS
    K_FACTOR_LOSS_BINS = (
        ("strong_low", 3.0, 15.0),
        ("strong_mid", 15.0, 30.0),
        ("strong_high", 30.0, 45.0),
        ("strong_very_high", 45.0, 70.0),
    )
    STRONG_K_BIN_LABELS = ("low", "mid", "high", "very_high")
    STRONG_K_POSITION_BINS = (
        ("low", 3.0, 15.0),
        ("mid", 15.0, 30.0),
        ("high", 30.0, 45.0),
        ("very_high", 45.0, 70.0),
    )
    DELAY_SPREAD_BIN_LABELS = DELAY_SPREAD_BIN_LABELS
    DELAY_SPREAD_TAIL_LABELS = DELAY_SPREAD_TAIL_LABELS
    DELAY_SPREAD_TAIL_THRESHOLDS_NS = DELAY_SPREAD_TAIL_THRESHOLDS_NS
    REFLECTION_COUNT_BIN_LABELS = REFLECTION_COUNT_BIN_LABELS
    REFLECTION_COUNT_BINS = (
        ("0_5", 0.0, 6.0),
        ("6_7", 6.0, 8.0),
        ("8_10", 8.0, 11.0),
        ("11_13", 11.0, 14.0),
        ("14_plus", 14.0, float("inf")),
    )
    DELAY_SPREAD_POSITION_BINS = DELAY_SPREAD_POSITION_BINS
    DELAY_SPREAD_BINS = (
        *DELAY_SPREAD_POSITION_BINS,
        ("400_plus", 400.0, float("inf")),
    )
    FIRST_PATH_DELAY_BIN_LABELS = FIRST_PATH_DELAY_BIN_LABELS
    FIRST_PATH_DELAY_POSITION_BINS = FIRST_PATH_DELAY_POSITION_BINS
    FIRST_PATH_DELAY_BINS = tuple(
        (label, lower, float("inf") if idx == len(FIRST_PATH_DELAY_POSITION_BINS) - 1 else upper)
        for idx, (label, lower, upper) in enumerate(FIRST_PATH_DELAY_POSITION_BINS)
    )

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
        model_bin_labels = tuple(
            getattr(model, "first_path_power_bin_labels", self.FIRST_PATH_POWER_BIN_LABELS)
        )
        if model_bin_labels != self.FIRST_PATH_POWER_BIN_LABELS:
            raise ValueError(
                "first-path-power bin label order mismatch: "
                f"model={model_bin_labels} trainer={self.FIRST_PATH_POWER_BIN_LABELS}."
            )
        bin_classifier_linear = self._last_linear(
            getattr(model, "first_path_power_bin_classifier", None)
        )
        if (
            bin_classifier_linear is not None
            and bin_classifier_linear.out_features != len(self.FIRST_PATH_POWER_BIN_LABELS)
        ):
            raise ValueError(
                "first-path-power bin classifier output size mismatch: "
                f"out_features={bin_classifier_linear.out_features} "
                f"labels={self.FIRST_PATH_POWER_BIN_LABELS}."
            )
        bin_position_linear = self._last_linear(
            getattr(model, "first_path_power_bin_position_head", None)
        )
        if bin_position_linear is not None and bin_position_linear.out_features != 1:
            raise ValueError(
                "first-path-power bin-position head output size mismatch: "
                f"out_features={bin_position_linear.out_features}, expected 1."
            )
        strong_k_labels = tuple(
            getattr(model, "k_factor_strong_bin_labels", self.STRONG_K_BIN_LABELS)
        )
        if strong_k_labels != self.STRONG_K_BIN_LABELS:
            raise ValueError(
                "strong K-factor bin label order mismatch: "
                f"model={strong_k_labels} trainer={self.STRONG_K_BIN_LABELS}."
            )
        strong_k_classifier_linear = self._last_linear(
            getattr(model, "k_factor_strong_bin_classifier", None)
        )
        if (
            strong_k_classifier_linear is not None
            and strong_k_classifier_linear.out_features != len(self.STRONG_K_BIN_LABELS)
        ):
            raise ValueError(
                "strong K-factor bin classifier output size mismatch: "
                f"out_features={strong_k_classifier_linear.out_features} "
                f"labels={self.STRONG_K_BIN_LABELS}."
            )
        strong_k_position_linear = self._last_linear(
            getattr(model, "k_factor_strong_position_head", None)
        )
        if strong_k_position_linear is not None and strong_k_position_linear.out_features != 1:
            raise ValueError(
                "strong K-factor bin-position head output size mismatch: "
                f"out_features={strong_k_position_linear.out_features}, expected 1."
            )
        delay_spread_labels = tuple(
            getattr(model, "delay_spread_bin_labels", self.DELAY_SPREAD_BIN_LABELS)
        )
        if delay_spread_labels != self.DELAY_SPREAD_BIN_LABELS:
            raise ValueError(
                "delay-spread bin label order mismatch: "
                f"model={delay_spread_labels} trainer={self.DELAY_SPREAD_BIN_LABELS}."
            )
        delay_spread_classifier_linear = self._last_linear(
            getattr(model, "delay_spread_bin_classifier", None)
        )
        if (
            delay_spread_classifier_linear is not None
            and delay_spread_classifier_linear.out_features != len(self.DELAY_SPREAD_BIN_LABELS)
        ):
            raise ValueError(
                "delay-spread bin classifier output size mismatch: "
                f"out_features={delay_spread_classifier_linear.out_features} "
                f"labels={self.DELAY_SPREAD_BIN_LABELS}."
            )
        delay_spread_position_linear = self._last_linear(
            getattr(model, "delay_spread_bin_position_head", None)
        )
        if delay_spread_position_linear is not None and delay_spread_position_linear.out_features != 1:
            raise ValueError(
                "delay-spread bin-position head output size mismatch: "
                f"out_features={delay_spread_position_linear.out_features}, expected 1."
            )
        first_path_delay_labels = tuple(
            getattr(model, "first_path_delay_bin_labels", self.FIRST_PATH_DELAY_BIN_LABELS)
        )
        if first_path_delay_labels != self.FIRST_PATH_DELAY_BIN_LABELS:
            raise ValueError(
                "first-path-delay bin label order mismatch: "
                f"model={first_path_delay_labels} trainer={self.FIRST_PATH_DELAY_BIN_LABELS}."
            )
        first_path_delay_classifier_linear = self._last_linear(
            getattr(model, "first_path_delay_bin_classifier", None)
        )
        if (
            first_path_delay_classifier_linear is not None
            and first_path_delay_classifier_linear.out_features
            != len(self.FIRST_PATH_DELAY_BIN_LABELS)
        ):
            raise ValueError(
                "first-path-delay bin classifier output size mismatch: "
                f"out_features={first_path_delay_classifier_linear.out_features} "
                f"labels={self.FIRST_PATH_DELAY_BIN_LABELS}."
            )
        first_path_delay_position_linear = self._last_linear(
            getattr(model, "first_path_delay_bin_position_head", None)
        )
        if (
            first_path_delay_position_linear is not None
            and first_path_delay_position_linear.out_features != 1
        ):
            raise ValueError(
                "first-path-delay bin-position head output size mismatch: "
                f"out_features={first_path_delay_position_linear.out_features}, expected 1."
            )
        delay_spread_tail_labels = tuple(
            getattr(model, "delay_spread_tail_labels", self.DELAY_SPREAD_TAIL_LABELS)
        )
        if delay_spread_tail_labels != self.DELAY_SPREAD_TAIL_LABELS:
            raise ValueError(
                "delay-spread tail label order mismatch: "
                f"model={delay_spread_tail_labels} trainer={self.DELAY_SPREAD_TAIL_LABELS}."
            )
        delay_spread_tail_linear = self._last_linear(
            getattr(model, "delay_spread_tail_classifier", None)
        )
        if (
            delay_spread_tail_linear is not None
            and delay_spread_tail_linear.out_features != len(self.DELAY_SPREAD_TAIL_LABELS)
        ):
            raise ValueError(
                "delay-spread tail classifier output size mismatch: "
                f"out_features={delay_spread_tail_linear.out_features} "
                f"labels={self.DELAY_SPREAD_TAIL_LABELS}."
            )
        reflection_count_labels = tuple(
            getattr(model, "reflection_count_bin_labels", self.REFLECTION_COUNT_BIN_LABELS)
        )
        if reflection_count_labels != self.REFLECTION_COUNT_BIN_LABELS:
            raise ValueError(
                "reflection-count bin label order mismatch: "
                f"model={reflection_count_labels} "
                f"trainer={self.REFLECTION_COUNT_BIN_LABELS}."
            )
        reflection_count_linear = self._last_linear(
            getattr(model, "reflection_count_classifier", None)
        )
        if (
            reflection_count_linear is not None
            and reflection_count_linear.out_features != len(self.REFLECTION_COUNT_BIN_LABELS)
        ):
            raise ValueError(
                "reflection-count classifier output size mismatch: "
                f"out_features={reflection_count_linear.out_features} "
                f"labels={self.REFLECTION_COUNT_BIN_LABELS}."
            )
        for name in ("reflection_count_regression_head", "reflection_path_count_regression_head"):
            linear = self._last_linear(getattr(model, name, None))
            if linear is not None and linear.out_features != 1:
                raise ValueError(
                    f"{name} output size mismatch: "
                    f"out_features={linear.out_features}, expected 1."
                )

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

    def _first_path_power_sample_weights(
        self,
        raw_first_path_power: torch.Tensor,
        cfg: TrainConfig,
    ) -> torch.Tensor:
        weights_cfg = cfg.first_path_power_bin_weights or {}
        weights = torch.ones_like(raw_first_path_power)
        for label, lower, upper in self.FIRST_PATH_POWER_BINS:
            bin_weight = float(weights_cfg.get(label, 1.0))
            if bin_weight == 1.0:
                continue
            mask = (raw_first_path_power >= lower) & (raw_first_path_power < upper)
            weights = torch.where(mask, torch.full_like(weights, bin_weight), weights)
        return weights

    def _first_path_power_supervision_weights(
        self,
        raw_first_path_power: torch.Tensor,
        semantic_keys: list[object],
        cfg: TrainConfig,
    ) -> torch.Tensor:
        weights = self._first_path_power_sample_weights(raw_first_path_power, cfg)
        nlos_weight = float(cfg.first_path_power_nlos_weight)
        if nlos_weight != 1.0:
            nlos_mask = torch.tensor(
                [
                    getattr(key, "los_status", None) != "los"
                    for key in semantic_keys
                ],
                dtype=torch.bool,
                device=raw_first_path_power.device,
            )
            weights = torch.where(
                nlos_mask,
                weights * nlos_weight,
                weights,
            )
        return weights

    def _k_factor_sample_weights(
        self,
        raw_k_factor: torch.Tensor,
        semantic_keys: list[object],
        cfg: TrainConfig,
    ) -> torch.Tensor:
        weights_cfg = cfg.k_factor_loss_weights or {}
        weights = torch.ones_like(raw_k_factor)
        if not weights_cfg:
            return weights

        weak_weight = float(weights_cfg.get("weak", 1.0))
        if weak_weight != 1.0:
            weak_mask = torch.tensor(
                [getattr(key, "k_factor_bin", None) == "weak" for key in semantic_keys],
                dtype=torch.bool,
                device=raw_k_factor.device,
            )
            weights = torch.where(weak_mask, torch.full_like(weights, weak_weight), weights)

        strong_mask = torch.tensor(
            [getattr(key, "k_factor_bin", None) == "strong" for key in semantic_keys],
            dtype=torch.bool,
            device=raw_k_factor.device,
        )
        for bin_idx, (label, lower, upper) in enumerate(self.K_FACTOR_LOSS_BINS):
            bin_weight = float(weights_cfg.get(label, 1.0))
            if bin_weight == 1.0:
                continue
            upper_mask = (
                raw_k_factor <= upper
                if bin_idx == len(self.K_FACTOR_LOSS_BINS) - 1
                else raw_k_factor < upper
            )
            mask = strong_mask & (raw_k_factor >= lower) & upper_mask
            weights = torch.where(mask, torch.full_like(weights, bin_weight), weights)
        return weights

    def _delay_spread_sample_weights(
        self,
        raw_delay_spread_ns: torch.Tensor,
        cfg: TrainConfig,
    ) -> torch.Tensor:
        weights_cfg = cfg.delay_spread_bin_weights or {}
        weights = torch.ones_like(raw_delay_spread_ns)
        if not weights_cfg:
            return weights

        for bin_idx, (label, lower, upper) in enumerate(self.DELAY_SPREAD_BINS):
            bin_weight = float(weights_cfg.get(label, 1.0))
            if bin_weight == 1.0:
                continue
            upper_mask = (
                raw_delay_spread_ns <= upper
                if bin_idx == len(self.DELAY_SPREAD_BINS) - 1
                else raw_delay_spread_ns < upper
            )
            mask = (raw_delay_spread_ns >= lower) & upper_mask
            weights = torch.where(mask, torch.full_like(weights, bin_weight), weights)
        return weights

    def _delay_spread_bin_targets(self, raw_delay_spread_ns: torch.Tensor) -> torch.Tensor:
        targets = torch.full_like(raw_delay_spread_ns, fill_value=-1, dtype=torch.long)
        for class_idx, (_, lower, upper) in enumerate(self.DELAY_SPREAD_BINS):
            upper_mask = (
                raw_delay_spread_ns <= upper
                if class_idx == len(self.DELAY_SPREAD_BINS) - 1
                else raw_delay_spread_ns < upper
            )
            mask = torch.isfinite(raw_delay_spread_ns) & (raw_delay_spread_ns >= lower) & upper_mask
            targets = torch.where(mask, torch.full_like(targets, class_idx), targets)
        return targets

    def _delay_spread_bin_class_weight(
        self,
        dtype: torch.dtype,
        cfg: TrainConfig,
    ) -> torch.Tensor | None:
        if not cfg.delay_spread_bin_weights:
            return None
        return torch.tensor(
            [
                float(cfg.delay_spread_bin_weights.get(label, 1.0))
                for label in self.DELAY_SPREAD_BIN_LABELS
            ],
            device=self.device,
            dtype=dtype,
        )

    def _delay_spread_bin_position_targets(
        self,
        raw_delay_spread_ns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.zeros_like(raw_delay_spread_ns)
        valid_mask = torch.zeros_like(raw_delay_spread_ns, dtype=torch.bool)
        for _, lower, upper in self.DELAY_SPREAD_POSITION_BINS:
            mask = torch.isfinite(raw_delay_spread_ns) & (raw_delay_spread_ns >= lower) & (raw_delay_spread_ns < upper)
            position = (raw_delay_spread_ns - lower) / max(upper - lower, 1e-6)
            targets = torch.where(mask, position.clamp(0.0, 1.0), targets)
            valid_mask = valid_mask | mask
        return targets, valid_mask

    def _first_path_delay_sample_weights(
        self,
        raw_first_path_delay_ns: torch.Tensor,
        cfg: TrainConfig,
    ) -> torch.Tensor:
        weights_cfg = cfg.first_path_delay_bin_weights or {}
        weights = torch.ones_like(raw_first_path_delay_ns)
        if not weights_cfg:
            return weights

        for bin_idx, (label, lower, upper) in enumerate(self.FIRST_PATH_DELAY_BINS):
            bin_weight = float(weights_cfg.get(label, 1.0))
            if bin_weight == 1.0:
                continue
            upper_mask = (
                raw_first_path_delay_ns <= upper
                if bin_idx == len(self.FIRST_PATH_DELAY_BINS) - 1
                else raw_first_path_delay_ns < upper
            )
            mask = torch.isfinite(raw_first_path_delay_ns) & (raw_first_path_delay_ns >= lower) & upper_mask
            weights = torch.where(mask, torch.full_like(weights, bin_weight), weights)
        return weights

    def _first_path_delay_bin_targets(self, raw_first_path_delay_ns: torch.Tensor) -> torch.Tensor:
        targets = torch.full_like(raw_first_path_delay_ns, fill_value=-1, dtype=torch.long)
        for class_idx, (_, lower, upper) in enumerate(self.FIRST_PATH_DELAY_BINS):
            upper_mask = (
                raw_first_path_delay_ns <= upper
                if class_idx == len(self.FIRST_PATH_DELAY_BINS) - 1
                else raw_first_path_delay_ns < upper
            )
            mask = (
                torch.isfinite(raw_first_path_delay_ns)
                & (raw_first_path_delay_ns >= lower)
                & upper_mask
            )
            targets = torch.where(mask, torch.full_like(targets, class_idx), targets)
        return targets

    def _first_path_delay_bin_class_weight(
        self,
        dtype: torch.dtype,
        cfg: TrainConfig,
    ) -> torch.Tensor | None:
        if not cfg.first_path_delay_bin_weights:
            return None
        return torch.tensor(
            [
                float(cfg.first_path_delay_bin_weights.get(label, 1.0))
                for label in self.FIRST_PATH_DELAY_BIN_LABELS
            ],
            device=self.device,
            dtype=dtype,
        )

    def _first_path_delay_tail_indices(self, labels: tuple[str, ...]) -> torch.Tensor:
        label_to_idx = {label: idx for idx, label in enumerate(self.FIRST_PATH_DELAY_BIN_LABELS)}
        unknown = sorted(set(labels) - set(label_to_idx))
        if unknown:
            raise ValueError(
                f"Unknown estimated-PDP first-path-delay tail labels: {unknown}. "
                f"Choose from: {', '.join(self.FIRST_PATH_DELAY_BIN_LABELS)}."
            )
        return torch.tensor(
            [label_to_idx[label] for label in labels],
            device=self.device,
            dtype=torch.long,
        )

    def _estimated_pdp_argmax_delay_bins(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        subcarrier_spacing_hz: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[2] % 2 != 0:
            return torch.full(
                (tokens.shape[0],),
                fill_value=-1,
                device=tokens.device,
                dtype=torch.long,
            )
        half = tokens.shape[2] // 2
        weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        complex_tokens = torch.complex(
            (tokens[:, :, :half, :] * weights).float(),
            (tokens[:, :, half:, :] * weights).float(),
        )
        delay_response = torch.fft.ifft(complex_tokens, dim=-1)
        profile = delay_response.abs().square().sum(dim=(1, 2)).float()
        peak_idx = profile.argmax(dim=-1).to(dtype=torch.float32)
        n_freq = max(int(tokens.shape[-1]), 1)
        spacing = subcarrier_spacing_hz.to(device=tokens.device, dtype=torch.float32).clamp(min=1e-6)
        peak_delay_ns = peak_idx / (n_freq * spacing) * 1e9
        return self._first_path_delay_bin_targets(peak_delay_ns)

    def _estimated_pdp_tail_gate_mask(
        self,
        *,
        true_targets: torch.Tensor,
        pdp_targets: torch.Tensor,
        tail_indices: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_tail = (true_targets.unsqueeze(1) == tail_indices.unsqueeze(0)).any(dim=1)
        pdp_tail = (pdp_targets.unsqueeze(1) == tail_indices.unsqueeze(0)).any(dim=1)
        if mode == "target":
            gate = target_tail
        elif mode == "pdp_argmax":
            gate = pdp_tail
        elif mode == "target_or_pdp_argmax":
            gate = target_tail | pdp_tail
        elif mode == "target_and_pdp_argmax":
            gate = target_tail & pdp_tail
        else:
            raise ValueError(
                "estimated_pdp_tail_gate_mode must be one of "
                "target, pdp_argmax, target_or_pdp_argmax, target_and_pdp_argmax; "
                f"got {mode!r}."
            )
        return gate, target_tail, pdp_tail

    def _first_path_delay_bin_position_targets(
        self,
        raw_first_path_delay_ns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.zeros_like(raw_first_path_delay_ns)
        valid_mask = torch.zeros_like(raw_first_path_delay_ns, dtype=torch.bool)
        for _, lower, upper in self.FIRST_PATH_DELAY_POSITION_BINS:
            mask = (
                torch.isfinite(raw_first_path_delay_ns)
                & (raw_first_path_delay_ns >= lower)
                & (raw_first_path_delay_ns < upper)
            )
            position = (raw_first_path_delay_ns - lower) / max(upper - lower, 1e-6)
            targets = torch.where(mask, position.clamp(0.0, 1.0), targets)
            valid_mask = valid_mask | mask
        return targets, valid_mask

    def _first_path_delay_bin_bounds(
        self,
        raw_first_path_delay_ns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lower_targets = torch.zeros_like(raw_first_path_delay_ns)
        upper_targets = torch.zeros_like(raw_first_path_delay_ns)
        finite_upper_mask = torch.zeros_like(raw_first_path_delay_ns, dtype=torch.bool)
        valid_mask = torch.zeros_like(raw_first_path_delay_ns, dtype=torch.bool)
        for class_idx, (_, lower, upper) in enumerate(self.FIRST_PATH_DELAY_BINS):
            upper_mask = (
                raw_first_path_delay_ns <= upper
                if class_idx == len(self.FIRST_PATH_DELAY_BINS) - 1
                else raw_first_path_delay_ns < upper
            )
            mask = (
                torch.isfinite(raw_first_path_delay_ns)
                & (raw_first_path_delay_ns >= lower)
                & upper_mask
            )
            lower_targets = torch.where(mask, torch.full_like(lower_targets, lower), lower_targets)
            if math.isfinite(upper):
                upper_targets = torch.where(mask, torch.full_like(upper_targets, upper), upper_targets)
                finite_upper_mask = finite_upper_mask | mask
            valid_mask = valid_mask | mask
        return lower_targets, upper_targets, finite_upper_mask, valid_mask

    def _first_path_power_bin_targets(self, raw_first_path_power: torch.Tensor) -> torch.Tensor:
        targets = torch.full_like(raw_first_path_power, fill_value=-1, dtype=torch.long)
        for class_idx, (_, lower, upper) in enumerate(self.FIRST_PATH_POWER_BINS):
            mask = (raw_first_path_power >= lower) & (raw_first_path_power < upper)
            targets = torch.where(mask, torch.full_like(targets, class_idx), targets)
        return targets

    def _interaction_count_bin_targets(
        self,
        raw_count: torch.Tensor,
        bins: tuple[tuple[str, float, float], ...],
    ) -> torch.Tensor:
        targets = torch.full_like(raw_count, fill_value=-1, dtype=torch.long)
        finite = torch.isfinite(raw_count)
        rounded = raw_count.round()
        for class_idx, (_, lower, upper) in enumerate(bins):
            upper_mask = rounded <= upper if math.isinf(upper) else rounded < upper
            mask = finite & (rounded >= lower) & upper_mask
            targets = torch.where(mask, torch.full_like(targets, class_idx), targets)
        return targets

    def _interaction_count_soft_targets(
        self,
        hard_targets: torch.Tensor,
        num_classes: int,
        target_mass: float = 0.8,
    ) -> torch.Tensor:
        soft_targets = torch.zeros(
            hard_targets.shape[0],
            num_classes,
            device=hard_targets.device,
            dtype=torch.float32,
        )
        soft_targets.scatter_(1, hard_targets.unsqueeze(1), target_mass)
        neighbor_mass = (1.0 - target_mass) * 0.5
        previous_mask = hard_targets > 0
        if bool(previous_mask.any()):
            soft_targets[
                previous_mask,
                hard_targets[previous_mask] - 1,
            ] += neighbor_mass
        next_mask = hard_targets < num_classes - 1
        if bool(next_mask.any()):
            soft_targets[
                next_mask,
                hard_targets[next_mask] + 1,
            ] += neighbor_mass
        return soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp(min=1e-12)

    def _reflection_count_sample_weights(
        self,
        semantic_keys: list[object],
        *,
        device: torch.device,
        dtype: torch.dtype,
        cfg: TrainConfig,
    ) -> torch.Tensor:
        weights = torch.ones(len(semantic_keys), device=device, dtype=dtype)
        nlos_weight = float(cfg.reflection_count_nlos_weight)
        if nlos_weight != 1.0:
            nlos_mask = torch.tensor(
                [
                    getattr(key, "los_status", None) != "los"
                    for key in semantic_keys
                ],
                dtype=torch.bool,
                device=device,
            )
            weights = torch.where(
                nlos_mask,
                weights * nlos_weight,
                weights,
            )
        return weights

    def _physics_relational_losses(
        self,
        physics_outputs: dict[str, torch.Tensor],
        batch: dict,
    ) -> dict[str, torch.Tensor]:
        final = physics_outputs["final"]
        device = final.device
        dtype = final.dtype
        scales = PHYSICS_TARGET_SCALES.to(device=device, dtype=dtype)
        offsets = PHYSICS_TARGET_OFFSETS.to(device=device, dtype=dtype)
        raw_final = final * scales + offsets
        losses: dict[str, torch.Tensor] = {}

        path_count_idx = PHYSICS_TARGET_NAMES.index("n_paths")
        delay_spread_idx = PHYSICS_TARGET_NAMES.index("delay_spread_ns")
        first_path_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
        reflection_count_idx = PHYSICS_TARGET_NAMES.index("reflection_count")

        path_count_raw = raw_final[:, path_count_idx]
        delay_spread_raw = raw_final[:, delay_spread_idx]
        reflection_count_raw = raw_final[:, reflection_count_idx]

        path_mask = batch["physics_target_mask"][:, path_count_idx].bool() & torch.isfinite(path_count_raw)
        if bool(path_mask.any()):
            losses["loss_physics_relational_path_count"] = (
                torch.relu(1.0 - path_count_raw[path_mask]) / scales[path_count_idx]
            ).mean()

        delay_spread_mask = (
            batch["physics_target_mask"][:, delay_spread_idx].bool()
            & torch.isfinite(delay_spread_raw)
        )
        if bool(delay_spread_mask.any()):
            losses["loss_physics_relational_delay_spread"] = (
                torch.relu(-delay_spread_raw[delay_spread_mask]) / scales[delay_spread_idx]
            ).mean()

        reflection_mask = (
            batch["physics_target_mask"][:, reflection_count_idx].bool()
            & torch.isfinite(reflection_count_raw)
        )
        if bool(reflection_mask.any()):
            losses["loss_physics_relational_reflection_nonnegative"] = (
                torch.relu(-reflection_count_raw[reflection_mask]) / scales[reflection_count_idx]
            ).mean()

        first_path_delay_raw = physics_outputs.get("first_path_delay_bin_soft_fused_raw")
        if first_path_delay_raw is None:
            first_path_delay_raw = raw_final[:, first_path_delay_idx]
        los_delay_raw = physics_outputs["los_delay_context"] * scales[first_path_delay_idx]
        los_sample_mask = torch.tensor(
            [getattr(key, "los_status", None) == "los" for key in batch["semantic_keys"]],
            device=device,
            dtype=torch.bool,
        )
        los_delay_mask = batch["los_delay_target_mask"].bool()
        relational_mask = (
            los_sample_mask
            & los_delay_mask
            & torch.isfinite(first_path_delay_raw)
            & torch.isfinite(los_delay_raw)
        )
        if bool(relational_mask.any()):
            eps = torch.tensor(1e-6, device=device, dtype=dtype)
            losses["loss_physics_relational_los_delay_positive"] = (
                torch.relu(eps - los_delay_raw[relational_mask]) / scales[first_path_delay_idx]
            ).mean()
            losses["loss_physics_relational_first_path_delay_ge_los_delay"] = (
                torch.relu(
                    los_delay_raw[relational_mask] - first_path_delay_raw[relational_mask]
                )
                / scales[first_path_delay_idx]
            ).mean()

        if losses:
            losses["loss_physics_relational"] = torch.stack(
                list(losses.values())
            ).mean()
        return losses

    def _first_path_power_bin_class_weight(
        self,
        dtype: torch.dtype,
        cfg: TrainConfig,
    ) -> torch.Tensor | None:
        if not cfg.first_path_power_bin_weights:
            return None
        return torch.tensor(
            [
                float(cfg.first_path_power_bin_weights.get(label, 1.0))
                for label, _, _ in self.FIRST_PATH_POWER_BINS
            ],
            device=self.device,
            dtype=dtype,
        )

    def _first_path_power_bin_position_targets(
        self,
        raw_first_path_power: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.zeros_like(raw_first_path_power)
        valid_mask = torch.zeros_like(raw_first_path_power, dtype=torch.bool)
        for _, lower, upper in self.FIRST_PATH_POWER_POSITION_BINS:
            mask = (raw_first_path_power >= lower) & (raw_first_path_power < upper)
            if upper == self.FIRST_PATH_POWER_POSITION_BINS[-1][2]:
                mask = (raw_first_path_power >= lower) & (raw_first_path_power <= upper)
            position = (raw_first_path_power - lower) / max(upper - lower, 1e-6)
            targets = torch.where(mask, position.clamp(0.0, 1.0), targets)
            valid_mask = valid_mask | mask
        finite_mask = torch.isfinite(raw_first_path_power)
        clipped_raw = raw_first_path_power.clamp(
            min=self.FIRST_PATH_POWER_POSITION_BINS[0][1],
            max=self.FIRST_PATH_POWER_POSITION_BINS[-1][2],
        )
        for _, lower, upper in self.FIRST_PATH_POWER_POSITION_BINS:
            mask = finite_mask & ~valid_mask & (clipped_raw >= lower) & (clipped_raw <= upper)
            position = (clipped_raw - lower) / max(upper - lower, 1e-6)
            targets = torch.where(mask, position.clamp(0.0, 1.0), targets)
            valid_mask = valid_mask | mask
        return targets, valid_mask

    def _strong_k_bin_targets(
        self,
        raw_k_factor: torch.Tensor,
        semantic_keys: list[object],
    ) -> torch.Tensor:
        targets = torch.full_like(raw_k_factor, fill_value=-1, dtype=torch.long)
        strong_mask = torch.tensor(
            [getattr(key, "k_factor_bin", None) == "strong" for key in semantic_keys],
            dtype=torch.bool,
            device=raw_k_factor.device,
        )
        for class_idx, (_, lower, upper) in enumerate(self.STRONG_K_POSITION_BINS):
            upper_mask = (
                raw_k_factor <= upper
                if class_idx == len(self.STRONG_K_POSITION_BINS) - 1
                else raw_k_factor < upper
            )
            mask = strong_mask & (raw_k_factor >= lower) & upper_mask
            targets = torch.where(mask, torch.full_like(targets, class_idx), targets)
        return targets

    def _strong_k_position_targets(
        self,
        raw_k_factor: torch.Tensor,
        semantic_keys: list[object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        targets = torch.zeros_like(raw_k_factor)
        valid_mask = torch.zeros_like(raw_k_factor, dtype=torch.bool)
        strong_mask = torch.tensor(
            [getattr(key, "k_factor_bin", None) == "strong" for key in semantic_keys],
            dtype=torch.bool,
            device=raw_k_factor.device,
        )
        for bin_idx, (_, lower, upper) in enumerate(self.STRONG_K_POSITION_BINS):
            upper_mask = (
                raw_k_factor <= upper
                if bin_idx == len(self.STRONG_K_POSITION_BINS) - 1
                else raw_k_factor < upper
            )
            mask = strong_mask & (raw_k_factor >= lower) & upper_mask
            position = (raw_k_factor - lower) / max(upper - lower, 1e-6)
            targets = torch.where(mask, position.clamp(0.0, 1.0), targets)
            valid_mask = valid_mask | mask
        return targets, valid_mask

    def _strong_k_bin_class_weight(
        self,
        dtype: torch.dtype,
        cfg: TrainConfig,
    ) -> torch.Tensor | None:
        if not cfg.strong_k_bin_weights:
            return None
        return torch.tensor(
            [
                float(cfg.strong_k_bin_weights.get(label, 1.0))
                for label in self.STRONG_K_BIN_LABELS
            ],
            device=self.device,
            dtype=dtype,
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

    def _predicted_los_mask_from_prototypes(
        self,
        csi_features: torch.Tensor,
        prototype_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        prototype_logits = logit_scale * csi_features @ prototype_features.T
        predicted_labels = prototype_logits.argmax(dim=1)
        return torch.tensor(
            [
                getattr(self.prototype_keys_by_index[int(label)], "los_status", None) == "los"
                for label in predicted_labels.detach().cpu().tolist()
            ],
            device=self.device,
            dtype=torch.bool,
        )

    def _apply_first_path_power_gate(
        self,
        physics_outputs: dict[str, torch.Tensor],
        predicted_los_mask: torch.Tensor,
    ) -> torch.Tensor:
        final = physics_outputs["final"].clone()
        first_path_power_idx = 5
        final[:, first_path_power_idx] = torch.where(
            predicted_los_mask,
            physics_outputs["base"][:, first_path_power_idx],
            physics_outputs["enhanced_first_path_power"],
        )
        return final

    def _apply_first_path_power_base(
        self,
        physics_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        final = physics_outputs["final"].clone()
        first_path_power_idx = 5
        final[:, first_path_power_idx] = physics_outputs["base"][:, first_path_power_idx]
        return final

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
        effective_strong_k_bin_classifier_weight = (
            0.0 if warmup_active else cfg.strong_k_bin_classifier_weight
        )
        effective_strong_k_position_weight = (
            0.0 if warmup_active else cfg.strong_k_position_weight
        )
        effective_first_path_power_bin_classifier_weight = (
            0.0 if warmup_active else cfg.first_path_power_bin_classifier_weight
        )
        effective_first_path_power_bin_position_weight = (
            0.0 if warmup_active else cfg.first_path_power_bin_position_weight
        )
        effective_delay_spread_weight = (
            0.0 if warmup_active else cfg.delay_spread_weight
        )
        effective_delay_spread_raw_weight = (
            0.0 if warmup_active else cfg.delay_spread_raw_weight
        )
        effective_first_path_delay_weight = (
            0.0 if warmup_active else cfg.first_path_delay_weight
        )
        effective_first_path_delay_raw_weight = (
            0.0 if warmup_active else cfg.first_path_delay_raw_weight
        )
        effective_first_path_delay_fused_raw_weight = (
            0.0 if warmup_active else cfg.first_path_delay_fused_raw_weight
        )
        effective_first_path_delay_bin_classifier_weight = (
            0.0 if warmup_active else cfg.first_path_delay_bin_classifier_weight
        )
        effective_first_path_delay_bin_position_weight = (
            0.0 if warmup_active else cfg.first_path_delay_bin_position_weight
        )
        effective_first_path_delay_bin_consistency_weight = (
            0.0 if warmup_active else cfg.first_path_delay_bin_consistency_weight
        )
        effective_estimated_pdp_tail_bin_weight = (
            0.0 if warmup_active else cfg.estimated_pdp_tail_bin_weight
        )
        effective_first_path_delay_tail_underestimate_weight = (
            0.0 if warmup_active else cfg.first_path_delay_tail_underestimate_weight
        )
        effective_los_delay_weight = (
            0.0 if warmup_active else cfg.los_delay_weight
        )
        effective_los_delay_nonnegative_weight = (
            0.0 if warmup_active else cfg.los_delay_nonnegative_weight
        )
        effective_los_delay_consistency_weight = (
            0.0
            if warmup_active or not cfg.use_physics_calibration_loss
            else cfg.los_delay_consistency_weight
        )
        effective_los_angle_weight = (
            0.0 if warmup_active else cfg.los_angle_weight
        )
        effective_first_path_angle_weight = (
            0.0 if warmup_active else cfg.first_path_angle_weight
        )
        effective_first_path_angle_nlos_weight = (
            0.0 if warmup_active else cfg.first_path_angle_nlos_weight
        )
        effective_delay_spread_bin_classifier_weight = (
            0.0 if warmup_active else cfg.delay_spread_bin_classifier_weight
        )
        effective_delay_spread_bin_position_weight = (
            0.0 if warmup_active else cfg.delay_spread_bin_position_weight
        )
        effective_delay_spread_tail_classifier_weight = (
            0.0 if warmup_active else cfg.delay_spread_tail_classifier_weight
        )
        effective_reflection_count_classifier_weight = (
            0.0
            if warmup_active
            else max(
                cfg.reflection_count_classifier_weight,
                cfg.interaction_count_classifier_weight,
            )
        )
        effective_reflection_count_regression_weight = (
            0.0
            if warmup_active
            else max(
                cfg.reflection_count_regression_weight,
                cfg.interaction_count_regression_weight,
            )
        )
        effective_reflection_path_count_regression_weight = (
            0.0 if warmup_active else cfg.reflection_path_count_regression_weight
        )
        effective_physics_relational_weight = (
            0.0 if warmup_active else cfg.physics_relational_weight
        )
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
        first_path_power_bin_target_histogram = None
        first_path_power_bin_prediction_histogram = None
        first_path_power_bin_loss_denominator = None
        first_path_power_bin_position_mae = None
        strong_k_bin_target_histogram = None
        strong_k_bin_prediction_histogram = None
        strong_k_bin_loss_denominator = None
        strong_k_position_mae = None
        delay_spread_bin_target_histogram = None
        delay_spread_bin_prediction_histogram = None
        delay_spread_bin_loss_denominator = None
        delay_spread_bin_position_mae = None
        delay_spread_raw_mae_ns = None
        delay_spread_normalized_mae_ns = None
        first_path_delay_bin_target_histogram = None
        first_path_delay_bin_prediction_histogram = None
        first_path_delay_bin_loss_denominator = None
        first_path_delay_bin_position_mae = None
        first_path_delay_raw_mae_ns = None
        first_path_delay_fused_raw_mae_ns = None
        first_path_delay_bin_consistency_violation_ns = None
        first_path_delay_bin_consistency_max_violation_ns = None
        estimated_pdp_tail_bin_loss_denominator = None
        estimated_pdp_tail_bin_accuracy = None
        estimated_pdp_tail_gate_fraction = None
        estimated_pdp_tail_target_fraction = None
        estimated_pdp_tail_argmax_fraction = None
        first_path_delay_tail_underestimate_mae_ns = None
        first_path_delay_tail_underestimate_mean_ns = None
        first_path_delay_tail_underestimate_fraction = None
        first_path_delay_tail_underestimate_count = None
        los_delay_raw_mae_ns = None
        los_angle_mae_deg = None
        los_angle_count = None
        first_path_angle_mae_deg = None
        first_path_angle_count = None
        first_path_angle_nlos_mae_deg = None
        first_path_angle_nlos_count = None
        direct_power_loss_count = None
        direct_power_loss_nlos_fraction = None
        los_first_path_power_base_mae_db = None
        nlos_first_path_power_base_mae_db = None
        nlos_first_path_power_enhanced_mae_db = None
        first_path_power_delta_abs_mean = None
        first_path_power_delta_saturation_fraction = None
        delay_spread_tail_accuracy = None
        delay_spread_tail_recall = None
        delay_spread_tail_false_positive = None
        delay_spread_tail_positive_fraction = None
        delay_spread_tail_prediction_fraction = None
        reflection_count_accuracy = None
        reflection_count_mae = None
        reflection_count_target_histogram = None
        reflection_count_prediction_histogram = None
        reflection_path_count_mae = None
        reflection_path_count_exact_accuracy = None
        reflection_path_count_target_histogram = None
        reflection_path_count_prediction_histogram = None
        if (
            effective_aux_regression_weight > 0
            or effective_strong_k_bin_classifier_weight > 0.0
            or effective_strong_k_position_weight > 0.0
            or effective_first_path_power_bin_classifier_weight > 0.0
            or effective_first_path_power_bin_position_weight > 0.0
            or effective_delay_spread_bin_classifier_weight > 0.0
            or effective_delay_spread_bin_position_weight > 0.0
            or effective_delay_spread_tail_classifier_weight > 0.0
            or cfg.direct_power_weight > 0.0
            or effective_delay_spread_weight > 0.0
            or effective_delay_spread_raw_weight > 0.0
            or effective_first_path_delay_weight > 0.0
            or effective_first_path_delay_raw_weight > 0.0
            or effective_first_path_delay_fused_raw_weight > 0.0
            or effective_first_path_delay_bin_classifier_weight > 0.0
            or effective_first_path_delay_bin_position_weight > 0.0
            or effective_first_path_delay_bin_consistency_weight > 0.0
            or effective_estimated_pdp_tail_bin_weight > 0.0
            or effective_first_path_delay_tail_underestimate_weight > 0.0
            or effective_los_delay_weight > 0.0
            or effective_los_delay_nonnegative_weight > 0.0
            or effective_los_angle_weight > 0.0
            or effective_first_path_angle_weight > 0.0
            or effective_first_path_angle_nlos_weight > 0.0
            or effective_reflection_count_classifier_weight > 0.0
            or effective_reflection_count_regression_weight > 0.0
            or effective_reflection_path_count_regression_weight > 0.0
            or effective_physics_relational_weight > 0.0
        ):
            power_context = None
            delay_context = None
            first_path_delay_context = None
            los_angle_context = None
            first_path_angle_context = None
            if hasattr(self.model, "encode_csi_delay_context"):
                delay_context = self.model.encode_csi_delay_context(
                    batch["tokens"],
                    batch["token_mask"],
                    subcarrier_spacing=batch.get("subcarrier_spacing"),
                )
            if (
                (
                    effective_first_path_delay_weight > 0.0
                    or effective_first_path_delay_raw_weight > 0.0
                    or effective_first_path_delay_fused_raw_weight > 0.0
                    or effective_first_path_delay_bin_classifier_weight > 0.0
                    or effective_first_path_delay_bin_position_weight > 0.0
                    or effective_first_path_delay_bin_consistency_weight > 0.0
                    or effective_estimated_pdp_tail_bin_weight > 0.0
                    or effective_first_path_delay_tail_underestimate_weight > 0.0
                    or effective_los_angle_weight > 0.0
                    or effective_first_path_angle_weight > 0.0
                    or effective_first_path_angle_nlos_weight > 0.0
                )
                and hasattr(self.model, "encode_first_path_delay_context")
            ):
                first_path_delay_context = self.model.encode_first_path_delay_context(
                    batch["tokens"],
                    batch["token_mask"],
                    beam_positions=batch.get("beam_positions"),
                    freq_bin=batch.get("freq_bin"),
                    bw_bin=batch.get("bw_bin"),
                    subcarrier_spacing=batch.get("subcarrier_spacing"),
                )
            if (
                (
                    effective_los_angle_weight > 0.0
                    or effective_first_path_angle_weight > 0.0
                    or effective_first_path_angle_nlos_weight > 0.0
                )
                and hasattr(self.model, "encode_los_angle_context")
            ):
                los_angle_context = self.model.encode_los_angle_context(
                    batch["tokens"],
                    batch["beam_positions"],
                    batch["token_mask"],
                    batch["freq_bin"],
                    batch["bw_bin"],
                    batch["subcarrier_spacing"],
                )
            if (
                (
                    effective_first_path_angle_weight > 0.0
                    or effective_first_path_angle_nlos_weight > 0.0
                )
                and hasattr(self.model, "encode_first_path_angle_context")
            ):
                first_path_angle_context = self.model.encode_first_path_angle_context(
                    batch["tokens"],
                    batch["beam_positions"],
                    batch["token_mask"],
                    subcarrier_spacing=batch.get("subcarrier_spacing"),
                )
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
                delay_context=delay_context,
                first_path_delay_context=first_path_delay_context,
                los_angle_context=los_angle_context,
                first_path_angle_context=first_path_angle_context,
            )
            physics_predictions = physics_outputs["final"]
            if cfg.first_path_power_gate_mode == "base":
                physics_predictions = self._apply_first_path_power_base(physics_outputs)
                physics_outputs["final"] = physics_predictions
            elif cfg.first_path_power_gate_mode != "none":
                raise ValueError(
                    "Unsupported first_path_power_gate_mode="
                    f"{cfg.first_path_power_gate_mode!r}. Choose from: none, base."
                )

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
        if (
            physics_outputs is not None
            and effective_strong_k_bin_classifier_weight > 0.0
        ):
            strong_k_bin_logits = physics_outputs["k_factor_strong_bin_logits"]
            strong_k_bin_targets = self._strong_k_bin_targets(
                batch["physics_raw_targets"][:, 3],
                batch["semantic_keys"],
            )
            strong_k_bin_mask = strong_k_bin_targets >= 0
            if bool(strong_k_bin_mask.any()):
                valid_logits = strong_k_bin_logits[strong_k_bin_mask]
                valid_targets = strong_k_bin_targets[strong_k_bin_mask]
                per_sample_classifier_loss = torch.nn.functional.cross_entropy(
                    valid_logits,
                    valid_targets,
                    reduction="none",
                )
                class_weight = self._strong_k_bin_class_weight(valid_logits.dtype, cfg)
                if class_weight is None:
                    sample_weight = torch.ones_like(per_sample_classifier_loss)
                else:
                    sample_weight = class_weight[valid_targets]
                strong_k_bin_loss_denominator = sample_weight.sum()
                losses["loss_strong_k_bin_classifier"] = (
                    (per_sample_classifier_loss * sample_weight).sum()
                    / strong_k_bin_loss_denominator.clamp(min=1.0)
                )
                strong_k_bin_predictions = valid_logits.argmax(dim=1)
                losses["accuracy_strong_k_bin_classifier"] = (
                    (strong_k_bin_predictions == valid_targets).float().mean()
                )
                strong_k_bin_target_histogram = self._histogram(
                    valid_targets,
                    len(self.STRONG_K_BIN_LABELS),
                )
                strong_k_bin_prediction_histogram = self._histogram(
                    strong_k_bin_predictions,
                    len(self.STRONG_K_BIN_LABELS),
                )
        if (
            physics_outputs is not None
            and effective_strong_k_position_weight > 0.0
        ):
            strong_k_position_predictions = physics_outputs["k_factor_strong_position"]
            strong_k_position_targets, strong_k_position_mask = self._strong_k_position_targets(
                batch["physics_raw_targets"][:, 3],
                batch["semantic_keys"],
            )
            if bool(strong_k_position_mask.any()):
                valid_position_predictions = strong_k_position_predictions[strong_k_position_mask]
                valid_position_targets = strong_k_position_targets[strong_k_position_mask]
                position_errors = torch.nn.functional.smooth_l1_loss(
                    valid_position_predictions,
                    valid_position_targets,
                    reduction="none",
                )
                losses["loss_strong_k_position"] = position_errors.mean()
                strong_k_position_mae = (
                    valid_position_predictions - valid_position_targets
                ).abs().mean()
        if (
            physics_outputs is not None
            and effective_delay_spread_bin_classifier_weight > 0.0
        ):
            delay_spread_bin_logits = physics_outputs["delay_spread_bin_logits"]
            delay_spread_bin_targets = self._delay_spread_bin_targets(
                batch["physics_raw_targets"][:, 1]
            )
            delay_spread_bin_mask = (
                (delay_spread_bin_targets >= 0)
                & batch["physics_target_mask"][:, 1].bool()
            )
            if bool(delay_spread_bin_mask.any()):
                valid_logits = delay_spread_bin_logits[delay_spread_bin_mask]
                valid_targets = delay_spread_bin_targets[delay_spread_bin_mask]
                per_sample_classifier_loss = torch.nn.functional.cross_entropy(
                    valid_logits,
                    valid_targets,
                    reduction="none",
                )
                class_weight = self._delay_spread_bin_class_weight(valid_logits.dtype, cfg)
                if class_weight is None:
                    sample_weight = torch.ones_like(per_sample_classifier_loss)
                else:
                    sample_weight = class_weight[valid_targets]
                delay_spread_bin_loss_denominator = sample_weight.sum()
                losses["loss_delay_spread_bin_classifier"] = (
                    (per_sample_classifier_loss * sample_weight).sum()
                    / delay_spread_bin_loss_denominator.clamp(min=1.0)
                )
                delay_spread_bin_predictions = valid_logits.argmax(dim=1)
                losses["accuracy_delay_spread_bin_classifier"] = (
                    (delay_spread_bin_predictions == valid_targets).float().mean()
                )
                delay_spread_bin_target_histogram = self._histogram(
                    valid_targets,
                    len(self.DELAY_SPREAD_BIN_LABELS),
                )
                delay_spread_bin_prediction_histogram = self._histogram(
                    delay_spread_bin_predictions,
                    len(self.DELAY_SPREAD_BIN_LABELS),
                )
                profile_direct_delay_spread = physics_outputs.get("profile_direct_delay_spread")
                if profile_direct_delay_spread is not None and cfg.delay_spread_teacher_weight > 0.0:
                    delay_idx = 1
                    scale = PHYSICS_TARGET_SCALES[delay_idx].to(
                        device=profile_direct_delay_spread.device,
                        dtype=profile_direct_delay_spread.dtype,
                    )
                    offset = PHYSICS_TARGET_OFFSETS[delay_idx].to(
                        device=profile_direct_delay_spread.device,
                        dtype=profile_direct_delay_spread.dtype,
                    )
                    teacher_raw = profile_direct_delay_spread.detach() * scale + offset
                    teacher_targets = self._delay_spread_bin_targets(teacher_raw)
                    teacher_mask = (
                        (teacher_targets >= 0)
                        & batch["physics_target_mask"][:, delay_idx].bool()
                    )
                    if bool(teacher_mask.any()):
                        teacher_logits = delay_spread_bin_logits[teacher_mask]
                        teacher_targets = teacher_targets[teacher_mask]
                        teacher_loss = torch.nn.functional.cross_entropy(
                            teacher_logits,
                            teacher_targets,
                            reduction="none",
                        )
                        teacher_class_weight = self._delay_spread_bin_class_weight(
                            teacher_logits.dtype,
                            cfg,
                        )
                        if teacher_class_weight is None:
                            teacher_weight = torch.ones_like(teacher_loss)
                        else:
                            teacher_weight = teacher_class_weight[teacher_targets]
                        teacher_loss = (
                            (teacher_loss * teacher_weight).sum()
                            / teacher_weight.sum().clamp(min=1.0)
                        )
                        losses["loss_delay_spread_bin_teacher"] = teacher_loss
                        losses["loss_delay_spread_bin_classifier"] = (
                            losses["loss_delay_spread_bin_classifier"]
                            + float(cfg.delay_spread_teacher_weight) * teacher_loss
                        )
        if (
            physics_outputs is not None
            and effective_delay_spread_bin_position_weight > 0.0
        ):
            delay_spread_position_predictions = physics_outputs["delay_spread_bin_position"]
            delay_spread_position_targets, delay_spread_position_mask = (
                self._delay_spread_bin_position_targets(batch["physics_raw_targets"][:, 1])
            )
            delay_spread_position_mask = (
                delay_spread_position_mask
                & batch["physics_target_mask"][:, 1].bool()
            )
            if bool(delay_spread_position_mask.any()):
                valid_position_predictions = delay_spread_position_predictions[
                    delay_spread_position_mask
                ]
                valid_position_targets = delay_spread_position_targets[
                    delay_spread_position_mask
                ]
                position_errors = torch.nn.functional.smooth_l1_loss(
                    valid_position_predictions,
                    valid_position_targets,
                    reduction="none",
                )
                position_weights = self._delay_spread_sample_weights(
                    batch["physics_raw_targets"][:, 1],
                    cfg,
                )[delay_spread_position_mask]
                delay_spread_position_denominator = position_weights.sum()
                losses["loss_delay_spread_bin_position"] = (
                    (position_errors * position_weights.to(dtype=position_errors.dtype)).sum()
                    / delay_spread_position_denominator.clamp(min=1.0).to(dtype=position_errors.dtype)
                )
                delay_spread_bin_position_mae = (
                    valid_position_predictions - valid_position_targets
                ).abs().mean()
                profile_direct_delay_spread = physics_outputs.get("profile_direct_delay_spread")
                if profile_direct_delay_spread is not None and cfg.delay_spread_teacher_weight > 0.0:
                    delay_idx = 1
                    scale = PHYSICS_TARGET_SCALES[delay_idx].to(
                        device=profile_direct_delay_spread.device,
                        dtype=profile_direct_delay_spread.dtype,
                    )
                    offset = PHYSICS_TARGET_OFFSETS[delay_idx].to(
                        device=profile_direct_delay_spread.device,
                        dtype=profile_direct_delay_spread.dtype,
                    )
                    teacher_raw = profile_direct_delay_spread.detach() * scale + offset
                    teacher_targets, teacher_mask = self._delay_spread_bin_position_targets(
                        teacher_raw
                    )
                    teacher_mask = (
                        teacher_mask
                        & batch["physics_target_mask"][:, delay_idx].bool()
                    )
                    if bool(teacher_mask.any()):
                        teacher_errors = torch.nn.functional.smooth_l1_loss(
                            delay_spread_position_predictions[teacher_mask],
                            teacher_targets[teacher_mask],
                            reduction="none",
                        )
                        teacher_weights = self._delay_spread_sample_weights(
                            teacher_raw,
                            cfg,
                        )[teacher_mask]
                        teacher_loss = (
                            (teacher_errors * teacher_weights.to(dtype=teacher_errors.dtype)).sum()
                            / teacher_weights.sum().clamp(min=1.0).to(dtype=teacher_errors.dtype)
                        )
                        losses["loss_delay_spread_bin_position_teacher"] = teacher_loss
                        losses["loss_delay_spread_bin_position"] = (
                            losses["loss_delay_spread_bin_position"]
                            + float(cfg.delay_spread_teacher_weight) * teacher_loss
                        )
        if (
            physics_outputs is not None
            and effective_delay_spread_tail_classifier_weight > 0.0
        ):
            delay_idx = 1
            tail_logits = physics_outputs["delay_spread_tail_logits"]
            tail_mask = batch["physics_target_mask"][:, delay_idx].bool()
            if bool(tail_mask.any()):
                thresholds = torch.tensor(
                    self.DELAY_SPREAD_TAIL_THRESHOLDS_NS,
                    device=self.device,
                    dtype=batch["physics_raw_targets"].dtype,
                )
                tail_targets = (
                    batch["physics_raw_targets"][:, delay_idx].unsqueeze(1)
                    >= thresholds.unsqueeze(0)
                ).to(dtype=tail_logits.dtype)
                valid_logits = tail_logits[tail_mask]
                valid_targets = tail_targets[tail_mask]
                positives = valid_targets.sum(dim=0)
                negatives = valid_targets.shape[0] - positives
                pos_weight = (negatives / positives.clamp(min=1.0)).clamp(1.0, 12.0)
                tail_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    valid_logits,
                    valid_targets,
                    pos_weight=pos_weight.to(dtype=valid_logits.dtype),
                    reduction="mean",
                )
                losses["loss_delay_spread_tail_classifier"] = tail_loss
                tail_probabilities = torch.sigmoid(valid_logits)
                tail_predictions = tail_probabilities >= 0.5
                tail_targets_bool = valid_targets.bool()
                delay_spread_tail_accuracy = (
                    tail_predictions == tail_targets_bool
                ).float().mean(dim=0)
                delay_spread_tail_recall = torch.where(
                    tail_targets_bool.any(dim=0),
                    (tail_predictions & tail_targets_bool).float().sum(dim=0)
                    / tail_targets_bool.float().sum(dim=0).clamp(min=1.0),
                    torch.full_like(delay_spread_tail_accuracy, float("nan")),
                )
                negative_mask = ~tail_targets_bool
                delay_spread_tail_false_positive = torch.where(
                    negative_mask.any(dim=0),
                    (tail_predictions & negative_mask).float().sum(dim=0)
                    / negative_mask.float().sum(dim=0).clamp(min=1.0),
                    torch.full_like(delay_spread_tail_accuracy, float("nan")),
                )
                delay_spread_tail_positive_fraction = valid_targets.mean(dim=0)
                delay_spread_tail_prediction_fraction = tail_predictions.float().mean(dim=0)
        if (
            physics_outputs is not None
            and effective_reflection_count_classifier_weight > 0.0
        ):
            target_idx = PHYSICS_TARGET_NAMES.index("reflection_count")
            logits = physics_outputs["reflection_count_logits"]
            targets = self._interaction_count_bin_targets(
                batch["physics_raw_targets"][:, target_idx],
                self.REFLECTION_COUNT_BINS,
            )
            mask = (
                (targets >= 0)
                & batch["physics_target_mask"][:, target_idx].bool()
            )
            if bool(mask.any()):
                valid_logits = logits[mask]
                valid_targets = targets[mask]
                sample_weight = self._reflection_count_sample_weights(
                    batch["semantic_keys"],
                    device=valid_logits.device,
                    dtype=valid_logits.dtype,
                    cfg=cfg,
                )[mask]
                if cfg.interaction_count_soft_labels:
                    soft_targets = self._interaction_count_soft_targets(
                        valid_targets,
                        len(self.REFLECTION_COUNT_BIN_LABELS),
                    ).to(dtype=valid_logits.dtype)
                    per_sample_classifier_loss = -(
                        soft_targets
                        * torch.nn.functional.log_softmax(valid_logits, dim=1)
                    ).sum(dim=1)
                else:
                    per_sample_classifier_loss = torch.nn.functional.cross_entropy(
                        valid_logits,
                        valid_targets,
                        reduction="none",
                    )
                loss = (
                    (per_sample_classifier_loss * sample_weight).sum()
                    / sample_weight.sum().clamp(min=1.0)
                )
                predictions = valid_logits.argmax(dim=1)
                reflection_count_accuracy = (predictions == valid_targets).float().mean()
                reflection_count_target_histogram = self._histogram(
                    valid_targets,
                    len(self.REFLECTION_COUNT_BIN_LABELS),
                )
                reflection_count_prediction_histogram = self._histogram(
                    predictions,
                    len(self.REFLECTION_COUNT_BIN_LABELS),
                )
                losses["loss_reflection_count_classifier"] = loss
                losses["accuracy_reflection_count_classifier"] = reflection_count_accuracy
                losses["loss_interaction_count_classifier"] = loss
        if (
            physics_outputs is not None
            and effective_reflection_count_regression_weight > 0.0
        ):
            target_idx = PHYSICS_TARGET_NAMES.index("reflection_count")
            prediction = physics_outputs["reflection_count_prediction"]
            target = batch["physics_targets"][:, target_idx]
            mask = batch["physics_target_mask"][:, target_idx].bool()
            if bool(mask.any()):
                sample_weight = self._reflection_count_sample_weights(
                    batch["semantic_keys"],
                    device=prediction.device,
                    dtype=prediction.dtype,
                    cfg=cfg,
                )[mask]
                errors = torch.nn.functional.smooth_l1_loss(
                    prediction[mask],
                    target[mask],
                    reduction="none",
                )
                loss = (
                    (errors * sample_weight).sum()
                    / sample_weight.sum().clamp(min=1.0)
                )
                raw_scale = PHYSICS_TARGET_SCALES[target_idx].to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                raw_offset = PHYSICS_TARGET_OFFSETS[target_idx].to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                raw_prediction = prediction * raw_scale + raw_offset
                raw_target = batch["physics_raw_targets"][:, target_idx].to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                reflection_count_mae = (
                    raw_prediction[mask] - raw_target[mask]
                ).abs().mean()
                losses["loss_reflection_count_regression"] = loss
                losses["loss_interaction_count_regression"] = loss
        if (
            physics_outputs is not None
            and effective_reflection_path_count_regression_weight > 0.0
        ):
            prediction = physics_outputs["reflection_path_count_prediction"]
            target = batch["reflection_path_count_target"]
            mask = batch["reflection_path_count_target_mask"].bool()
            if bool(mask.any()):
                loss = torch.nn.functional.smooth_l1_loss(
                    prediction[mask],
                    target[mask],
                    reduction="mean",
                )
                raw_prediction = prediction * 10.0
                raw_target = batch["reflection_path_count_raw_target"].to(
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                valid_raw_prediction = raw_prediction[mask]
                valid_raw_target = raw_target[mask]
                rounded_prediction = valid_raw_prediction.round().clamp(min=0).long()
                rounded_target = valid_raw_target.round().clamp(min=0).long()
                reflection_path_count_mae = (
                    valid_raw_prediction - valid_raw_target
                ).abs().mean()
                reflection_path_count_exact_accuracy = (
                    rounded_prediction == rounded_target
                ).float().mean()
                max_bin = int(
                    max(
                        rounded_prediction.max().item(),
                        rounded_target.max().item(),
                    )
                ) + 1
                reflection_path_count_target_histogram = self._histogram(
                    rounded_target,
                    max_bin,
                )
                reflection_path_count_prediction_histogram = self._histogram(
                    rounded_prediction,
                    max_bin,
                )
                losses["loss_reflection_path_count_regression"] = loss
        if (
            physics_outputs is not None
            and effective_first_path_power_bin_classifier_weight > 0.0
        ):
            first_path_power_bin_logits = physics_outputs["first_path_power_bin_logits"]
            first_path_power_bin_targets = self._first_path_power_bin_targets(
                batch["physics_raw_targets"][:, 5]
            )
            first_path_power_bin_mask = first_path_power_bin_targets >= 0
            if bool(first_path_power_bin_mask.any()):
                valid_logits = first_path_power_bin_logits[first_path_power_bin_mask]
                valid_targets = first_path_power_bin_targets[first_path_power_bin_mask]
                per_sample_classifier_loss = torch.nn.functional.cross_entropy(
                    valid_logits,
                    valid_targets,
                    reduction="none",
                )
                class_weight = self._first_path_power_bin_class_weight(valid_logits.dtype, cfg)
                if class_weight is None:
                    losses["loss_first_path_power_bin_classifier"] = (
                        per_sample_classifier_loss.mean()
                    )
                    first_path_power_bin_loss_denominator = torch.tensor(
                        valid_targets.numel(),
                        device=self.device,
                        dtype=valid_logits.dtype,
                    )
                else:
                    sample_weight = class_weight[valid_targets]
                    first_path_power_bin_loss_denominator = sample_weight.sum()
                    losses["loss_first_path_power_bin_classifier"] = (
                        (per_sample_classifier_loss * sample_weight).sum()
                        / first_path_power_bin_loss_denominator.clamp(min=1.0)
                    )
                first_path_power_bin_predictions = valid_logits.argmax(dim=1)
                losses["accuracy_first_path_power_bin_classifier"] = (
                    (first_path_power_bin_predictions == valid_targets).float().mean()
                )
                first_path_power_bin_target_histogram = self._histogram(
                    valid_targets,
                    len(self.FIRST_PATH_POWER_BIN_LABELS),
                )
                first_path_power_bin_prediction_histogram = self._histogram(
                    first_path_power_bin_predictions,
                    len(self.FIRST_PATH_POWER_BIN_LABELS),
                )
        if (
            physics_outputs is not None
            and effective_first_path_power_bin_position_weight > 0.0
        ):
            position_predictions = physics_outputs["first_path_power_bin_position"]
            position_targets, position_mask = self._first_path_power_bin_position_targets(
                batch["physics_raw_targets"][:, 5]
            )
            if bool(position_mask.any()):
                valid_position_predictions = position_predictions[position_mask]
                valid_position_targets = position_targets[position_mask]
                position_errors = torch.nn.functional.smooth_l1_loss(
                    valid_position_predictions,
                    valid_position_targets,
                    reduction="none",
                )
                losses["loss_first_path_power_bin_position"] = position_errors.mean()
                first_path_power_bin_position_mae = (
                    valid_position_predictions - valid_position_targets
                ).abs().mean()
        if physics_predictions is not None and effective_aux_regression_weight > 0:
            regression_predictions = physics_predictions
            regression_targets = batch["physics_targets"]
            regression_mask = batch["physics_target_mask"]
            regression_weights = torch.ones_like(regression_targets)
            if cfg.aux_regression_indices:
                indices = torch.tensor(cfg.aux_regression_indices, device=self.device, dtype=torch.long)
                regression_predictions = regression_predictions.index_select(dim=1, index=indices)
                regression_targets = regression_targets.index_select(dim=1, index=indices)
                regression_mask = regression_mask.index_select(dim=1, index=indices)
                regression_weights = torch.ones_like(regression_targets)
                if 3 in cfg.aux_regression_indices:
                    k_factor_position = cfg.aux_regression_indices.index(3)
                    regression_weights[:, k_factor_position] = self._k_factor_sample_weights(
                        batch["physics_raw_targets"][:, 3],
                        batch["semantic_keys"],
                        cfg,
                    )
                if 5 in cfg.aux_regression_indices:
                    first_path_position = cfg.aux_regression_indices.index(5)
                    regression_weights[:, first_path_position] = self._first_path_power_supervision_weights(
                        batch["physics_raw_targets"][:, 5],
                        batch["semantic_keys"],
                        cfg,
                    )
            else:
                regression_weights[:, 3] = self._k_factor_sample_weights(
                    batch["physics_raw_targets"][:, 3],
                    batch["semantic_keys"],
                    cfg,
                )
                regression_weights[:, 5] = self._first_path_power_supervision_weights(
                    batch["physics_raw_targets"][:, 5],
                    batch["semantic_keys"],
                    cfg,
                )
            regression_errors = torch.nn.functional.smooth_l1_loss(
                regression_predictions,
                regression_targets,
                reduction="none",
            )
            weighted_regression_mask = (
                regression_mask.to(dtype=regression_errors.dtype)
                * regression_weights.to(dtype=regression_errors.dtype)
            )
            losses["loss_aux_regression"] = (
                (regression_errors * weighted_regression_mask).sum()
                / weighted_regression_mask.sum().clamp(min=1).to(dtype=regression_errors.dtype)
            )
        if (
            physics_outputs is not None
            and bool(getattr(self.model, "use_power_branch", False))
        ):
            first_path_power_idx = 5
            if cfg.direct_power_weight > 0.0:
                if cfg.nlos_enhanced_power_loss:
                    direct_power_prediction = physics_outputs["enhanced_first_path_power"]
                    direct_power_sample_mask = torch.tensor(
                        [
                            getattr(key, "los_status", None) != "los"
                            for key in batch["semantic_keys"]
                        ],
                        device=self.device,
                        dtype=torch.bool,
                    )
                else:
                    direct_power_prediction = physics_predictions[:, first_path_power_idx]
                    direct_power_sample_mask = torch.ones(
                        direct_power_prediction.shape,
                        device=self.device,
                        dtype=torch.bool,
                    )
                direct_power_target = batch["physics_targets"][:, first_path_power_idx]
                direct_power_mask = (
                    batch["physics_target_mask"][:, first_path_power_idx]
                    & direct_power_sample_mask
                )
                direct_power_errors = torch.nn.functional.smooth_l1_loss(
                    direct_power_prediction,
                    direct_power_target,
                    reduction="none",
                )
                direct_power_weights = self._first_path_power_supervision_weights(
                    batch["physics_raw_targets"][:, first_path_power_idx],
                    batch["semantic_keys"],
                    cfg,
                )
                weighted_direct_power_mask = (
                    direct_power_mask.to(dtype=direct_power_errors.dtype)
                    * direct_power_weights.to(dtype=direct_power_errors.dtype)
                )
                direct_power_errors = direct_power_errors * weighted_direct_power_mask
                losses["loss_direct_power"] = (
                    direct_power_errors.sum()
                    / weighted_direct_power_mask.sum().clamp(min=1).to(dtype=direct_power_errors.dtype)
                )
                direct_power_loss_count = direct_power_mask.sum()
                direct_power_loss_nlos_fraction = direct_power_sample_mask.float().mean()
            with torch.no_grad():
                target_scale = PHYSICS_TARGET_SCALES[first_path_power_idx].to(
                    device=self.device,
                    dtype=physics_predictions.dtype,
                )
                target_offset = PHYSICS_TARGET_OFFSETS[first_path_power_idx].to(
                    device=self.device,
                    dtype=physics_predictions.dtype,
                )
                base_raw = (
                    physics_outputs["base"][:, first_path_power_idx] * target_scale
                    + target_offset
                )
                enhanced_raw = (
                    physics_outputs["enhanced_first_path_power"] * target_scale
                    + target_offset
                )
                target_raw = batch["physics_raw_targets"][:, first_path_power_idx].to(
                    device=self.device,
                    dtype=physics_predictions.dtype,
                )
                valid_power_mask = (
                    batch["physics_target_mask"][:, first_path_power_idx].bool()
                    & torch.isfinite(target_raw)
                )
                los_sample_mask = torch.tensor(
                    [
                        getattr(key, "los_status", None) == "los"
                        for key in batch["semantic_keys"]
                    ],
                    device=self.device,
                    dtype=torch.bool,
                )
                los_power_mask = valid_power_mask & los_sample_mask
                nlos_power_mask = valid_power_mask & ~los_sample_mask
                if bool(los_power_mask.any()):
                    los_first_path_power_base_mae_db = (
                        base_raw[los_power_mask] - target_raw[los_power_mask]
                    ).abs().mean()
                if bool(nlos_power_mask.any()):
                    nlos_first_path_power_base_mae_db = (
                        base_raw[nlos_power_mask] - target_raw[nlos_power_mask]
                    ).abs().mean()
                    nlos_first_path_power_enhanced_mae_db = (
                        enhanced_raw[nlos_power_mask] - target_raw[nlos_power_mask]
                    ).abs().mean()
                enhanced_delta = physics_outputs.get("enhanced_delta")
                if enhanced_delta is not None:
                    first_path_power_delta_abs_mean = enhanced_delta.detach().abs().mean()
                    delta_limit = float(getattr(self.model, "first_path_power_delta_limit", 0.0))
                    if delta_limit > 0.0:
                        first_path_power_delta_saturation_fraction = (
                            enhanced_delta.detach().abs() >= 0.95 * delta_limit
                        ).float().mean()
        if (
            physics_outputs is not None
            and (
                effective_delay_spread_weight > 0.0
                or effective_delay_spread_raw_weight > 0.0
            )
        ):
            delay_spread_idx = 1
            csi_delay_spread = physics_outputs["csi_delay_spread"]
            delay_spread_context = physics_outputs.get("delay_spread_context")
            enhanced_delay_spread = physics_outputs["enhanced_delay_spread"]
            profile_delay_spread = physics_outputs.get("profile_delay_spread")
            profile_direct_delay_spread = physics_outputs.get("profile_direct_delay_spread")
            delay_spread_target = batch["physics_targets"][:, delay_spread_idx]
            delay_spread_mask = batch["physics_target_mask"][:, delay_spread_idx]
            delay_spread_weights = self._delay_spread_sample_weights(
                batch["physics_raw_targets"][:, delay_spread_idx],
                cfg,
            )
            csi_delay_spread_errors = torch.nn.functional.smooth_l1_loss(
                csi_delay_spread,
                delay_spread_target,
                reduction="none",
            )
            weighted_delay_spread_mask = (
                delay_spread_mask.to(dtype=csi_delay_spread_errors.dtype)
                * delay_spread_weights.to(dtype=csi_delay_spread_errors.dtype)
            )
            csi_delay_spread_loss = (
                (csi_delay_spread_errors * weighted_delay_spread_mask).sum()
                / weighted_delay_spread_mask.sum().clamp(min=1).to(dtype=csi_delay_spread_errors.dtype)
            )
            losses["loss_delay_spread_csi"] = csi_delay_spread_loss
            enhanced_delay_spread_errors = torch.nn.functional.smooth_l1_loss(
                enhanced_delay_spread,
                delay_spread_target,
                reduction="none",
            )
            enhanced_delay_spread_loss = (
                (enhanced_delay_spread_errors * weighted_delay_spread_mask).sum()
                / weighted_delay_spread_mask.sum().clamp(min=1).to(dtype=enhanced_delay_spread_errors.dtype)
            )
            losses["loss_delay_spread_enhanced"] = enhanced_delay_spread_loss
            losses["loss_delay_spread"] = torch.zeros((), device=self.device)
            if delay_spread_context is not None:
                context_delay_spread_errors = torch.nn.functional.smooth_l1_loss(
                    delay_spread_context,
                    delay_spread_target,
                    reduction="none",
                )
                context_delay_spread_loss = (
                    (context_delay_spread_errors * weighted_delay_spread_mask).sum()
                    / weighted_delay_spread_mask.sum().clamp(min=1).to(dtype=context_delay_spread_errors.dtype)
                )
                losses["loss_delay_spread_context"] = context_delay_spread_loss
                losses["loss_delay_spread"] = context_delay_spread_loss
                target_scale = PHYSICS_TARGET_SCALES[delay_spread_idx].to(
                    device=delay_spread_context.device,
                    dtype=delay_spread_context.dtype,
                )
                target_offset = PHYSICS_TARGET_OFFSETS[delay_spread_idx].to(
                    device=delay_spread_context.device,
                    dtype=delay_spread_context.dtype,
                )
                raw_prediction = delay_spread_context * target_scale + target_offset
                raw_target = batch["physics_raw_targets"][:, delay_spread_idx].to(
                    device=raw_prediction.device,
                    dtype=raw_prediction.dtype,
                )
                raw_mask = delay_spread_mask.bool() & torch.isfinite(raw_target)
                if bool(raw_mask.any()):
                    raw_abs_error = (raw_prediction - raw_target).abs()
                    beta = max(float(cfg.delay_spread_raw_beta_ns), 1e-6)
                    raw_errors = torch.where(
                        raw_abs_error < beta,
                        0.5 * raw_abs_error.square() / beta,
                        raw_abs_error - 0.5 * beta,
                    )
                    raw_weights = delay_spread_weights[raw_mask]
                    losses["loss_delay_spread_raw"] = (
                        (raw_errors[raw_mask] * raw_weights.to(dtype=raw_errors.dtype)).sum()
                        / raw_weights.sum().clamp(min=1.0).to(dtype=raw_errors.dtype)
                    )
                    delay_spread_raw_mae_ns = raw_abs_error[raw_mask].mean()
                    delay_spread_normalized_mae_ns = (
                        (delay_spread_context - delay_spread_target).abs()[raw_mask].mean()
                        * target_scale
                    )
            if profile_delay_spread is not None:
                profile_delay_spread_errors = torch.nn.functional.smooth_l1_loss(
                    profile_delay_spread,
                    delay_spread_target,
                    reduction="none",
                )
                profile_delay_spread_loss = (
                    (profile_delay_spread_errors * weighted_delay_spread_mask).sum()
                    / weighted_delay_spread_mask.sum().clamp(min=1).to(dtype=profile_delay_spread_errors.dtype)
                )
                losses["loss_delay_spread_profile"] = profile_delay_spread_loss
            if profile_direct_delay_spread is not None:
                profile_direct_delay_spread_errors = torch.nn.functional.smooth_l1_loss(
                    profile_direct_delay_spread,
                    delay_spread_target,
                    reduction="none",
                )
                profile_direct_delay_spread_loss = (
                    (profile_direct_delay_spread_errors * weighted_delay_spread_mask).sum()
                    / weighted_delay_spread_mask.sum().clamp(min=1).to(dtype=profile_direct_delay_spread_errors.dtype)
                )
                losses["loss_delay_spread_direct"] = profile_direct_delay_spread_loss
                teacher_errors = torch.nn.functional.smooth_l1_loss(
                    csi_delay_spread,
                    profile_direct_delay_spread.detach(),
                    reduction="none",
                )
                teacher_loss = (
                    (teacher_errors * weighted_delay_spread_mask).sum()
                    / weighted_delay_spread_mask.sum().clamp(min=1).to(dtype=teacher_errors.dtype)
                )
                losses["loss_delay_spread_teacher"] = teacher_loss
                if delay_spread_context is not None:
                    context_teacher_errors = torch.nn.functional.smooth_l1_loss(
                        delay_spread_context,
                        profile_direct_delay_spread.detach(),
                        reduction="none",
                    )
                    context_teacher_loss = (
                        (context_teacher_errors * weighted_delay_spread_mask).sum()
                        / weighted_delay_spread_mask.sum().clamp(min=1).to(dtype=context_teacher_errors.dtype)
                    )
                    losses["loss_delay_spread_context_teacher"] = context_teacher_loss
        if (
            physics_outputs is not None
            and effective_first_path_delay_weight > 0.0
        ):
            first_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
            first_delay_prediction = physics_outputs["first_path_delay_context"]
            first_delay_target = batch["physics_targets"][:, first_delay_idx]
            first_delay_mask = batch["physics_target_mask"][:, first_delay_idx]
            first_delay_errors = torch.nn.functional.smooth_l1_loss(
                first_delay_prediction,
                first_delay_target,
                reduction="none",
            )
            first_delay_weight = first_delay_mask.to(dtype=first_delay_errors.dtype)
            losses["loss_first_path_delay"] = (
                (first_delay_errors * first_delay_weight).sum()
                / first_delay_weight.sum().clamp(min=1).to(dtype=first_delay_errors.dtype)
            )
        if (
            physics_outputs is not None
            and effective_first_path_delay_raw_weight > 0.0
        ):
            first_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
            first_delay_prediction = physics_outputs["first_path_delay_context"]
            target_scale = PHYSICS_TARGET_SCALES[first_delay_idx].to(
                device=first_delay_prediction.device,
                dtype=first_delay_prediction.dtype,
            )
            target_offset = PHYSICS_TARGET_OFFSETS[first_delay_idx].to(
                device=first_delay_prediction.device,
                dtype=first_delay_prediction.dtype,
            )
            raw_prediction = first_delay_prediction * target_scale + target_offset
            raw_target = batch["physics_raw_targets"][:, first_delay_idx].to(
                device=raw_prediction.device,
                dtype=raw_prediction.dtype,
            )
            raw_mask = (
                batch["physics_target_mask"][:, first_delay_idx].bool()
                & torch.isfinite(raw_target)
            )
            if bool(raw_mask.any()):
                raw_abs_error = (raw_prediction - raw_target).abs()
                beta = max(float(cfg.first_path_delay_raw_beta_ns), 1e-6)
                raw_errors = torch.where(
                    raw_abs_error < beta,
                    0.5 * raw_abs_error.square() / beta,
                    raw_abs_error - 0.5 * beta,
                )
                raw_weights = self._first_path_delay_sample_weights(
                    batch["physics_raw_targets"][:, first_delay_idx],
                    cfg,
                )[raw_mask]
                losses["loss_first_path_delay_raw"] = (
                    (raw_errors[raw_mask] * raw_weights.to(dtype=raw_errors.dtype)).sum()
                    / raw_weights.sum().clamp(min=1.0).to(dtype=raw_errors.dtype)
                )
                first_path_delay_raw_mae_ns = raw_abs_error[raw_mask].mean()
        if (
            physics_outputs is not None
            and effective_first_path_delay_fused_raw_weight > 0.0
        ):
            first_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
            raw_prediction = physics_outputs["first_path_delay_bin_soft_fused_raw"]
            raw_target = batch["physics_raw_targets"][:, first_delay_idx].to(
                device=raw_prediction.device,
                dtype=raw_prediction.dtype,
            )
            raw_mask = (
                batch["physics_target_mask"][:, first_delay_idx].bool()
                & torch.isfinite(raw_target)
            )
            if bool(raw_mask.any()):
                raw_abs_error = (raw_prediction - raw_target).abs()
                beta = max(float(cfg.first_path_delay_raw_beta_ns), 1e-6)
                raw_errors = torch.where(
                    raw_abs_error < beta,
                    0.5 * raw_abs_error.square() / beta,
                    raw_abs_error - 0.5 * beta,
                )
                raw_weights = self._first_path_delay_sample_weights(
                    batch["physics_raw_targets"][:, first_delay_idx],
                    cfg,
                )[raw_mask]
                losses["loss_first_path_delay_fused_raw"] = (
                    (raw_errors[raw_mask] * raw_weights.to(dtype=raw_errors.dtype)).sum()
                    / raw_weights.sum().clamp(min=1.0).to(dtype=raw_errors.dtype)
                )
                first_path_delay_fused_raw_mae_ns = raw_abs_error[raw_mask].mean()
        if (
            physics_outputs is not None
            and effective_first_path_delay_bin_consistency_weight > 0.0
        ):
            first_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
            first_delay_prediction = physics_outputs["first_path_delay_context"]
            target_scale = PHYSICS_TARGET_SCALES[first_delay_idx].to(
                device=first_delay_prediction.device,
                dtype=first_delay_prediction.dtype,
            )
            target_offset = PHYSICS_TARGET_OFFSETS[first_delay_idx].to(
                device=first_delay_prediction.device,
                dtype=first_delay_prediction.dtype,
            )
            raw_prediction = first_delay_prediction * target_scale + target_offset
            lower_targets, upper_targets, finite_upper_mask, consistency_mask = (
                self._first_path_delay_bin_bounds(
                    batch["physics_raw_targets"][:, first_delay_idx]
                )
            )
            consistency_mask = (
                consistency_mask
                & batch["physics_target_mask"][:, first_delay_idx].bool()
            )
            if bool(consistency_mask.any()):
                lower_violation = torch.relu(lower_targets - raw_prediction)
                upper_violation = torch.where(
                    finite_upper_mask,
                    torch.relu(raw_prediction - upper_targets),
                    torch.zeros_like(raw_prediction),
                )
                violation_ns = lower_violation + upper_violation
                valid_violation_ns = violation_ns[consistency_mask]
                consistency_weights = self._first_path_delay_sample_weights(
                    batch["physics_raw_targets"][:, first_delay_idx],
                    cfg,
                )[consistency_mask]
                normalized_violation = valid_violation_ns / target_scale.clamp(min=1e-6)
                losses["loss_first_path_delay_bin_consistency"] = (
                    (normalized_violation * consistency_weights.to(dtype=normalized_violation.dtype)).sum()
                    / consistency_weights.sum().clamp(min=1.0).to(dtype=normalized_violation.dtype)
                )
                first_path_delay_bin_consistency_violation_ns = valid_violation_ns.mean()
                first_path_delay_bin_consistency_max_violation_ns = valid_violation_ns.max()
        if (
            physics_outputs is not None
            and effective_first_path_delay_bin_classifier_weight > 0.0
        ):
            first_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
            first_path_delay_bin_logits = physics_outputs["first_path_delay_bin_logits"]
            first_path_delay_bin_targets = self._first_path_delay_bin_targets(
                batch["physics_raw_targets"][:, first_delay_idx]
            )
            first_path_delay_bin_mask = (
                (first_path_delay_bin_targets >= 0)
                & batch["physics_target_mask"][:, first_delay_idx].bool()
            )
            if bool(first_path_delay_bin_mask.any()):
                valid_logits = first_path_delay_bin_logits[first_path_delay_bin_mask]
                valid_targets = first_path_delay_bin_targets[first_path_delay_bin_mask]
                per_sample_classifier_loss = torch.nn.functional.cross_entropy(
                    valid_logits,
                    valid_targets,
                    reduction="none",
                )
                class_weight = self._first_path_delay_bin_class_weight(valid_logits.dtype, cfg)
                if class_weight is None:
                    sample_weight = torch.ones_like(per_sample_classifier_loss)
                else:
                    sample_weight = class_weight[valid_targets]
                first_path_delay_bin_loss_denominator = sample_weight.sum()
                losses["loss_first_path_delay_bin_classifier"] = (
                    (per_sample_classifier_loss * sample_weight).sum()
                    / first_path_delay_bin_loss_denominator.clamp(min=1.0)
                )
                first_path_delay_bin_predictions = valid_logits.argmax(dim=1)
                losses["accuracy_first_path_delay_bin_classifier"] = (
                    (first_path_delay_bin_predictions == valid_targets).float().mean()
                )
                first_path_delay_bin_target_histogram = self._histogram(
                    valid_targets,
                    len(self.FIRST_PATH_DELAY_BIN_LABELS),
                )
                first_path_delay_bin_prediction_histogram = self._histogram(
                    first_path_delay_bin_predictions,
                    len(self.FIRST_PATH_DELAY_BIN_LABELS),
                )
        if (
            physics_outputs is not None
            and effective_estimated_pdp_tail_bin_weight > 0.0
        ):
            first_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
            first_path_delay_bin_logits = physics_outputs["first_path_delay_bin_logits"]
            first_path_delay_bin_targets = self._first_path_delay_bin_targets(
                batch["physics_raw_targets"][:, first_delay_idx]
            )
            first_path_delay_bin_mask = (
                (first_path_delay_bin_targets >= 0)
                & batch["physics_target_mask"][:, first_delay_idx].bool()
            )
            if bool(first_path_delay_bin_mask.any()):
                with torch.no_grad():
                    tail_indices = self._first_path_delay_tail_indices(
                        tuple(cfg.estimated_pdp_tail_labels)
                    )
                    pdp_bin_targets = self._estimated_pdp_argmax_delay_bins(
                        batch["tokens"],
                        batch["token_mask"],
                        batch["subcarrier_spacing"],
                    )
                    tail_gate, target_tail, pdp_tail = self._estimated_pdp_tail_gate_mask(
                        true_targets=first_path_delay_bin_targets,
                        pdp_targets=pdp_bin_targets,
                        tail_indices=tail_indices,
                        mode=cfg.estimated_pdp_tail_gate_mode,
                    )
                    tail_gate = tail_gate & first_path_delay_bin_mask
                    estimated_pdp_tail_gate_fraction = tail_gate.float().mean()
                    estimated_pdp_tail_target_fraction = (
                        target_tail & first_path_delay_bin_mask
                    ).float().mean()
                    estimated_pdp_tail_argmax_fraction = (
                        pdp_tail & first_path_delay_bin_mask
                    ).float().mean()
                if bool(tail_gate.any()):
                    tail_logits = first_path_delay_bin_logits[tail_gate]
                    tail_targets = first_path_delay_bin_targets[tail_gate]
                    tail_classifier_loss = torch.nn.functional.cross_entropy(
                        tail_logits,
                        tail_targets,
                        reduction="none",
                    )
                    estimated_pdp_tail_bin_loss_denominator = torch.tensor(
                        float(tail_targets.numel()),
                        device=self.device,
                        dtype=tail_classifier_loss.dtype,
                    )
                    losses["loss_estimated_pdp_tail_bin_classifier"] = (
                        tail_classifier_loss.mean()
                    )
                    tail_predictions = tail_logits.argmax(dim=1)
                    estimated_pdp_tail_bin_accuracy = (
                        tail_predictions == tail_targets
                    ).float().mean()
        if (
            physics_outputs is not None
            and effective_first_path_delay_tail_underestimate_weight > 0.0
        ):
            first_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
            raw_prediction = physics_outputs["first_path_delay_bin_soft_fused_raw"]
            raw_target = batch["physics_raw_targets"][:, first_delay_idx].to(
                device=raw_prediction.device,
                dtype=raw_prediction.dtype,
            )
            first_path_delay_bin_targets = self._first_path_delay_bin_targets(
                batch["physics_raw_targets"][:, first_delay_idx]
            )
            tail_indices = self._first_path_delay_tail_indices(
                tuple(cfg.estimated_pdp_tail_labels)
            )
            tail_mask = (
                (first_path_delay_bin_targets.unsqueeze(1) == tail_indices.unsqueeze(0))
                .any(dim=1)
                & batch["physics_target_mask"][:, first_delay_idx].bool()
                & torch.isfinite(raw_target)
            )
            if bool(tail_mask.any()):
                tail_errors = raw_prediction[tail_mask] - raw_target[tail_mask]
                tail_underestimate = torch.relu(-tail_errors)
                beta = max(float(cfg.first_path_delay_raw_beta_ns), 1e-6)
                tail_underestimate_loss = torch.where(
                    tail_underestimate < beta,
                    0.5 * tail_underestimate.square() / beta,
                    tail_underestimate - 0.5 * beta,
                )
                tail_weights = self._first_path_delay_sample_weights(
                    batch["physics_raw_targets"][:, first_delay_idx],
                    cfg,
                )[tail_mask]
                losses["loss_first_path_delay_tail_underestimate"] = (
                    (
                        tail_underestimate_loss
                        * tail_weights.to(dtype=tail_underestimate_loss.dtype)
                    ).sum()
                    / tail_weights.sum().clamp(min=1.0).to(dtype=tail_underestimate_loss.dtype)
                )
                first_path_delay_tail_underestimate_count = torch.tensor(
                    float(tail_errors.numel()),
                    device=self.device,
                    dtype=tail_errors.dtype,
                )
                first_path_delay_tail_underestimate_mae_ns = tail_errors.abs().mean()
                first_path_delay_tail_underestimate_mean_ns = tail_underestimate.mean()
                first_path_delay_tail_underestimate_fraction = (
                    tail_errors < 0.0
                ).float().mean()
        if (
            physics_outputs is not None
            and effective_first_path_delay_bin_position_weight > 0.0
        ):
            first_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
            first_path_delay_position_predictions = physics_outputs[
                "first_path_delay_bin_position"
            ]
            first_path_delay_position_targets, first_path_delay_position_mask = (
                self._first_path_delay_bin_position_targets(
                    batch["physics_raw_targets"][:, first_delay_idx]
                )
            )
            first_path_delay_position_mask = (
                first_path_delay_position_mask
                & batch["physics_target_mask"][:, first_delay_idx].bool()
            )
            if bool(first_path_delay_position_mask.any()):
                valid_position_predictions = first_path_delay_position_predictions[
                    first_path_delay_position_mask
                ]
                valid_position_targets = first_path_delay_position_targets[
                    first_path_delay_position_mask
                ]
                position_errors = torch.nn.functional.smooth_l1_loss(
                    valid_position_predictions,
                    valid_position_targets,
                    reduction="none",
                )
                position_weights = self._first_path_delay_sample_weights(
                    batch["physics_raw_targets"][:, first_delay_idx],
                    cfg,
                )[first_path_delay_position_mask]
                losses["loss_first_path_delay_bin_position"] = (
                    (position_errors * position_weights.to(dtype=position_errors.dtype)).sum()
                    / position_weights.sum().clamp(min=1.0).to(dtype=position_errors.dtype)
                )
                first_path_delay_bin_position_mae = (
                    valid_position_predictions - valid_position_targets
                ).abs().mean()
        if (
            physics_outputs is not None
            and (
                effective_los_delay_weight > 0.0
                or effective_los_delay_nonnegative_weight > 0.0
            )
        ):
            los_delay_prediction = physics_outputs["los_delay_context"]
            los_delay_target = batch["los_delay_target"].to(
                device=los_delay_prediction.device,
                dtype=los_delay_prediction.dtype,
            )
            los_delay_raw_target = batch["los_delay_raw_target"].to(
                device=los_delay_prediction.device,
                dtype=los_delay_prediction.dtype,
            )
            los_target_mask = batch["los_delay_target_mask"].bool()
            los_sample_mask = torch.tensor(
                [key.los_status == "los" for key in batch["semantic_keys"]],
                device=self.device,
                dtype=torch.bool,
            )
            los_delay_mask = (
                los_target_mask
                & los_sample_mask
                & torch.isfinite(los_delay_raw_target)
            )
            if effective_los_delay_weight > 0.0:
                los_delay_errors = torch.nn.functional.smooth_l1_loss(
                    los_delay_prediction,
                    los_delay_target,
                    reduction="none",
                )
                los_delay_weight = los_delay_mask.to(dtype=los_delay_errors.dtype)
                losses["loss_los_delay"] = (
                    (los_delay_errors * los_delay_weight).sum()
                    / los_delay_weight.sum().clamp(min=1).to(dtype=los_delay_errors.dtype)
                )
            if bool(los_delay_mask.any()):
                los_delay_raw_prediction = los_delay_prediction[los_delay_mask] * 3000.0
                los_delay_raw_mae_ns = (
                    los_delay_raw_prediction
                    - los_delay_raw_target[los_delay_mask]
                ).abs().mean()
                if effective_los_delay_nonnegative_weight > 0.0:
                    losses["loss_los_delay_nonnegative"] = (
                        torch.relu(-los_delay_raw_prediction).mean() / 3000.0
                    )
        if (
            physics_outputs is not None
            and effective_los_angle_weight > 0.0
        ):
            los_angle_prediction = physics_outputs["los_angle_sincos"]
            los_angle_target = batch["los_angle_target"].to(
                device=los_angle_prediction.device,
                dtype=los_angle_prediction.dtype,
            )
            los_angle_target_mask = batch["los_angle_target_mask"].bool()
            los_sample_mask = torch.tensor(
                [key.los_status == "los" for key in batch["semantic_keys"]],
                device=self.device,
                dtype=torch.bool,
            )
            los_angle_mask = (
                los_angle_target_mask
                & los_sample_mask
                & torch.isfinite(los_angle_target).all(dim=1)
            )
            if bool(los_angle_mask.any()):
                valid_predictions = torch.nn.functional.normalize(
                    los_angle_prediction[los_angle_mask],
                    dim=-1,
                    eps=1e-6,
                )
                valid_targets = torch.nn.functional.normalize(
                    los_angle_target[los_angle_mask],
                    dim=-1,
                    eps=1e-6,
                )
                cosine = (valid_predictions * valid_targets).sum(dim=-1).clamp(-1.0, 1.0)
                losses["loss_los_angle"] = (1.0 - cosine).mean()
                predicted_angle = torch.atan2(valid_predictions[:, 0], valid_predictions[:, 1])
                target_angle = torch.atan2(valid_targets[:, 0], valid_targets[:, 1])
                angle_error = torch.atan2(
                    torch.sin(predicted_angle - target_angle),
                    torch.cos(predicted_angle - target_angle),
                ).abs()
                los_angle_mae_deg = angle_error.mean() * (180.0 / math.pi)
                los_angle_count = los_angle_mask.sum()
        if (
            physics_outputs is not None
            and effective_los_delay_consistency_weight > 0.0
        ):
            first_delay_raw_prediction = physics_outputs[
                "first_path_delay_bin_soft_fused_raw"
            ]
            los_delay_raw_prediction = physics_outputs["los_delay_context"] * 3000.0
            los_delay_target_mask = batch["los_delay_target_mask"].bool()
            los_sample_mask = torch.tensor(
                [key.los_status == "los" for key in batch["semantic_keys"]],
                device=self.device,
                dtype=torch.bool,
            )
            consistency_mask = (
                los_sample_mask
                & los_delay_target_mask
                & torch.isfinite(first_delay_raw_prediction)
                & torch.isfinite(los_delay_raw_prediction)
            )
            if bool(consistency_mask.any()):
                consistency_diff_ns = (
                    first_delay_raw_prediction[consistency_mask]
                    - los_delay_raw_prediction[consistency_mask]
                ).abs()
                beta = max(float(cfg.first_path_delay_raw_beta_ns), 1e-6)
                losses["loss_los_delay_consistency"] = torch.where(
                    consistency_diff_ns < beta,
                    0.5 * consistency_diff_ns.square() / beta,
                    consistency_diff_ns - 0.5 * beta,
                ).mean() / 3000.0
                los_delay_consistency_mae_ns = consistency_diff_ns.mean()
                los_delay_consistency_count = consistency_mask.sum()
        if (
            physics_outputs is not None
            and (
                effective_first_path_angle_weight > 0.0
                or effective_first_path_angle_nlos_weight > 0.0
            )
        ):
            first_path_angle_prediction = physics_outputs["first_path_angle_sincos"]
            sin_idx = PHYSICS_TARGET_NAMES.index("first_path_aoa_az_sin")
            cos_idx = PHYSICS_TARGET_NAMES.index("first_path_aoa_az_cos")
            first_path_angle_target = batch["physics_targets"][:, [sin_idx, cos_idx]].to(
                device=first_path_angle_prediction.device,
                dtype=first_path_angle_prediction.dtype,
            )
            first_path_angle_mask = (
                batch["physics_target_mask"][:, sin_idx].bool()
                & batch["physics_target_mask"][:, cos_idx].bool()
                & torch.isfinite(first_path_angle_target).all(dim=1)
            )
            if bool(first_path_angle_mask.any()):
                valid_predictions = torch.nn.functional.normalize(
                    first_path_angle_prediction[first_path_angle_mask],
                    dim=-1,
                    eps=1e-6,
                )
                valid_targets = torch.nn.functional.normalize(
                    first_path_angle_target[first_path_angle_mask],
                    dim=-1,
                    eps=1e-6,
                )
                cosine = (valid_predictions * valid_targets).sum(dim=-1).clamp(-1.0, 1.0)
                if effective_first_path_angle_weight > 0.0:
                    losses["loss_first_path_angle"] = (1.0 - cosine).mean()
                predicted_angle = torch.atan2(valid_predictions[:, 0], valid_predictions[:, 1])
                target_angle = torch.atan2(valid_targets[:, 0], valid_targets[:, 1])
                angle_error = torch.atan2(
                    torch.sin(predicted_angle - target_angle),
                    torch.cos(predicted_angle - target_angle),
                ).abs()
                first_path_angle_mae_deg = angle_error.mean() * (180.0 / math.pi)
                first_path_angle_count = first_path_angle_mask.sum()
                nlos_sample_mask = torch.tensor(
                    [key.los_status != "los" for key in batch["semantic_keys"]],
                    device=first_path_angle_mask.device,
                    dtype=torch.bool,
                )
                first_path_angle_nlos_mask = first_path_angle_mask & nlos_sample_mask
                if bool(first_path_angle_nlos_mask.any()):
                    valid_nlos_predictions = torch.nn.functional.normalize(
                        first_path_angle_prediction[first_path_angle_nlos_mask],
                        dim=-1,
                        eps=1e-6,
                    )
                    valid_nlos_targets = torch.nn.functional.normalize(
                        first_path_angle_target[first_path_angle_nlos_mask],
                        dim=-1,
                        eps=1e-6,
                    )
                    nlos_cosine = (
                        valid_nlos_predictions * valid_nlos_targets
                    ).sum(dim=-1).clamp(-1.0, 1.0)
                    if effective_first_path_angle_nlos_weight > 0.0:
                        losses["loss_first_path_angle_nlos"] = (
                            1.0 - nlos_cosine
                        ).mean()
                    predicted_nlos_angle = torch.atan2(
                        valid_nlos_predictions[:, 0],
                        valid_nlos_predictions[:, 1],
                    )
                    target_nlos_angle = torch.atan2(
                        valid_nlos_targets[:, 0],
                        valid_nlos_targets[:, 1],
                    )
                    nlos_angle_error = torch.atan2(
                        torch.sin(predicted_nlos_angle - target_nlos_angle),
                        torch.cos(predicted_nlos_angle - target_nlos_angle),
                    ).abs()
                    first_path_angle_nlos_mae_deg = (
                        nlos_angle_error.mean() * (180.0 / math.pi)
                    )
                    first_path_angle_nlos_count = first_path_angle_nlos_mask.sum()
        if physics_outputs is not None and effective_physics_relational_weight > 0.0:
            losses.update(self._physics_relational_losses(physics_outputs, batch))
        total_loss = (
            effective_csi_to_text_weight * losses["loss_csi_to_text"] +
            effective_prototype_weight * losses["loss_csi_to_prototype"] +
            cfg.text_prototype_weight * losses["loss_text_to_prototype"] +
            effective_semantic_classifier_weight * losses.get("loss_semantic_classifier", torch.zeros((), device=self.device)) +
            effective_attribute_classifier_weight * losses.get("loss_attribute_classifier", torch.zeros((), device=self.device)) +
            effective_aux_regression_weight * losses.get("loss_aux_regression", torch.zeros((), device=self.device)) +
            effective_strong_k_bin_classifier_weight * losses.get("loss_strong_k_bin_classifier", torch.zeros((), device=self.device)) +
            effective_strong_k_position_weight * losses.get("loss_strong_k_position", torch.zeros((), device=self.device)) +
            effective_first_path_power_bin_classifier_weight * losses.get("loss_first_path_power_bin_classifier", torch.zeros((), device=self.device)) +
            effective_first_path_power_bin_position_weight * losses.get("loss_first_path_power_bin_position", torch.zeros((), device=self.device)) +
            effective_delay_spread_bin_classifier_weight * losses.get("loss_delay_spread_bin_classifier", torch.zeros((), device=self.device)) +
            effective_delay_spread_bin_position_weight * losses.get("loss_delay_spread_bin_position", torch.zeros((), device=self.device)) +
            effective_delay_spread_tail_classifier_weight * losses.get("loss_delay_spread_tail_classifier", torch.zeros((), device=self.device)) +
            effective_reflection_count_classifier_weight * losses.get("loss_interaction_count_classifier", torch.zeros((), device=self.device)) +
            effective_reflection_count_regression_weight * losses.get("loss_interaction_count_regression", torch.zeros((), device=self.device)) +
            effective_reflection_path_count_regression_weight * losses.get("loss_reflection_path_count_regression", torch.zeros((), device=self.device)) +
            effective_physics_relational_weight * losses.get("loss_physics_relational", torch.zeros((), device=self.device)) +
            cfg.direct_power_weight * losses.get("loss_direct_power", torch.zeros((), device=self.device)) +
            effective_delay_spread_weight * losses.get("loss_delay_spread", torch.zeros((), device=self.device)) +
            effective_delay_spread_raw_weight * losses.get("loss_delay_spread_raw", torch.zeros((), device=self.device)) +
            effective_first_path_delay_weight * losses.get("loss_first_path_delay", torch.zeros((), device=self.device)) +
            effective_first_path_delay_raw_weight * losses.get("loss_first_path_delay_raw", torch.zeros((), device=self.device)) +
            effective_first_path_delay_fused_raw_weight * losses.get("loss_first_path_delay_fused_raw", torch.zeros((), device=self.device)) +
            effective_first_path_delay_bin_classifier_weight * losses.get("loss_first_path_delay_bin_classifier", torch.zeros((), device=self.device)) +
            effective_first_path_delay_bin_position_weight * losses.get("loss_first_path_delay_bin_position", torch.zeros((), device=self.device)) +
            effective_first_path_delay_bin_consistency_weight * losses.get("loss_first_path_delay_bin_consistency", torch.zeros((), device=self.device)) +
            effective_estimated_pdp_tail_bin_weight * losses.get("loss_estimated_pdp_tail_bin_classifier", torch.zeros((), device=self.device)) +
            effective_first_path_delay_tail_underestimate_weight * losses.get("loss_first_path_delay_tail_underestimate", torch.zeros((), device=self.device)) +
            effective_los_delay_weight * losses.get("loss_los_delay", torch.zeros((), device=self.device)) +
            effective_los_delay_nonnegative_weight * losses.get("loss_los_delay_nonnegative", torch.zeros((), device=self.device)) +
            effective_los_delay_consistency_weight * losses.get("loss_los_delay_consistency", torch.zeros((), device=self.device)) +
            effective_los_angle_weight * losses.get("loss_los_angle", torch.zeros((), device=self.device)) +
            effective_first_path_angle_weight * losses.get("loss_first_path_angle", torch.zeros((), device=self.device)) +
            effective_first_path_angle_nlos_weight * losses.get("loss_first_path_angle_nlos", torch.zeros((), device=self.device))
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
        if physics_outputs is not None and "enhanced_gate" in physics_outputs:
            metrics["enhanced_gate_mean"] = float(
                physics_outputs["enhanced_gate"].detach().float().mean()
            )
            metrics["enhanced_gate_std"] = float(
                physics_outputs["enhanced_gate"].detach().float().std()
            )
        if physics_outputs is not None and "enhanced_delay_spread_gate" in physics_outputs:
            metrics["delay_spread_gate_mean"] = float(
                physics_outputs["enhanced_delay_spread_gate"].detach().float().mean()
            )
            metrics["delay_spread_delta_mean"] = float(
                physics_outputs["enhanced_delay_spread_delta"].detach().float().mean()
            )
            metrics["delay_spread_delta_std"] = float(
                physics_outputs["enhanced_delay_spread_delta"].detach().float().std()
            )
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
        metrics["delay_spread_weight"] = float(effective_delay_spread_weight)
        metrics["delay_spread_raw_weight"] = float(effective_delay_spread_raw_weight)
        metrics["delay_spread_raw_beta_ns"] = float(cfg.delay_spread_raw_beta_ns)
        if delay_spread_raw_mae_ns is not None:
            metrics["delay_spread_raw_mae_ns"] = float(
                delay_spread_raw_mae_ns.detach()
            )
        if delay_spread_normalized_mae_ns is not None:
            metrics["delay_spread_normalized_mae_ns"] = float(
                delay_spread_normalized_mae_ns.detach()
            )
        metrics["first_path_delay_weight"] = float(effective_first_path_delay_weight)
        metrics["first_path_delay_raw_weight"] = float(effective_first_path_delay_raw_weight)
        metrics["first_path_delay_fused_raw_weight"] = float(
            effective_first_path_delay_fused_raw_weight
        )
        metrics["first_path_delay_raw_beta_ns"] = float(cfg.first_path_delay_raw_beta_ns)
        metrics["first_path_delay_bin_classifier_weight"] = float(
            effective_first_path_delay_bin_classifier_weight
        )
        metrics["first_path_delay_bin_position_weight"] = float(
            effective_first_path_delay_bin_position_weight
        )
        metrics["first_path_delay_bin_consistency_weight"] = float(
            effective_first_path_delay_bin_consistency_weight
        )
        metrics["estimated_pdp_tail_bin_weight"] = float(
            effective_estimated_pdp_tail_bin_weight
        )
        metrics["estimated_pdp_tail_labels"] = ",".join(
            tuple(cfg.estimated_pdp_tail_labels)
        )
        metrics["estimated_pdp_tail_gate_mode"] = str(
            cfg.estimated_pdp_tail_gate_mode
        )
        metrics["first_path_delay_tail_underestimate_weight"] = float(
            effective_first_path_delay_tail_underestimate_weight
        )
        metrics["first_path_delay_bin_label_order"] = ",".join(
            self.FIRST_PATH_DELAY_BIN_LABELS
        )
        metrics["first_path_delay_bin_class_weights"] = self._format_float_vector(
            self._first_path_delay_bin_class_weight(
                torch.float32,
                cfg,
            )
            if cfg.first_path_delay_bin_weights
            else torch.ones(len(self.FIRST_PATH_DELAY_BIN_LABELS), device=self.device)
        )
        if first_path_delay_bin_target_histogram is not None:
            metrics["first_path_delay_bin_target_histogram"] = self._format_histogram(
                first_path_delay_bin_target_histogram
            )
        if first_path_delay_bin_prediction_histogram is not None:
            metrics["first_path_delay_bin_prediction_histogram"] = self._format_histogram(
                first_path_delay_bin_prediction_histogram
            )
        if first_path_delay_bin_loss_denominator is not None:
            metrics["first_path_delay_bin_loss_denominator"] = float(
                first_path_delay_bin_loss_denominator.detach()
            )
        if first_path_delay_bin_position_mae is not None:
            metrics["first_path_delay_bin_position_mae"] = float(
                first_path_delay_bin_position_mae.detach()
            )
        if first_path_delay_raw_mae_ns is not None:
            metrics["first_path_delay_raw_mae_ns"] = float(
                first_path_delay_raw_mae_ns.detach()
            )
        if first_path_delay_fused_raw_mae_ns is not None:
            metrics["first_path_delay_fused_raw_mae_ns"] = float(
                first_path_delay_fused_raw_mae_ns.detach()
            )
        if first_path_delay_bin_consistency_violation_ns is not None:
            metrics["first_path_delay_bin_consistency_violation_ns"] = float(
                first_path_delay_bin_consistency_violation_ns.detach()
            )
        if first_path_delay_bin_consistency_max_violation_ns is not None:
            metrics["first_path_delay_bin_consistency_max_violation_ns"] = float(
                first_path_delay_bin_consistency_max_violation_ns.detach()
            )
        if estimated_pdp_tail_bin_loss_denominator is not None:
            metrics["estimated_pdp_tail_bin_loss_denominator"] = float(
                estimated_pdp_tail_bin_loss_denominator.detach()
            )
        if estimated_pdp_tail_bin_accuracy is not None:
            metrics["accuracy_estimated_pdp_tail_bin_classifier"] = float(
                estimated_pdp_tail_bin_accuracy.detach()
            )
        if estimated_pdp_tail_gate_fraction is not None:
            metrics["estimated_pdp_tail_gate_fraction"] = float(
                estimated_pdp_tail_gate_fraction.detach()
            )
        if estimated_pdp_tail_target_fraction is not None:
            metrics["estimated_pdp_tail_target_fraction"] = float(
                estimated_pdp_tail_target_fraction.detach()
            )
        if estimated_pdp_tail_argmax_fraction is not None:
            metrics["estimated_pdp_tail_argmax_fraction"] = float(
                estimated_pdp_tail_argmax_fraction.detach()
            )
        if first_path_delay_tail_underestimate_count is not None:
            metrics["first_path_delay_tail_underestimate_count"] = float(
                first_path_delay_tail_underestimate_count.detach()
            )
        if first_path_delay_tail_underestimate_mae_ns is not None:
            metrics["first_path_delay_tail_underestimate_mae_ns"] = float(
                first_path_delay_tail_underestimate_mae_ns.detach()
            )
        if first_path_delay_tail_underestimate_mean_ns is not None:
            metrics["first_path_delay_tail_underestimate_mean_ns"] = float(
                first_path_delay_tail_underestimate_mean_ns.detach()
            )
        if first_path_delay_tail_underestimate_fraction is not None:
            metrics["first_path_delay_tail_underestimate_fraction"] = float(
                first_path_delay_tail_underestimate_fraction.detach()
            )
        metrics["los_delay_weight"] = float(effective_los_delay_weight)
        metrics["los_delay_nonnegative_weight"] = float(
            effective_los_delay_nonnegative_weight
        )
        metrics["use_physics_calibration_loss"] = float(
            cfg.use_physics_calibration_loss
        )
        metrics["los_delay_consistency_weight"] = float(
            effective_los_delay_consistency_weight
        )
        if los_delay_raw_mae_ns is not None:
            metrics["los_delay_raw_mae_ns"] = float(los_delay_raw_mae_ns.detach())
        if "los_delay_consistency_mae_ns" in locals():
            metrics["los_delay_consistency_mae_ns"] = float(
                los_delay_consistency_mae_ns.detach()
            )
        if "los_delay_consistency_count" in locals():
            metrics["los_delay_consistency_count"] = float(
                los_delay_consistency_count.detach()
            )
        metrics["los_angle_weight"] = float(effective_los_angle_weight)
        if los_angle_mae_deg is not None:
            metrics["los_angle_mae_deg"] = float(los_angle_mae_deg.detach())
        if los_angle_count is not None:
            metrics["los_angle_count"] = float(los_angle_count.detach())
        metrics["first_path_angle_weight"] = float(effective_first_path_angle_weight)
        metrics["first_path_angle_nlos_weight"] = float(
            effective_first_path_angle_nlos_weight
        )
        if first_path_angle_mae_deg is not None:
            metrics["first_path_angle_mae_deg"] = float(
                first_path_angle_mae_deg.detach()
            )
        if first_path_angle_count is not None:
            metrics["first_path_angle_count"] = float(
                first_path_angle_count.detach()
            )
        if first_path_angle_nlos_mae_deg is not None:
            metrics["first_path_angle_nlos_mae_deg"] = float(
                first_path_angle_nlos_mae_deg.detach()
            )
        if first_path_angle_nlos_count is not None:
            metrics["first_path_angle_nlos_count"] = float(
                first_path_angle_nlos_count.detach()
            )
        metrics["delay_spread_teacher_weight"] = float(cfg.delay_spread_teacher_weight)
        metrics["delay_spread_bin_classifier_weight"] = float(
            effective_delay_spread_bin_classifier_weight
        )
        metrics["delay_spread_bin_position_weight"] = float(
            effective_delay_spread_bin_position_weight
        )
        metrics["delay_spread_tail_classifier_weight"] = float(
            effective_delay_spread_tail_classifier_weight
        )
        metrics["delay_spread_bin_label_order"] = ",".join(self.DELAY_SPREAD_BIN_LABELS)
        metrics["delay_spread_tail_label_order"] = ",".join(self.DELAY_SPREAD_TAIL_LABELS)
        metrics["delay_spread_bin_class_weights"] = self._format_float_vector(
            self._delay_spread_bin_class_weight(
                torch.float32,
                cfg,
            )
            if cfg.delay_spread_bin_weights
            else torch.ones(len(self.DELAY_SPREAD_BIN_LABELS), device=self.device)
        )
        if delay_spread_bin_target_histogram is not None:
            metrics["delay_spread_bin_target_histogram"] = self._format_histogram(
                delay_spread_bin_target_histogram
            )
        if delay_spread_bin_prediction_histogram is not None:
            metrics["delay_spread_bin_prediction_histogram"] = self._format_histogram(
                delay_spread_bin_prediction_histogram
            )
        if delay_spread_bin_loss_denominator is not None:
            metrics["delay_spread_bin_loss_denominator"] = float(
                delay_spread_bin_loss_denominator.detach()
            )
        if delay_spread_bin_position_mae is not None:
            metrics["delay_spread_bin_position_mae"] = float(
                delay_spread_bin_position_mae.detach()
            )
        if delay_spread_tail_accuracy is not None:
            metrics["delay_spread_tail_accuracy"] = float(
                delay_spread_tail_accuracy.detach().nanmean()
            )
            metrics["delay_spread_tail_recall"] = float(
                delay_spread_tail_recall.detach().nanmean()
            )
            metrics["delay_spread_tail_false_positive"] = float(
                delay_spread_tail_false_positive.detach().nanmean()
            )
            metrics["delay_spread_tail_positive_fraction"] = self._format_float_vector(
                delay_spread_tail_positive_fraction,
            )
            metrics["delay_spread_tail_prediction_fraction"] = self._format_float_vector(
                delay_spread_tail_prediction_fraction,
            )
            for idx, label in enumerate(self.DELAY_SPREAD_TAIL_LABELS):
                metrics[f"delay_spread_tail_{label}_accuracy"] = float(
                    delay_spread_tail_accuracy[idx].detach()
                )
                metrics[f"delay_spread_tail_{label}_recall"] = float(
                    delay_spread_tail_recall[idx].detach()
                )
                metrics[f"delay_spread_tail_{label}_false_positive"] = float(
                    delay_spread_tail_false_positive[idx].detach()
                )
        metrics["reflection_count_classifier_weight"] = float(
            effective_reflection_count_classifier_weight
        )
        metrics["reflection_count_regression_weight"] = float(
            effective_reflection_count_regression_weight
        )
        metrics["reflection_path_count_regression_weight"] = float(
            effective_reflection_path_count_regression_weight
        )
        metrics["interaction_count_classifier_weight"] = float(
            effective_reflection_count_classifier_weight
        )
        metrics["interaction_count_regression_weight"] = float(
            effective_reflection_count_regression_weight
        )
        metrics["reflection_count_nlos_weight"] = float(cfg.reflection_count_nlos_weight)
        metrics["interaction_count_soft_labels"] = float(
            cfg.interaction_count_soft_labels
        )
        metrics["physics_relational_weight"] = float(effective_physics_relational_weight)
        metrics["reflection_count_bin_label_order"] = ",".join(
            self.REFLECTION_COUNT_BIN_LABELS
        )
        if reflection_count_accuracy is not None:
            metrics["accuracy_reflection_count_classifier"] = float(
                reflection_count_accuracy.detach()
            )
        if reflection_count_mae is not None:
            metrics["reflection_count_mae"] = float(reflection_count_mae.detach())
        if reflection_count_target_histogram is not None:
            metrics["reflection_count_target_histogram"] = self._format_histogram(
                reflection_count_target_histogram
            )
        if reflection_count_prediction_histogram is not None:
            metrics["reflection_count_prediction_histogram"] = self._format_histogram(
                reflection_count_prediction_histogram
            )
        if reflection_path_count_mae is not None:
            metrics["reflection_path_count_mae"] = float(
                reflection_path_count_mae.detach()
            )
        if reflection_path_count_exact_accuracy is not None:
            metrics["reflection_path_count_exact_accuracy"] = float(
                reflection_path_count_exact_accuracy.detach()
            )
        if reflection_path_count_target_histogram is not None:
            metrics["reflection_path_count_target_histogram"] = self._format_histogram(
                reflection_path_count_target_histogram
            )
        if reflection_path_count_prediction_histogram is not None:
            metrics["reflection_path_count_prediction_histogram"] = self._format_histogram(
                reflection_path_count_prediction_histogram
            )
        metrics["strong_k_bin_classifier_weight"] = float(
            effective_strong_k_bin_classifier_weight
        )
        metrics["strong_k_position_weight"] = float(
            effective_strong_k_position_weight
        )
        metrics["strong_k_bin_label_order"] = ",".join(self.STRONG_K_BIN_LABELS)
        metrics["strong_k_bin_class_weights"] = self._format_float_vector(
            self._strong_k_bin_class_weight(
                torch.float32,
                cfg,
            )
            if cfg.strong_k_bin_weights
            else torch.ones(len(self.STRONG_K_BIN_LABELS), device=self.device)
        )
        if strong_k_bin_target_histogram is not None:
            metrics["strong_k_bin_target_histogram"] = self._format_histogram(
                strong_k_bin_target_histogram
            )
        if strong_k_bin_prediction_histogram is not None:
            metrics["strong_k_bin_prediction_histogram"] = self._format_histogram(
                strong_k_bin_prediction_histogram
            )
        if strong_k_bin_loss_denominator is not None:
            metrics["strong_k_bin_loss_denominator"] = float(
                strong_k_bin_loss_denominator.detach()
            )
        if strong_k_position_mae is not None:
            metrics["strong_k_position_mae"] = float(strong_k_position_mae.detach())
        metrics["first_path_power_bin_classifier_weight"] = float(
            effective_first_path_power_bin_classifier_weight
        )
        metrics["first_path_power_bin_position_weight"] = float(
            effective_first_path_power_bin_position_weight
        )
        metrics["first_path_power_bin_label_order"] = ",".join(self.FIRST_PATH_POWER_BIN_LABELS)
        if first_path_power_bin_target_histogram is not None:
            metrics["first_path_power_bin_target_histogram"] = self._format_histogram(
                first_path_power_bin_target_histogram
            )
        if first_path_power_bin_prediction_histogram is not None:
            metrics["first_path_power_bin_prediction_histogram"] = self._format_histogram(
                first_path_power_bin_prediction_histogram
            )
        if first_path_power_bin_loss_denominator is not None:
            metrics["first_path_power_bin_loss_denominator"] = float(
                first_path_power_bin_loss_denominator.detach()
            )
        if first_path_power_bin_position_mae is not None:
            metrics["first_path_power_bin_position_mae"] = float(
                first_path_power_bin_position_mae.detach()
            )
        metrics["direct_power_weight"] = float(cfg.direct_power_weight)
        metrics["nlos_enhanced_power_loss"] = float(cfg.nlos_enhanced_power_loss)
        if direct_power_loss_count is not None:
            metrics["direct_power_loss_count"] = float(direct_power_loss_count.detach())
        if direct_power_loss_nlos_fraction is not None:
            metrics["direct_power_loss_nlos_fraction"] = float(
                direct_power_loss_nlos_fraction.detach()
            )
        metrics["first_path_power_mode_absolute"] = float(
            cfg.first_path_power_mode == "absolute"
        )
        metrics["first_path_power_use_internal_gate"] = float(
            cfg.first_path_power_use_internal_gate
        )
        if los_first_path_power_base_mae_db is not None:
            metrics["los_first_path_power_base_mae_db"] = float(
                los_first_path_power_base_mae_db.detach()
            )
        if nlos_first_path_power_base_mae_db is not None:
            metrics["nlos_first_path_power_base_mae_db"] = float(
                nlos_first_path_power_base_mae_db.detach()
            )
        if nlos_first_path_power_enhanced_mae_db is not None:
            metrics["nlos_first_path_power_enhanced_mae_db"] = float(
                nlos_first_path_power_enhanced_mae_db.detach()
            )
        if first_path_power_delta_abs_mean is not None:
            metrics["first_path_power_delta_abs_mean"] = float(
                first_path_power_delta_abs_mean.detach()
            )
        if first_path_power_delta_saturation_fraction is not None:
            metrics["first_path_power_delta_saturation_fraction"] = float(
                first_path_power_delta_saturation_fraction.detach()
            )
        if physics_predictions is not None:
            metrics["k_factor_sample_weight_mean"] = float(
                self._k_factor_sample_weights(
                    batch["physics_raw_targets"][:, 3],
                    batch["semantic_keys"],
                    cfg,
                ).detach().float().mean()
            )
            metrics["first_path_power_sample_weight_mean"] = float(
                self._first_path_power_supervision_weights(
                    batch["physics_raw_targets"][:, 5],
                    batch["semantic_keys"],
                    cfg,
                ).detach().float().mean()
            )
            metrics["first_path_power_nlos_weight"] = float(
                cfg.first_path_power_nlos_weight
            )
            if "first_path_delay_ns" in PHYSICS_TARGET_NAMES:
                first_delay_idx = PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
                metrics["first_path_delay_sample_weight_mean"] = float(
                    self._first_path_delay_sample_weights(
                        batch["physics_raw_targets"][:, first_delay_idx],
                        cfg,
                    ).detach().float().mean()
                )
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
