from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model import (
    CSIArrayInvariantDelayEncoder,
    CSI_DELAY_CONTEXT_DIM,
)  # noqa: E402
from data.dataset import PreprocessedCSIDataset  # noqa: E402
from scripts.diagnose_multinumerology_delay_fusion import metric_rows  # noqa: E402
from scripts.train_multinumerology_delay_fusion import (  # noqa: E402
    TARGET_NAMES,
    PairedSample,
    circular_mae,
    finite_float,
    make_collate,
    move_nested,
    target_value,
    target_unit,
)


FINAL_OUTPUT_KEY = "fused_gate_ab_hard"
FINAL_OUTPUT_METHOD = "paired_gate_ab_hard"


@dataclass(frozen=True)
class PairSpec:
    name_a: str
    period_a_ns: float
    path_a: str
    name_b: str
    period_b_ns: float
    path_b: str

    @property
    def name(self) -> str:
        return f"{self.name_a}+{self.name_b}"


class PairDataset(Dataset[PairedSample]):
    def __init__(self, pairs: list[PairedSample]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> PairedSample:
        return self.pairs[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_pair(values: list[str]) -> PairSpec:
    name_a, raw_period_a, path_a, name_b, raw_period_b, path_b = values
    period_a = float(raw_period_a)
    period_b = float(raw_period_b)
    if period_a <= 0.0 or period_b <= 0.0:
        raise ValueError("Numerology periods must be positive.")
    if period_a == period_b:
        raise ValueError("A fusion pair must use two different periods.")
    if period_a > period_b:
        name_a, name_b = name_b, name_a
        period_a, period_b = period_b, period_a
        path_a, path_b = path_b, path_a
    return PairSpec(name_a, period_a, path_a, name_b, period_b, path_b)


def pair_collate(spec: PairSpec, max_delay_ns: float):
    base_collate = make_collate(max_delay_ns, spec.period_a_ns)

    def collate(batch: list[PairedSample]) -> dict:
        result = base_collate(batch)
        batch_size = len(batch)
        result["period_a_ns"] = torch.full(
            (batch_size,), spec.period_a_ns, dtype=torch.float32
        )
        result["period_b_ns"] = torch.full(
            (batch_size,), spec.period_b_ns, dtype=torch.float32
        )
        result["availability_a"] = torch.ones(batch_size, dtype=torch.bool)
        result["availability_b"] = torch.ones(batch_size, dtype=torch.bool)
        return result

    return collate


def load_sample_map(
    path: str, cache: dict[str, tuple[list[object], dict[str, object]]]
):
    cached = cache.get(path)
    if cached is not None:
        return cached
    samples = PreprocessedCSIDataset.from_pt(path).samples
    grouped = {}
    for sample in samples:
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError(f"Sample without group_id in {path}.")
        if group_id in grouped:
            raise ValueError(f"Duplicate group_id={group_id!r} in {path}.")
        grouped[group_id] = sample
    cached = (samples, grouped)
    cache[path] = cached
    return cached


def build_cached_pairs(
    spec: PairSpec,
    *,
    max_delay_spread_ns: float | None,
    cache: dict[str, tuple[list[object], dict[str, object]]],
    target_tolerance_ns: float = 1e-3,
) -> tuple[list[PairedSample], dict[str, int]]:
    samples_a, grouped_a = load_sample_map(spec.path_a, cache)
    _, grouped_b = load_sample_map(spec.path_b, cache)
    pairs = []
    missing = 0
    label_mismatch = 0
    delay_spread_filtered = 0
    for sample_a in samples_a:
        sample_b = grouped_b.get(str(sample_a.group_id))
        if sample_b is None:
            missing += 1
            continue
        if sample_a.semantic_key.los_status != sample_b.semantic_key.los_status:
            label_mismatch += 1
            continue
        mismatch = False
        for target_name in TARGET_NAMES:
            value_a = target_value(sample_a, target_name)
            value_b = target_value(sample_b, target_name)
            if (value_a is None) != (value_b is None):
                mismatch = True
                break
            if (
                value_a is not None
                and value_b is not None
                and abs(value_a - value_b) > target_tolerance_ns
            ):
                mismatch = True
                break
        if mismatch:
            label_mismatch += 1
            continue
        if max_delay_spread_ns is not None:
            delay_spread_s = finite_float(getattr(sample_a, "delay_spread_s", math.nan))
            delay_spread_ns = (
                math.nan if delay_spread_s is None else delay_spread_s * 1e9
            )
            if (
                not math.isfinite(delay_spread_ns)
                or delay_spread_ns >= max_delay_spread_ns
            ):
                delay_spread_filtered += 1
                continue
        pairs.append(PairedSample(sample_a=sample_a, sample_b=sample_b))
    if label_mismatch:
        raise ValueError(
            f"Found {label_mismatch} samples with mismatched labels in {spec.name}."
        )
    if not pairs:
        raise ValueError(f"No valid paired samples remain for {spec.name}.")
    return pairs, {
        "input_a_count": len(grouped_a),
        "input_b_count": len(grouped_b),
        "paired_count": len(pairs),
        "missing_count": missing,
        "delay_spread_filtered_count": delay_spread_filtered,
    }


class PredictionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PeriodConditionedDelayFusion(nn.Module):
    def __init__(
        self,
        *,
        max_delay_ns: float,
        hidden_dim: int,
        residual_scale_ns: float,
        period_prior_strength: float,
        fallback_confidence_threshold: float,
    ):
        super().__init__()
        self.max_delay_ns = float(max_delay_ns)
        self.residual_scale_ns = float(residual_scale_ns)
        self.period_prior_strength = float(period_prior_strength)
        self.fallback_confidence_threshold = float(fallback_confidence_threshold)
        self.encoder = CSIArrayInvariantDelayEncoder(out_dim=CSI_DELAY_CONTEXT_DIM)
        self.single_heads = nn.ModuleDict(
            {
                name: PredictionHead(CSI_DELAY_CONTEXT_DIM, 1, hidden_dim)
                for name in TARGET_NAMES
            }
        )
        self.residue_heads = nn.ModuleDict(
            {
                name: PredictionHead(CSI_DELAY_CONTEXT_DIM, 2, hidden_dim)
                for name in TARGET_NAMES
            }
        )
        self.uncertainty_heads = nn.ModuleDict(
            {
                name: PredictionHead(CSI_DELAY_CONTEXT_DIM, 1, hidden_dim)
                for name in TARGET_NAMES
            }
        )
        config_feature_dim = self.encoder.config_gamma.in_features
        fusion_input_dim = CSI_DELAY_CONTEXT_DIM * 4 + 6 + config_feature_dim * 2 + 2
        self.fusion_trunks = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(fusion_input_dim),
                    nn.Linear(fusion_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                for name in TARGET_NAMES
            }
        )
        self.fused_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, 1) for name in TARGET_NAMES}
        )
        gate_diagnostic_dim = 11
        self.gate_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim + gate_diagnostic_dim),
                    nn.Linear(hidden_dim + gate_diagnostic_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 3),
                )
                for name in TARGET_NAMES
            }
        )
        self.residual_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, 1) for name in TARGET_NAMES}
        )

    def encode(self, view: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(
            view["tokens"],
            view["token_mask"],
            subcarrier_spacing=view["subcarrier_spacing"],
            config_features=view["config_features"],
        )

    @staticmethod
    def decode_residue(unit: torch.Tensor, period_ns: torch.Tensor) -> torch.Tensor:
        angle = torch.atan2(unit[:, 0], unit[:, 1])
        fraction = torch.remainder(angle / (2.0 * math.pi), 1.0)
        return fraction * period_ns

    @staticmethod
    def residue_consistency(
        prediction: torch.Tensor,
        residue_unit: torch.Tensor,
        period_ns: torch.Tensor,
    ) -> torch.Tensor:
        return (target_unit(prediction, period_ns) * residue_unit).sum(dim=-1)

    def forward(
        self,
        view_a: dict[str, torch.Tensor],
        view_b: dict[str, torch.Tensor],
        period_a_ns: torch.Tensor,
        period_b_ns: torch.Tensor,
        availability_a: torch.Tensor,
        availability_b: torch.Tensor,
    ) -> dict:
        context_a = self.encode(view_a)
        context_b = self.encode(view_b)
        availability_a_float = availability_a.to(dtype=context_a.dtype)
        availability_b_float = availability_b.to(dtype=context_b.dtype)
        context_a_fusion = context_a * availability_a_float.unsqueeze(-1)
        context_b_fusion = context_b * availability_b_float.unsqueeze(-1)
        period_features = (
            torch.stack([period_a_ns, period_b_ns], dim=-1) / self.max_delay_ns
        )
        availability_features = torch.stack(
            [availability_a_float, availability_b_float], dim=-1
        )
        outputs = {}
        for target_name in TARGET_NAMES:
            single_a = (
                self.single_heads[target_name](context_a).squeeze(-1)
                * self.max_delay_ns
            )
            single_b = (
                self.single_heads[target_name](context_b).squeeze(-1)
                * self.max_delay_ns
            )
            single_a_expert = single_a.detach()
            single_b_expert = single_b.detach()
            log_variance_a = (
                self.uncertainty_heads[target_name](context_a)
                .squeeze(-1)
                .clamp(-6.0, 4.0)
            )
            log_variance_b = (
                self.uncertainty_heads[target_name](context_b)
                .squeeze(-1)
                .clamp(-6.0, 4.0)
            )
            residue_unit_a = F.normalize(
                self.residue_heads[target_name](context_a), dim=-1, eps=1e-6
            )
            residue_unit_b = F.normalize(
                self.residue_heads[target_name](context_b), dim=-1, eps=1e-6
            )
            fusion_input = torch.cat(
                [
                    context_a_fusion,
                    context_b_fusion,
                    (context_a_fusion - context_b_fusion).abs(),
                    context_a_fusion * context_b_fusion,
                    residue_unit_a * availability_a_float.unsqueeze(-1),
                    residue_unit_b * availability_b_float.unsqueeze(-1),
                    period_features,
                    view_a["config_features"],
                    view_b["config_features"],
                    availability_features,
                ],
                dim=-1,
            )
            fused_features = self.fusion_trunks[target_name](fusion_input)
            direct_fused = (
                self.fused_heads[target_name](fused_features).squeeze(-1)
                * self.max_delay_ns
            )
            self_consistency_a = self.residue_consistency(
                single_a_expert, residue_unit_a, period_a_ns
            )
            self_consistency_b = self.residue_consistency(
                single_b_expert, residue_unit_b, period_b_ns
            )
            cross_consistency_a_to_b = self.residue_consistency(
                single_a_expert, residue_unit_b, period_b_ns
            )
            cross_consistency_b_to_a = self.residue_consistency(
                single_b_expert, residue_unit_a, period_a_ns
            )
            gate_diagnostics = torch.stack(
                [
                    log_variance_a,
                    log_variance_b,
                    self_consistency_a,
                    self_consistency_b,
                    cross_consistency_a_to_b,
                    cross_consistency_b_to_a,
                    (single_a_expert - single_b_expert).abs() / self.max_delay_ns,
                    (direct_fused.detach() - single_a_expert).abs() / self.max_delay_ns,
                    (direct_fused.detach() - single_b_expert).abs() / self.max_delay_ns,
                    availability_a_float,
                    availability_b_float,
                ],
                dim=-1,
            )
            learned_gate_logits = self.gate_heads[target_name](
                torch.cat([fused_features, gate_diagnostics], dim=-1)
            )
            uncertainty_prior = torch.stack(
                [
                    -log_variance_a,
                    -log_variance_b,
                    torch.zeros_like(log_variance_a),
                ],
                dim=-1,
            )
            log_period_a = torch.log(period_a_ns.clamp(min=1.0))
            log_period_b = torch.log(period_b_ns.clamp(min=1.0))
            centered_log_period = 0.5 * (log_period_a + log_period_b)
            period_prior = self.period_prior_strength * torch.stack(
                [
                    log_period_a - centered_log_period,
                    log_period_b - centered_log_period,
                    torch.zeros_like(log_period_a),
                ],
                dim=-1,
            )
            gate_logits = learned_gate_logits + uncertainty_prior + period_prior
            availability = torch.stack(
                [availability_a, availability_b, torch.ones_like(availability_a)],
                dim=-1,
            )
            gate_logits = gate_logits.masked_fill(~availability, -1.0e4)
            gate_weights = torch.softmax(gate_logits, dim=-1)
            gate_ab_logits = gate_logits[:, :2]
            gate_ab_hard_index = gate_ab_logits.argmax(dim=-1)
            gate_ab_hard = torch.where(
                gate_ab_hard_index == 0,
                single_a_expert,
                single_b_expert,
            )
            residual = (
                torch.tanh(self.residual_heads[target_name](fused_features).squeeze(-1))
                * self.residual_scale_ns
            )
            expert_candidates = torch.stack(
                [single_a_expert, single_b_expert, direct_fused], dim=-1
            )
            expert_soft = (gate_weights * expert_candidates).sum(dim=-1) + residual
            expert_hard_index = gate_logits.argmax(dim=-1)
            expert_hard = expert_candidates.gather(
                1, expert_hard_index.unsqueeze(-1)
            ).squeeze(-1)
            gate_confidence = gate_weights.max(dim=-1).values
            prefer_a_for_period = availability_a & (
                ~availability_b | (period_a_ns >= period_b_ns)
            )
            max_period_single = torch.where(
                prefer_a_for_period,
                single_a_expert,
                single_b_expert,
            )
            fallback_mask = gate_confidence < self.fallback_confidence_threshold
            confidence_fallback = torch.where(
                fallback_mask,
                max_period_single,
                expert_soft,
            )
            outputs[target_name] = {
                "single_a": single_a,
                "single_b": single_b,
                "log_variance_a": log_variance_a,
                "log_variance_b": log_variance_b,
                "residue_unit_a": residue_unit_a,
                "residue_unit_b": residue_unit_b,
                "residue_a": self.decode_residue(residue_unit_a, period_a_ns),
                "residue_b": self.decode_residue(residue_unit_b, period_b_ns),
                "fused_direct": direct_fused,
                "fused_gate_ab_hard": gate_ab_hard,
                "fused_expert_soft": expert_soft,
                "fused_expert_hard": expert_hard,
                "fused_confidence_fallback": confidence_fallback,
                "gate_logits": gate_logits,
                "gate_ab_logits": gate_ab_logits,
                "gate_weight_a": gate_weights[:, 0],
                "gate_weight_b": gate_weights[:, 1],
                "gate_weight_direct": gate_weights[:, 2],
                "gate_confidence": gate_confidence,
                "fallback_mask": fallback_mask,
                "residual": residual,
            }
        return outputs


def compute_loss(
    outputs: dict,
    batch: dict,
    *,
    max_delay_ns: float,
    single_weight: float,
    residue_weight: float,
    consistency_weight: float,
    direct_weight: float,
    uncertainty_weight: float,
    uncertainty_ranking_weight: float,
    uncertainty_ranking_margin: float,
    uncertainty_ranking_min_gap_ns: float,
    gate_weight: float,
    residual_weight: float,
    regret_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = next(iter(outputs.values()))["single_a"].new_zeros(())
    components = {}
    active_targets = 0
    for target_name in TARGET_NAMES:
        mask = batch["masks"][target_name]
        if not bool(mask.any()):
            continue
        active_targets += 1
        target = batch["targets"][target_name][mask]
        normalized_target = target / max_delay_ns
        output = outputs[target_name]
        period_a = batch["period_a_ns"][mask]
        period_b = batch["period_b_ns"][mask]
        single_loss = 0.5 * (
            F.smooth_l1_loss(output["single_a"][mask] / max_delay_ns, normalized_target)
            + F.smooth_l1_loss(
                output["single_b"][mask] / max_delay_ns, normalized_target
            )
        )
        target_unit_a = target_unit(target, period_a)
        target_unit_b = target_unit(target, period_b)
        residue_loss = 0.5 * (
            (1.0 - (output["residue_unit_a"][mask] * target_unit_a).sum(dim=-1)).mean()
            + (
                1.0 - (output["residue_unit_b"][mask] * target_unit_b).sum(dim=-1)
            ).mean()
        )
        direct_fused = output["fused_direct"][mask]
        expert_soft = output["fused_expert_soft"][mask]
        direct_loss = F.smooth_l1_loss(direct_fused / max_delay_ns, normalized_target)
        expert_soft_loss = F.smooth_l1_loss(
            expert_soft / max_delay_ns, normalized_target
        )
        normalized_error_a = output["single_a"][mask] / max_delay_ns - normalized_target
        normalized_error_b = output["single_b"][mask] / max_delay_ns - normalized_target
        log_variance_a = output["log_variance_a"][mask]
        log_variance_b = output["log_variance_b"][mask]
        uncertainty_loss = 0.25 * (
            (
                torch.exp(-log_variance_a) * normalized_error_a.square()
                + log_variance_a
            ).mean()
            + (
                torch.exp(-log_variance_b) * normalized_error_b.square()
                + log_variance_b
            ).mean()
        )
        single_errors = torch.stack(
            [normalized_error_a.abs(), normalized_error_b.abs()], dim=-1
        )
        availability_a = batch["availability_a"][mask]
        availability_b = batch["availability_b"][mask]
        ranking_gap = uncertainty_ranking_min_gap_ns / max_delay_ns
        absolute_error_a = normalized_error_a.detach().abs()
        absolute_error_b = normalized_error_b.detach().abs()
        ranking_available = availability_a & availability_b
        rank_a_better = ranking_available & (
            absolute_error_b - absolute_error_a > ranking_gap
        )
        rank_b_better = ranking_available & (
            absolute_error_a - absolute_error_b > ranking_gap
        )
        ranking_terms = []
        if bool(rank_a_better.any()):
            ranking_terms.append(
                F.relu(
                    uncertainty_ranking_margin
                    - (log_variance_b[rank_a_better] - log_variance_a[rank_a_better])
                ).mean()
            )
        if bool(rank_b_better.any()):
            ranking_terms.append(
                F.relu(
                    uncertainty_ranking_margin
                    - (log_variance_a[rank_b_better] - log_variance_b[rank_b_better])
                ).mean()
            )
        uncertainty_ranking_loss = (
            torch.stack(ranking_terms).mean()
            if ranking_terms
            else uncertainty_loss.new_zeros(())
        )
        oracle_ab_target = single_errors.argmin(dim=-1)
        oracle_ab_target = torch.where(
            ~availability_a,
            torch.ones_like(oracle_ab_target),
            oracle_ab_target,
        )
        oracle_ab_target = torch.where(
            ~availability_b,
            torch.zeros_like(oracle_ab_target),
            oracle_ab_target,
        )
        gate_ab_loss = F.cross_entropy(output["gate_ab_logits"][mask], oracle_ab_target)
        direct_error = (direct_fused / max_delay_ns - normalized_target).abs()
        expert_errors = torch.cat([single_errors, direct_error.unsqueeze(-1)], dim=-1)
        expert_availability = torch.stack(
            [availability_a, availability_b, torch.ones_like(availability_a)],
            dim=-1,
        )
        expert_errors = expert_errors.masked_fill(~expert_availability, float("inf"))
        oracle_expert_target = expert_errors.argmin(dim=-1)
        expert_gate_loss = F.cross_entropy(
            output["gate_logits"][mask], oracle_expert_target
        )
        gate_loss = 0.5 * (gate_ab_loss + expert_gate_loss)
        consistency_a = 1.0 - (
            target_unit(expert_soft, period_a) * output["residue_unit_a"][mask]
        ).sum(dim=-1)
        consistency_b = 1.0 - (
            target_unit(expert_soft, period_b) * output["residue_unit_b"][mask]
        ).sum(dim=-1)
        availability_a_float = availability_a.to(dtype=consistency_a.dtype)
        availability_b_float = availability_b.to(dtype=consistency_b.dtype)
        consistency_loss = (
            consistency_a * availability_a_float + consistency_b * availability_b_float
        ).sum() / (availability_a_float + availability_b_float).sum().clamp(min=1.0)
        residual_loss = (output["residual"][mask] / max_delay_ns).square().mean()
        best_available_single_error = (
            single_errors.masked_fill(
                ~torch.stack([availability_a, availability_b], dim=-1),
                float("inf"),
            )
            .min(dim=-1)
            .values
        )
        regret_loss = F.relu(
            (expert_soft / max_delay_ns - normalized_target).abs()
            - best_available_single_error
        ).mean()
        target_loss = (
            expert_soft_loss
            + direct_weight * direct_loss
            + single_weight * single_loss
            + residue_weight * residue_loss
            + consistency_weight * consistency_loss
            + uncertainty_weight * uncertainty_loss
            + uncertainty_ranking_weight * uncertainty_ranking_loss
            + gate_weight * gate_loss
            + residual_weight * residual_loss
            + regret_weight * regret_loss
        )
        total = total + target_loss
        components[f"{target_name}_loss"] = float(target_loss.detach())
        components[f"{target_name}_expert_soft_loss"] = float(expert_soft_loss.detach())
        components[f"{target_name}_gate_loss"] = float(gate_loss.detach())
        components[f"{target_name}_uncertainty_ranking_loss"] = float(
            uncertainty_ranking_loss.detach()
        )
        components[f"{target_name}_regret_loss"] = float(regret_loss.detach())
    if not active_targets:
        raise ValueError("Batch has no valid delay targets.")
    return total / active_targets, components


def make_loaders(
    specs: list[PairSpec],
    *,
    max_delay_ns: float,
    max_delay_spread_ns: float | None,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    sample_cache: dict[str, tuple[list[object], dict[str, object]]],
) -> tuple[list[tuple[PairSpec, DataLoader]], dict[str, dict[str, int]]]:
    loaders = []
    pair_info = {}
    for index, spec in enumerate(specs):
        pairs, info = build_cached_pairs(
            spec,
            max_delay_spread_ns=max_delay_spread_ns,
            cache=sample_cache,
        )
        generator = torch.Generator().manual_seed(seed + index * 1009)
        loader = DataLoader(
            PairDataset(pairs),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=pair_collate(spec, max_delay_ns),
            num_workers=num_workers,
            generator=generator if shuffle else None,
        )
        loaders.append((spec, loader))
        pair_info[spec.name] = info
    return loaders, pair_info


def apply_view_dropout(batch: dict, probability: float) -> None:
    if probability <= 0.0:
        return
    batch_size = int(batch["period_a_ns"].shape[0])
    device = batch["period_a_ns"].device
    drop = torch.rand(batch_size, device=device) < probability
    drop_a = drop & (torch.rand(batch_size, device=device) < 0.5)
    drop_b = drop & ~drop_a
    batch["availability_a"] = ~drop_a
    batch["availability_b"] = ~drop_b


@torch.no_grad()
def evaluate_pair(
    model: PeriodConditionedDelayFusion,
    spec: PairSpec,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[dict], dict]:
    model.eval()
    collected = {
        target_name: {
            "target": [],
            "single_a": [],
            "single_b": [],
            "fused_direct": [],
            "fused_gate_ab_hard": [],
            "fused_expert_soft": [],
            "fused_expert_hard": [],
            "fused_confidence_fallback": [],
            "residue_a": [],
            "residue_b": [],
            "gate_weight_a": [],
            "gate_weight_b": [],
            "gate_weight_direct": [],
            "log_variance_a": [],
            "log_variance_b": [],
            "gate_confidence": [],
            "fallback_mask": [],
            "residual": [],
        }
        for target_name in TARGET_NAMES
    }
    for batch in loader:
        moved = move_nested(batch, device)
        outputs = model(
            moved["view_a"],
            moved["view_b"],
            moved["period_a_ns"],
            moved["period_b_ns"],
            moved["availability_a"],
            moved["availability_b"],
        )
        for target_name in TARGET_NAMES:
            mask = moved["masks"][target_name]
            if not bool(mask.any()):
                continue
            collected[target_name]["target"].append(
                moved["targets"][target_name][mask].cpu()
            )
            for key in (
                "single_a",
                "single_b",
                "fused_direct",
                "fused_gate_ab_hard",
                "fused_expert_soft",
                "fused_expert_hard",
                "fused_confidence_fallback",
                "residue_a",
                "residue_b",
                "gate_weight_a",
                "gate_weight_b",
                "gate_weight_direct",
                "log_variance_a",
                "log_variance_b",
                "gate_confidence",
                "fallback_mask",
                "residual",
            ):
                collected[target_name][key].append(
                    outputs[target_name][key][mask].cpu()
                )

    rows = []
    summary = {}
    methods = {
        "single_a": f"single_{spec.name_a}",
        "single_b": f"single_{spec.name_b}",
        "fused_direct": "paired_fused_direct",
        "fused_gate_ab_hard": "paired_gate_ab_hard",
        "fused_expert_soft": "paired_expert_soft",
        "fused_expert_hard": "paired_expert_hard",
        "fused_confidence_fallback": "paired_confidence_fallback",
    }
    for target_name in TARGET_NAMES:
        if not collected[target_name]["target"]:
            summary[target_name] = {"count": 0, "methods": {}}
            continue
        target = torch.cat(collected[target_name]["target"])
        target_summary = {"count": target.numel(), "methods": {}}
        for key, method_name in methods.items():
            prediction = torch.cat(collected[target_name][key])
            method_rows = metric_rows(
                field=target_name,
                method=method_name,
                predictions=prediction,
                targets=target,
            )
            for row in method_rows:
                row["pair"] = spec.name
            rows.extend(method_rows)
            target_summary["methods"][method_name] = method_rows[0]
        target_summary["final_output_method"] = FINAL_OUTPUT_METHOD
        target_summary["final_metrics"] = target_summary["methods"][FINAL_OUTPUT_METHOD]
        target_summary["residue_a_circular_MAE_ns"] = circular_mae(
            torch.cat(collected[target_name]["residue_a"]),
            target,
            spec.period_a_ns,
        )
        target_summary["residue_b_circular_MAE_ns"] = circular_mae(
            torch.cat(collected[target_name]["residue_b"]),
            target,
            spec.period_b_ns,
        )
        single_a = torch.cat(collected[target_name]["single_a"])
        single_b = torch.cat(collected[target_name]["single_b"])
        fused_direct = torch.cat(collected[target_name]["fused_direct"])
        fused_gate_ab_hard = torch.cat(collected[target_name]["fused_gate_ab_hard"])
        fused_expert_soft = torch.cat(collected[target_name]["fused_expert_soft"])
        fused_expert_hard = torch.cat(collected[target_name]["fused_expert_hard"])
        fused_confidence_fallback = torch.cat(
            collected[target_name]["fused_confidence_fallback"]
        )
        gate_weight_a = torch.cat(collected[target_name]["gate_weight_a"])
        gate_weight_b = torch.cat(collected[target_name]["gate_weight_b"])
        gate_weight_direct = torch.cat(collected[target_name]["gate_weight_direct"])
        log_variance_a = torch.cat(collected[target_name]["log_variance_a"])
        log_variance_b = torch.cat(collected[target_name]["log_variance_b"])
        gate_confidence = torch.cat(collected[target_name]["gate_confidence"])
        fallback_mask = torch.cat(collected[target_name]["fallback_mask"]).bool()
        oracle_select_a = (single_a - target).abs() <= (single_b - target).abs()
        gate_select_a = gate_weight_a >= gate_weight_b
        expert_errors = torch.stack(
            [
                (single_a - target).abs(),
                (single_b - target).abs(),
                (fused_direct - target).abs(),
            ],
            dim=-1,
        )
        oracle_expert = expert_errors.argmin(dim=-1)
        selected_expert = torch.stack(
            [gate_weight_a, gate_weight_b, gate_weight_direct], dim=-1
        ).argmax(dim=-1)
        best_single_error = torch.minimum(
            (single_a - target).abs(), (single_b - target).abs()
        )
        target_summary["gate_ab_oracle_selection_accuracy"] = float(
            (gate_select_a == oracle_select_a).float().mean()
        )
        target_summary["uncertainty_order_accuracy"] = float(
            ((log_variance_a <= log_variance_b) == oracle_select_a).float().mean()
        )
        target_summary["expert_gate_oracle_selection_accuracy"] = float(
            (selected_expert == oracle_expert).float().mean()
        )
        target_summary["gate_mean_weight_a"] = float(gate_weight_a.mean())
        target_summary["gate_mean_weight_b"] = float(gate_weight_b.mean())
        target_summary["gate_mean_weight_direct"] = float(gate_weight_direct.mean())
        target_summary["gate_ab_hard_beats_best_single_rate"] = float(
            ((fused_gate_ab_hard - target).abs() <= best_single_error).float().mean()
        )
        target_summary["expert_soft_beats_best_single_rate"] = float(
            ((fused_expert_soft - target).abs() <= best_single_error).float().mean()
        )
        target_summary["expert_hard_beats_best_single_rate"] = float(
            ((fused_expert_hard - target).abs() <= best_single_error).float().mean()
        )
        target_summary["confidence_fallback_beats_best_single_rate"] = float(
            ((fused_confidence_fallback - target).abs() <= best_single_error)
            .float()
            .mean()
        )
        target_summary["gate_mean_confidence"] = float(gate_confidence.mean())
        target_summary["confidence_fallback_rate"] = float(fallback_mask.float().mean())
        target_summary["expert_residual_abs_mean_ns"] = float(
            torch.cat(collected[target_name]["residual"]).abs().mean()
        )
        summary[target_name] = target_summary
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    pair_metavar = (
        "NAME_A",
        "PERIOD_A_NS",
        "PATH_A",
        "NAME_B",
        "PERIOD_B_NS",
        "PATH_B",
    )
    parser.add_argument(
        "--train-pair",
        action="append",
        nargs=6,
        metavar=pair_metavar,
        required=True,
        help="Repeat for every training numerology pair.",
    )
    parser.add_argument(
        "--test-pair",
        action="append",
        nargs=6,
        metavar=pair_metavar,
        required=True,
        help="Repeat for every held-out test pair.",
    )
    parser.add_argument(
        "--held-out-name",
        help="Assert that this nf is absent from every training pair and present in every test pair.",
    )
    parser.add_argument("--max-delay-ns", type=float, default=1920.0)
    parser.add_argument("--max-delay-spread-ns", type=float, default=400.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--single-weight", type=float, default=1.0)
    parser.add_argument("--residue-weight", type=float, default=0.25)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--direct-weight", type=float, default=0.25)
    parser.add_argument("--uncertainty-weight", type=float, default=0.05)
    parser.add_argument("--uncertainty-ranking-weight", type=float, default=0.0)
    parser.add_argument("--uncertainty-ranking-margin", type=float, default=0.25)
    parser.add_argument("--uncertainty-ranking-min-gap-ns", type=float, default=10.0)
    parser.add_argument("--gate-weight", type=float, default=0.1)
    parser.add_argument("--period-prior-strength", type=float, default=0.0)
    parser.add_argument("--fallback-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--regret-weight", type=float, default=0.25)
    parser.add_argument("--residual-scale-ns", type=float, default=50.0)
    parser.add_argument("--view-dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive.")
    if not 0.0 <= args.view_dropout < 1.0:
        raise ValueError("--view-dropout must be in [0, 1).")
    if args.residual_scale_ns < 0.0:
        raise ValueError("--residual-scale-ns must be nonnegative.")
    if args.uncertainty_ranking_weight < 0.0:
        raise ValueError("--uncertainty-ranking-weight must be nonnegative.")
    if args.uncertainty_ranking_margin < 0.0:
        raise ValueError("--uncertainty-ranking-margin must be nonnegative.")
    if args.uncertainty_ranking_min_gap_ns < 0.0:
        raise ValueError("--uncertainty-ranking-min-gap-ns must be nonnegative.")
    if args.period_prior_strength < 0.0:
        raise ValueError("--period-prior-strength must be nonnegative.")
    if not 0.0 <= args.fallback_confidence_threshold <= 1.0:
        raise ValueError("--fallback-confidence-threshold must be in [0, 1].")
    train_specs = [parse_pair(values) for values in args.train_pair]
    test_specs = [parse_pair(values) for values in args.test_pair]
    if len({spec.name for spec in train_specs}) != len(train_specs):
        raise ValueError("Duplicate training pair names are not allowed.")
    if len({spec.name for spec in test_specs}) != len(test_specs):
        raise ValueError("Duplicate test pair names are not allowed.")
    if args.held_out_name:
        held_out = args.held_out_name
        if any(held_out in (spec.name_a, spec.name_b) for spec in train_specs):
            raise ValueError(f"Held-out nf {held_out!r} appears in a training pair.")
        if any(held_out not in (spec.name_a, spec.name_b) for spec in test_specs):
            raise ValueError(f"Every test pair must contain held-out nf {held_out!r}.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_cache: dict[str, tuple[list[object], dict[str, object]]] = {}
    train_loaders, train_info = make_loaders(
        train_specs,
        max_delay_ns=args.max_delay_ns,
        max_delay_spread_ns=args.max_delay_spread_ns,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        shuffle=True,
        sample_cache=sample_cache,
    )
    model = PeriodConditionedDelayFusion(
        max_delay_ns=args.max_delay_ns,
        hidden_dim=args.hidden_dim,
        residual_scale_ns=args.residual_scale_ns,
        period_prior_strength=args.period_prior_strength,
        fallback_confidence_threshold=args.fallback_confidence_threshold,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    print(f"seed={args.seed}")
    print(f"held_out_name={args.held_out_name or 'none'}")
    print(f"train_pairs={','.join(spec.name for spec in train_specs)}")
    print(f"test_pairs={','.join(spec.name for spec in test_specs)}")
    print(f"final_output_method={FINAL_OUTPUT_METHOD}")
    print(
        "calibration_config="
        + json.dumps(
            {
                "uncertainty_ranking_weight": args.uncertainty_ranking_weight,
                "uncertainty_ranking_margin": args.uncertainty_ranking_margin,
                "uncertainty_ranking_min_gap_ns": args.uncertainty_ranking_min_gap_ns,
                "gate_weight": args.gate_weight,
                "period_prior_strength": args.period_prior_strength,
                "fallback_confidence_threshold": args.fallback_confidence_threshold,
            },
            sort_keys=True,
        )
    )
    for pair_name, info in train_info.items():
        print(f"train_pair_info_{pair_name}={json.dumps(info, sort_keys=True)}")
    print(
        f"model_parameters={sum(parameter.numel() for parameter in model.parameters())}"
    )

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        component_sums: dict[str, float] = {}
        pair_order = list(train_loaders)
        random.shuffle(pair_order)
        for _, loader in pair_order:
            for batch in loader:
                moved = move_nested(batch, device)
                apply_view_dropout(moved, args.view_dropout)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    moved["view_a"],
                    moved["view_b"],
                    moved["period_a_ns"],
                    moved["period_b_ns"],
                    moved["availability_a"],
                    moved["availability_b"],
                )
                loss, components = compute_loss(
                    outputs,
                    moved,
                    max_delay_ns=args.max_delay_ns,
                    single_weight=args.single_weight,
                    residue_weight=args.residue_weight,
                    consistency_weight=args.consistency_weight,
                    direct_weight=args.direct_weight,
                    uncertainty_weight=args.uncertainty_weight,
                    uncertainty_ranking_weight=args.uncertainty_ranking_weight,
                    uncertainty_ranking_margin=args.uncertainty_ranking_margin,
                    uncertainty_ranking_min_gap_ns=args.uncertainty_ranking_min_gap_ns,
                    gate_weight=args.gate_weight,
                    residual_weight=args.residual_weight,
                    regret_weight=args.regret_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                total_loss += float(loss.detach())
                total_batches += 1
                for key, value in components.items():
                    component_sums[key] = component_sums.get(key, 0.0) + value
        component_text = " ".join(
            f"{key}={value / max(total_batches, 1):.6f}"
            for key, value in sorted(component_sums.items())
        )
        print(
            f"epoch={epoch:03d}/{args.epochs} steps={total_batches} "
            f"loss={total_loss / max(total_batches, 1):.6f} {component_text}",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start
    del train_loaders, pair_order, loader
    sample_cache.clear()
    gc.collect()
    test_loaders, test_info = make_loaders(
        test_specs,
        max_delay_ns=args.max_delay_ns,
        max_delay_spread_ns=args.max_delay_spread_ns,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        shuffle=False,
        sample_cache=sample_cache,
    )
    for pair_name, info in test_info.items():
        print(f"test_pair_info_{pair_name}={json.dumps(info, sort_keys=True)}")
    all_rows = []
    all_summary = {}
    for spec, loader in test_loaders:
        rows, summary = evaluate_pair(model, spec, loader, device)
        all_rows.extend(rows)
        all_summary[spec.name] = summary
        for target_name, target_summary in summary.items():
            print(f"{spec.name}_{target_name}_count={target_summary['count']}")
            for method, metrics in target_summary["methods"].items():
                print(
                    f"{spec.name}_{target_name}_{method}_MAE="
                    f"{float(metrics['MAE']):.4f}"
                )
                print(
                    f"{spec.name}_{target_name}_{method}_accuracy_at_50ns="
                    f"{float(metrics['accuracy_at_50ns']):.4f}"
                )
            final_metrics = target_summary["final_metrics"]
            print(
                f"{spec.name}_{target_name}_final_MAE="
                f"{float(final_metrics['MAE']):.4f}"
            )
            print(
                f"{spec.name}_{target_name}_final_accuracy_at_50ns="
                f"{float(final_metrics['accuracy_at_50ns']):.4f}"
            )
            for diagnostic in (
                "gate_ab_oracle_selection_accuracy",
                "uncertainty_order_accuracy",
                "expert_gate_oracle_selection_accuracy",
                "gate_mean_weight_a",
                "gate_mean_weight_b",
                "gate_mean_weight_direct",
                "gate_ab_hard_beats_best_single_rate",
                "expert_soft_beats_best_single_rate",
                "expert_hard_beats_best_single_rate",
                "confidence_fallback_beats_best_single_rate",
                "gate_mean_confidence",
                "confidence_fallback_rate",
                "expert_residual_abs_mean_ns",
            ):
                print(
                    f"{spec.name}_{target_name}_{diagnostic}="
                    f"{float(target_summary[diagnostic]):.4f}"
                )

    checkpoint_path = args.output_dir / "multinumerology_generalization.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "train_info": train_info,
            "test_info": test_info,
            "summary": all_summary,
            "final_output_key": FINAL_OUTPUT_KEY,
            "final_output_method": FINAL_OUTPUT_METHOD,
            "training_elapsed_seconds": elapsed_seconds,
        },
        checkpoint_path,
    )
    metrics_path = args.output_dir / "multinumerology_generalization_test_metrics.csv"
    fieldnames = [
        "pair",
        "field",
        "method",
        "target_range",
        "count",
        "MAE",
        "RMSE",
        "signed_mean",
        "pearson",
        "accuracy_at_50ns",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = args.output_dir / "multinumerology_generalization_test_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "train_info": train_info,
                "test_info": test_info,
                "summary": all_summary,
                "final_output_key": FINAL_OUTPUT_KEY,
                "final_output_method": FINAL_OUTPUT_METHOD,
                "training_elapsed_seconds": elapsed_seconds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"training_elapsed_seconds={elapsed_seconds:.2f}")
    print(f"saved_checkpoint={checkpoint_path}")
    print(f"saved_metrics_csv={metrics_path}")
    print(f"saved_summary_json={summary_path}")


if __name__ == "__main__":
    main()
