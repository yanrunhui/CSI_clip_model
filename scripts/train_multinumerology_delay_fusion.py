from __future__ import annotations

import argparse
import csv
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

from data.dataset import (  # noqa: E402
    CONFIG_FEATURE_DIM,
    PreprocessedCSIDataset,
    _sample_configuration_features,
)
from models.model import CSIArrayInvariantDelayEncoder, CSI_DELAY_CONTEXT_DIM  # noqa: E402
from scripts.diagnose_multinumerology_delay_fusion import (  # noqa: E402
    metric_rows,
)


TARGET_NAMES = ("first_path_delay_ns", "los_delay_ns")


@dataclass(frozen=True)
class PairedSample:
    sample_a: object
    sample_b: object


class PairedDataset(Dataset[PairedSample]):
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


def finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def target_value(sample, target_name: str) -> float | None:
    if target_name == "first_path_delay_ns":
        value = finite_float(getattr(sample, "first_path_delay_s", math.nan))
    elif target_name == "los_delay_ns":
        value = finite_float(getattr(sample, "los_delay_s", math.nan))
    else:
        raise ValueError(f"Unsupported target: {target_name}")
    return None if value is None else value * 1e9


def samples_by_group(path: str) -> tuple[list[object], dict[str, object]]:
    samples = PreprocessedCSIDataset.from_pt(path).samples
    grouped = {}
    for sample in samples:
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError(f"Sample without group_id in {path}.")
        if group_id in grouped:
            raise ValueError(f"Duplicate group_id={group_id!r} in {path}.")
        grouped[group_id] = sample
    return samples, grouped


def build_pairs(
    path_a: str,
    path_b: str,
    *,
    max_delay_spread_ns: float | None,
    target_tolerance_ns: float = 1e-3,
) -> tuple[list[PairedSample], dict[str, int]]:
    samples_a, grouped_a = samples_by_group(path_a)
    _, grouped_b = samples_by_group(path_b)
    pairs = []
    missing = 0
    label_mismatch = 0
    delay_spread_filtered = 0
    for sample_a in samples_a:
        group_id = str(sample_a.group_id)
        sample_b = grouped_b.get(group_id)
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
            delay_spread_ns = finite_float(
                getattr(sample_a, "delay_spread_s", math.nan)
            )
            delay_spread_ns = (
                math.nan if delay_spread_ns is None else delay_spread_ns * 1e9
            )
            if not math.isfinite(delay_spread_ns) or delay_spread_ns >= max_delay_spread_ns:
                delay_spread_filtered += 1
                continue
        pairs.append(PairedSample(sample_a=sample_a, sample_b=sample_b))

    if label_mismatch:
        raise ValueError(
            f"Found {label_mismatch} paired samples with mismatched delay labels."
        )
    if not pairs:
        raise ValueError("No valid paired samples remain.")
    return pairs, {
        "input_a_count": len(samples_a),
        "input_b_count": len(grouped_b),
        "paired_count": len(pairs),
        "missing_count": missing,
        "delay_spread_filtered_count": delay_spread_filtered,
    }


def pack_view(samples: list[object]) -> dict[str, torch.Tensor]:
    batch_size = len(samples)
    max_tokens = max(int(sample.n_tokens) for sample in samples)
    d_token = int(samples[0].tokens.shape[1])
    n_freq = int(samples[0].tokens.shape[2])
    tokens = torch.zeros(
        batch_size,
        max_tokens,
        d_token,
        n_freq,
        dtype=torch.float32,
    )
    token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    config_features = torch.zeros(
        batch_size,
        CONFIG_FEATURE_DIM,
        dtype=torch.float32,
    )
    subcarrier_spacing = torch.zeros(batch_size, dtype=torch.float32)
    for index, sample in enumerate(samples):
        count = int(sample.n_tokens)
        tokens[index, :count] = sample.tokens.to(dtype=torch.float32)
        token_mask[index, :count] = True
        config_features[index] = _sample_configuration_features(sample)
        subcarrier_spacing[index] = float(sample.subcarrier_spacing_hz)
    return {
        "tokens": tokens,
        "token_mask": token_mask,
        "config_features": config_features,
        "subcarrier_spacing": subcarrier_spacing,
    }


def make_collate(max_delay_ns: float, period_a_ns: float):
    def collate(batch: list[PairedSample]) -> dict:
        samples_a = [pair.sample_a for pair in batch]
        samples_b = [pair.sample_b for pair in batch]
        targets = {
            target_name: torch.zeros(len(batch), dtype=torch.float32)
            for target_name in TARGET_NAMES
        }
        masks = {
            target_name: torch.zeros(len(batch), dtype=torch.bool)
            for target_name in TARGET_NAMES
        }
        cycle_targets = {
            target_name: torch.zeros(len(batch), dtype=torch.long)
            for target_name in TARGET_NAMES
        }
        for index, pair in enumerate(batch):
            for target_name in TARGET_NAMES:
                value = target_value(pair.sample_a, target_name)
                if value is None or value < 0.0 or value >= max_delay_ns:
                    continue
                targets[target_name][index] = value
                masks[target_name][index] = True
                cycle_targets[target_name][index] = int(value // period_a_ns)
        return {
            "view_a": pack_view(samples_a),
            "view_b": pack_view(samples_b),
            "targets": targets,
            "masks": masks,
            "cycle_targets": cycle_targets,
            "group_ids": [str(pair.sample_a.group_id) for pair in batch],
        }

    return collate


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


class MultiNumerologyDelayFusion(nn.Module):
    def __init__(
        self,
        *,
        period_a_ns: float,
        period_b_ns: float,
        max_delay_ns: float,
        hidden_dim: int,
    ):
        super().__init__()
        self.period_a_ns = float(period_a_ns)
        self.period_b_ns = float(period_b_ns)
        self.max_delay_ns = float(max_delay_ns)
        self.cycle_count = int(math.ceil(max_delay_ns / period_a_ns))
        self.encoder = CSIArrayInvariantDelayEncoder(out_dim=CSI_DELAY_CONTEXT_DIM)

        self.single_a_heads = nn.ModuleDict(
            {
                name: PredictionHead(CSI_DELAY_CONTEXT_DIM, 1, hidden_dim)
                for name in TARGET_NAMES
            }
        )
        self.single_b_heads = nn.ModuleDict(
            {
                name: PredictionHead(CSI_DELAY_CONTEXT_DIM, 1, hidden_dim)
                for name in TARGET_NAMES
            }
        )
        self.residue_a_heads = nn.ModuleDict(
            {
                name: PredictionHead(CSI_DELAY_CONTEXT_DIM, 2, hidden_dim)
                for name in TARGET_NAMES
            }
        )
        self.residue_b_heads = nn.ModuleDict(
            {
                name: PredictionHead(CSI_DELAY_CONTEXT_DIM, 2, hidden_dim)
                for name in TARGET_NAMES
            }
        )
        fusion_input_dim = CSI_DELAY_CONTEXT_DIM * 4 + 4
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
        self.direct_fused_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, 1) for name in TARGET_NAMES}
        )
        self.cycle_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, self.cycle_count) for name in TARGET_NAMES}
        )

    def encode_view(self, view: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(
            view["tokens"],
            view["token_mask"],
            subcarrier_spacing=view["subcarrier_spacing"],
            config_features=view["config_features"],
        )

    @staticmethod
    def normalized_unit(vector: torch.Tensor) -> torch.Tensor:
        return F.normalize(vector, dim=-1, eps=1e-6)

    @staticmethod
    def decode_residue(unit_vector: torch.Tensor, period_ns: float) -> torch.Tensor:
        angle = torch.atan2(unit_vector[:, 0], unit_vector[:, 1])
        fraction = torch.remainder(angle / (2.0 * math.pi), 1.0)
        return fraction * period_ns

    def forward(self, view_a: dict[str, torch.Tensor], view_b: dict[str, torch.Tensor]) -> dict:
        context_a = self.encode_view(view_a)
        context_b = self.encode_view(view_b)
        outputs = {}
        cycle_values = torch.arange(
            self.cycle_count,
            device=context_a.device,
            dtype=context_a.dtype,
        )
        for target_name in TARGET_NAMES:
            residue_unit_a = self.normalized_unit(
                self.residue_a_heads[target_name](context_a)
            )
            residue_unit_b = self.normalized_unit(
                self.residue_b_heads[target_name](context_b)
            )
            fusion_input = torch.cat(
                [
                    context_a,
                    context_b,
                    (context_a - context_b).abs(),
                    context_a * context_b,
                    residue_unit_a,
                    residue_unit_b,
                ],
                dim=-1,
            )
            fused_features = self.fusion_trunks[target_name](fusion_input)
            cycle_logits = self.cycle_heads[target_name](fused_features)
            cycle_probabilities = torch.softmax(cycle_logits, dim=-1)
            soft_cycle = (cycle_probabilities * cycle_values.unsqueeze(0)).sum(dim=-1)
            hard_cycle = cycle_logits.argmax(dim=-1).to(dtype=context_a.dtype)
            residue_a = self.decode_residue(residue_unit_a, self.period_a_ns)
            outputs[target_name] = {
                "single_a": self.single_a_heads[target_name](context_a).squeeze(-1)
                * self.max_delay_ns,
                "single_b": self.single_b_heads[target_name](context_b).squeeze(-1)
                * self.max_delay_ns,
                "residue_unit_a": residue_unit_a,
                "residue_unit_b": residue_unit_b,
                "residue_a": residue_a,
                "residue_b": self.decode_residue(residue_unit_b, self.period_b_ns),
                "cycle_logits": cycle_logits,
                "fused_direct": self.direct_fused_heads[target_name](fused_features).squeeze(-1)
                * self.max_delay_ns,
                "fused_cycle_soft": residue_a + soft_cycle * self.period_a_ns,
                "fused_cycle_hard": residue_a + hard_cycle * self.period_a_ns,
            }
        return outputs


def target_unit(target: torch.Tensor, period_ns: float) -> torch.Tensor:
    angle = torch.remainder(target, period_ns) / period_ns * (2.0 * math.pi)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def move_nested(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_nested(item, device) for key, item in value.items()}
    return value


def compute_loss(
    outputs: dict,
    batch: dict,
    *,
    period_a_ns: float,
    period_b_ns: float,
    max_delay_ns: float,
    single_weight: float,
    residue_weight: float,
    cycle_classifier_weight: float,
    cycle_regression_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = next(iter(outputs.values()))["single_a"].new_zeros(())
    metrics = {}
    active_targets = 0
    for target_name in TARGET_NAMES:
        mask = batch["masks"][target_name]
        if not bool(mask.any()):
            continue
        active_targets += 1
        target = batch["targets"][target_name][mask]
        normalized_target = target / max_delay_ns
        output = outputs[target_name]
        single_loss = 0.5 * (
            F.smooth_l1_loss(output["single_a"][mask] / max_delay_ns, normalized_target)
            + F.smooth_l1_loss(output["single_b"][mask] / max_delay_ns, normalized_target)
        )
        residue_loss = 0.5 * (
            (1.0 - (output["residue_unit_a"][mask] * target_unit(target, period_a_ns)).sum(dim=-1)).mean()
            + (1.0 - (output["residue_unit_b"][mask] * target_unit(target, period_b_ns)).sum(dim=-1)).mean()
        )
        direct_loss = F.smooth_l1_loss(
            output["fused_direct"][mask] / max_delay_ns,
            normalized_target,
        )
        cycle_classifier_loss = F.cross_entropy(
            output["cycle_logits"][mask],
            batch["cycle_targets"][target_name][mask],
        )
        cycle_regression_loss = F.smooth_l1_loss(
            output["fused_cycle_soft"][mask] / max_delay_ns,
            normalized_target,
        )
        target_loss = (
            direct_loss
            + single_weight * single_loss
            + residue_weight * residue_loss
            + cycle_classifier_weight * cycle_classifier_loss
            + cycle_regression_weight * cycle_regression_loss
        )
        total = total + target_loss
        metrics[f"{target_name}_loss"] = float(target_loss.detach())
    if not active_targets:
        raise ValueError("Batch has no valid delay targets.")
    return total / active_targets, metrics


def circular_mae(prediction: torch.Tensor, target: torch.Tensor, period_ns: float) -> float:
    difference = torch.remainder(prediction - target + period_ns / 2.0, period_ns) - period_ns / 2.0
    return float(difference.abs().mean())


@torch.no_grad()
def evaluate_model(
    model: MultiNumerologyDelayFusion,
    loader: DataLoader,
    device: torch.device,
    name_a: str,
    name_b: str,
) -> tuple[list[dict], dict]:
    model.eval()
    collected = {
        target_name: {
            "target": [],
            "single_a": [],
            "single_b": [],
            "fused_direct": [],
            "fused_cycle_soft": [],
            "fused_cycle_hard": [],
            "residue_a": [],
            "residue_b": [],
        }
        for target_name in TARGET_NAMES
    }
    for batch in loader:
        moved = move_nested(batch, device)
        outputs = model(moved["view_a"], moved["view_b"])
        for target_name in TARGET_NAMES:
            mask = moved["masks"][target_name]
            if not bool(mask.any()):
                continue
            collected[target_name]["target"].append(
                moved["targets"][target_name][mask].cpu()
            )
            for method in collected[target_name]:
                if method == "target":
                    continue
                collected[target_name][method].append(outputs[target_name][method][mask].cpu())

    rows = []
    summary = {}
    method_labels = {
        "single_a": name_a,
        "single_b": name_b,
        "fused_direct": "paired_fused_direct",
        "fused_cycle_soft": "paired_fused_cycle_soft",
        "fused_cycle_hard": "paired_fused_cycle_hard",
    }
    for target_name in TARGET_NAMES:
        if not collected[target_name]["target"]:
            summary[target_name] = {"count": 0, "methods": {}}
            continue
        target = torch.cat(collected[target_name]["target"])
        target_summary = {"count": target.numel(), "methods": {}}
        for key, method_label in method_labels.items():
            prediction = torch.cat(collected[target_name][key])
            method_rows = metric_rows(
                field=target_name,
                method=method_label,
                predictions=prediction,
                targets=target,
            )
            rows.extend(method_rows)
            target_summary["methods"][method_label] = method_rows[0]
        residue_a = torch.cat(collected[target_name]["residue_a"])
        residue_b = torch.cat(collected[target_name]["residue_b"])
        target_summary["residue_a_circular_MAE_ns"] = circular_mae(
            residue_a,
            target,
            model.period_a_ns,
        )
        target_summary["residue_b_circular_MAE_ns"] = circular_mae(
            residue_b,
            target,
            model.period_b_ns,
        )
        summary[target_name] = target_summary
    return rows, summary


def initialize_encoder_from_checkpoint(model: MultiNumerologyDelayFusion, path: str) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    prefixes = (
        "first_path_delay_context_encoder.",
        "csi_delay_context_encoder.",
    )
    model_state = model.encoder.state_dict()
    for prefix in prefixes:
        compatible = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
            and key[len(prefix) :] in model_state
            and model_state[key[len(prefix) :]].shape == value.shape
        }
        if compatible:
            model.encoder.load_state_dict(compatible, strict=False)
            return len(compatible)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-a", required=True)
    parser.add_argument("--train-b", required=True)
    parser.add_argument("--test-a", required=True)
    parser.add_argument("--test-b", required=True)
    parser.add_argument("--name-a", default="nf64")
    parser.add_argument("--name-b", default="nf96")
    parser.add_argument("--period-a-ns", type=float, default=640.0)
    parser.add_argument("--period-b-ns", type=float, default=960.0)
    parser.add_argument("--max-delay-ns", type=float, default=1920.0)
    parser.add_argument("--max-delay-spread-ns", type=float, default=400.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--single-weight", type=float, default=0.25)
    parser.add_argument("--residue-weight", type=float, default=0.25)
    parser.add_argument("--cycle-classifier-weight", type=float, default=0.25)
    parser.add_argument("--cycle-regression-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive.")
    if args.max_delay_ns <= 0.0:
        raise ValueError("--max-delay-ns must be positive.")
    cycle_count = int(math.ceil(args.max_delay_ns / args.period_a_ns))
    if cycle_count <= 1:
        raise ValueError("Fusion range must contain more than one period-A cycle.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pairs, train_info = build_pairs(
        args.train_a,
        args.train_b,
        max_delay_spread_ns=args.max_delay_spread_ns,
    )
    test_pairs, test_info = build_pairs(
        args.test_a,
        args.test_b,
        max_delay_spread_ns=args.max_delay_spread_ns,
    )
    collate = make_collate(args.max_delay_ns, args.period_a_ns)
    train_loader = DataLoader(
        PairedDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        PairedDataset(test_pairs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    model = MultiNumerologyDelayFusion(
        period_a_ns=args.period_a_ns,
        period_b_ns=args.period_b_ns,
        max_delay_ns=args.max_delay_ns,
        hidden_dim=args.hidden_dim,
    )
    initialized_keys = (
        initialize_encoder_from_checkpoint(model, args.checkpoint)
        if args.checkpoint
        else 0
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    print(f"train_pairs={len(train_pairs)}")
    print(f"test_pairs={len(test_pairs)}")
    print(f"encoder_initialized_checkpoint_keys={initialized_keys}")
    print(f"period_a_ns={args.period_a_ns:g}")
    print(f"period_b_ns={args.period_b_ns:g}")
    print(f"max_delay_ns={args.max_delay_ns:g}")
    print(f"model_parameters={sum(parameter.numel() for parameter in model.parameters())}")

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        component_sums = {}
        for batch in train_loader:
            moved = move_nested(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(moved["view_a"], moved["view_b"])
            loss, components = compute_loss(
                outputs,
                moved,
                period_a_ns=args.period_a_ns,
                period_b_ns=args.period_b_ns,
                max_delay_ns=args.max_delay_ns,
                single_weight=args.single_weight,
                residue_weight=args.residue_weight,
                cycle_classifier_weight=args.cycle_classifier_weight,
                cycle_regression_weight=args.cycle_regression_weight,
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
            f"epoch={epoch:03d}/{args.epochs} "
            f"loss={total_loss / max(total_batches, 1):.6f} {component_text}",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start
    rows, summary = evaluate_model(
        model,
        test_loader,
        device,
        args.name_a,
        args.name_b,
    )
    for target_name, target_summary in summary.items():
        print(f"{target_name}_count={target_summary['count']}")
        if not target_summary["count"]:
            continue
        print(
            f"{target_name}_{args.name_a}_residue_circular_MAE_ns="
            f"{target_summary['residue_a_circular_MAE_ns']:.4f}"
        )
        print(
            f"{target_name}_{args.name_b}_residue_circular_MAE_ns="
            f"{target_summary['residue_b_circular_MAE_ns']:.4f}"
        )
        for method, metrics in target_summary["methods"].items():
            print(f"{target_name}_{method}_MAE={float(metrics['MAE']):.4f}")
            print(
                f"{target_name}_{method}_accuracy_at_50ns="
                f"{float(metrics['accuracy_at_50ns']):.4f}"
            )

    checkpoint_path = args.output_dir / "multinumerology_delay_fusion.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "train_info": train_info,
            "test_info": test_info,
            "summary": summary,
            "training_elapsed_seconds": elapsed_seconds,
        },
        checkpoint_path,
    )
    csv_path = args.output_dir / "multinumerology_delay_fusion_test_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "multinumerology_delay_fusion_test_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "train_info": train_info,
                "test_info": test_info,
                "summary": summary,
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
    print(f"saved_metrics_csv={csv_path}")
    print(f"saved_summary_json={summary_path}")


if __name__ == "__main__":
    main()
