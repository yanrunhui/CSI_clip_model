from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate as eval_lib  # noqa: E402
import preprocess_all as preprocess_lib  # noqa: E402
from data.dataset import (  # noqa: E402
    PHYSICS_TARGET_NAMES,
    PHYSICS_TARGET_OFFSETS,
    PHYSICS_TARGET_SCALES,
    PreprocessedCSIDataset,
    apply_semantic_key_mode,
    collate_fn,
)
from data.semantic_key import (  # noqa: E402
    implied_attribute_value_filters,
)
from models.encoder import CSIEncoder  # noqa: E402
from models.model import CSIClip  # noqa: E402
from models.text_encoder import PhysicsTextEncoder  # noqa: E402
from scripts.pretrain import assert_checkpoint_prototype_compatibility  # noqa: E402


COUNT_FIELDS = (
    "reflection_path_count",
    "diffraction_path_count",
    "direct_path_count",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _histogram_text(values: torch.Tensor) -> str:
    if values.numel() == 0:
        return "none"
    counts = Counter(int(value) for value in values.long().tolist())
    return ",".join(f"{value}:{count}" for value, count in sorted(counts.items()))


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) == 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def _count_metrics(
    prefix: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, str | float | int]:
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    result: dict[str, str | float | int] = {
        "name": prefix,
        "count": int(valid.sum().item()),
    }
    if not bool(valid.any()):
        result.update(
            {
                "MAE": math.nan,
                "RMSE": math.nan,
                "signed_mean": math.nan,
                "exact_accuracy": math.nan,
                "within_1_accuracy": math.nan,
                "pearson": math.nan,
                "target_histogram": "none",
                "prediction_histogram": "none",
            }
        )
        return result
    pred = prediction[valid].float()
    tgt = target[valid].float()
    errors = pred - tgt
    rounded_pred = pred.round().clamp(min=0).long()
    rounded_tgt = tgt.round().clamp(min=0).long()
    result.update(
        {
            "MAE": float(errors.abs().mean()),
            "RMSE": float(torch.sqrt(errors.square().mean())),
            "signed_mean": float(errors.mean()),
            "exact_accuracy": float((rounded_pred == rounded_tgt).float().mean()),
            "within_1_accuracy": float(
                ((rounded_pred - rounded_tgt).abs() <= 1).float().mean()
            ),
            "pearson": _safe_pearson(pred, tgt),
            "target_histogram": _histogram_text(rounded_tgt),
            "prediction_histogram": _histogram_text(rounded_pred),
        }
    )
    return result


def _print_metric_row(row: dict[str, str | float | int]) -> None:
    name = row["name"]
    for key, value in row.items():
        if key == "name":
            continue
        if isinstance(value, float):
            text = "nan" if math.isnan(value) else f"{value:.6g}"
        else:
            text = str(value)
        print(f"{name}_{key}={text}")


def _write_results(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _preprocess_samples(args: argparse.Namespace):
    if args.data_path is not None:
        samples = PreprocessedCSIDataset.from_pt(args.data_path).samples
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        return samples

    d2los_root = Path(args.d2los_root) if args.d2los_root else None
    scenario_name = args.scenario
    if d2los_root is None and args.scenario:
        candidate_root = (
            Path(args.scenario_root) / args.scenario
            if args.scenario_root is not None
            else ROOT / "Raytracing_scenarios" / args.scenario
        )
        if preprocess_lib.is_d2los_root(candidate_root):
            d2los_root = candidate_root
            scenario_name = args.scenario

    if d2los_root is not None:
        raw_dataset = preprocess_lib.load_d2los_dataset(
            d2los_root=d2los_root,
            max_samples=args.max_samples,
            max_maps=args.max_maps,
            max_sources_per_map=args.max_sources_per_map,
            max_rx_per_source=args.max_rx_per_source,
            tx_shape=tuple(args.tx_shape),
            bandwidth_hz=args.bandwidth_hz,
            total_subcarriers=args.total_subcarriers,
            tx_power_dbm=args.tx_power_dbm,
            sampling=args.d2los_sampling,
            sample_seed=args.d2los_sample_seed,
        )
        scenario_name = scenario_name or d2los_root.name
    else:
        if args.scenario is None:
            raise SystemExit("Provide --data-path, --d2los-root, or --scenario.")
        raw_dataset = preprocess_lib.load_deepmimo_dataset(
            args.scenario,
            scenario_root=args.scenario_root,
            max_samples=args.max_samples,
        )

    samples = preprocess_lib.preprocess_deepmimo_dataset(
        dataset=raw_dataset,
        scenario=scenario_name or "D2Los_Data",
        freq_bin=args.freq_bin,
        rx_index=args.rx_index,
        env_type=args.env_type,
        max_samples=args.max_samples,
        patch_1d=args.patch_1d,
        patch_2d=tuple(args.patch_2d),
        target_nf=args.target_nf,
        include_empty_samples=args.include_empty_samples,
    )
    if args.save_samples is not None:
        output_path = Path(args.save_samples)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, output_path)
        print(f"saved_diagnostic_samples={output_path}")
    return samples


def _prepare_samples(samples, checkpoint: dict):
    semantic_key_mode = eval_lib._infer_semantic_key_mode(checkpoint, None)
    attribute_fields = eval_lib._infer_attribute_fields(checkpoint)
    attribute_remap = eval_lib._infer_attribute_remap(checkpoint)
    filter_attribute_values = eval_lib._infer_filter_attribute_values(checkpoint, None)
    filter_attribute_values = {
        **implied_attribute_value_filters(attribute_fields, attribute_remap),
        **filter_attribute_values,
    }
    samples = apply_semantic_key_mode(samples, semantic_key_mode)
    samples = eval_lib.filter_samples_by_min_class_size(
        samples,
        min_class_size=eval_lib._infer_min_class_size(checkpoint, None),
    )
    samples = eval_lib.filter_samples_by_attribute_values(samples, filter_attribute_values)
    samples = eval_lib.filter_samples_by_max_delay_spread(
        samples,
        eval_lib._infer_max_delay_spread_ns(checkpoint, None),
    )
    samples, checkpoint_prototype_keys = eval_lib.align_samples_to_checkpoint_prototypes(
        samples,
        checkpoint,
    )
    return samples, checkpoint_prototype_keys, attribute_fields, attribute_remap


def _build_model(
    samples,
    checkpoint: dict,
    checkpoint_prototype_keys,
    attribute_fields: tuple[str, ...],
    attribute_remap,
    device: torch.device,
):
    tokenizer = eval_lib.build_tokenizer(samples, checkpoint)
    prototype_keys, prototype_token_ids, prototype_token_mask, _ = eval_lib.build_prototype_bank(
        samples,
        tokenizer,
        prototype_keys_override=checkpoint_prototype_keys,
    )
    attribute_label_maps = eval_lib.build_attribute_label_maps(
        samples,
        attribute_fields,
        attribute_remap=attribute_remap,
    )
    token_norm_mode = eval_lib._infer_token_norm_mode(checkpoint, None)
    use_power_branch = eval_lib._infer_use_power_branch(checkpoint, None)
    first_path_power_mode = eval_lib._infer_first_path_power_mode(checkpoint)
    first_path_power_use_internal_gate = (
        eval_lib._infer_first_path_power_use_internal_gate(checkpoint)
    )
    use_delay_spread_head = eval_lib._infer_use_delay_spread_head(checkpoint)
    use_delay_specific_encoder = eval_lib._infer_use_delay_specific_encoder(checkpoint)
    use_los_angle_context_encoder = eval_lib._infer_use_los_angle_context_encoder(checkpoint)
    use_first_path_angle_context_encoder = (
        eval_lib._infer_use_first_path_angle_context_encoder(checkpoint)
    )
    use_shared_physics_token = eval_lib._infer_use_shared_physics_token(checkpoint)
    shared_physics_token_residual_scale = (
        eval_lib._infer_shared_physics_token_residual_scale(checkpoint)
    )

    model = CSIClip(
        CSIEncoder(
            d_token=8,
            d_model=384,
            d_clip=256,
            token_norm_mode=token_norm_mode,
        ),
        PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300)),
        num_prototypes=len(prototype_keys),
        semantic_num_classes=len(prototype_keys),
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        use_power_branch=use_power_branch,
        first_path_power_mode=first_path_power_mode,
        first_path_power_use_internal_gate=first_path_power_use_internal_gate,
        use_delay_spread_head=use_delay_spread_head,
        use_delay_specific_encoder=use_delay_specific_encoder,
        use_los_angle_context_encoder=use_los_angle_context_encoder,
        use_first_path_angle_context_encoder=use_first_path_angle_context_encoder,
        los_angle_context_token_norm_mode=token_norm_mode,
        use_shared_physics_token=use_shared_physics_token,
        shared_physics_token_residual_scale=shared_physics_token_residual_scale,
        attribute_num_classes={
            field: len(label_map)
            for field, label_map in attribute_label_maps.items()
        },
    ).to(device)
    eval_lib._assert_checkpoint_first_path_delay_bin_labels(checkpoint)
    assert_checkpoint_prototype_compatibility(
        checkpoint,
        prototype_keys,
        expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
        context="reflection path count diagnostic checkpoint",
    )
    eval_lib._load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def _collect_predictions(
    model: CSIClip,
    samples,
    tokenizer,
    batch_size: int,
    device: torch.device,
):
    loader = DataLoader(
        PreprocessedCSIDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
    use_power_branch = bool(getattr(model, "use_power_branch", False))
    all_features = []
    all_final = []
    all_base = []
    all_reflection_head = []
    all_reflection_path_head = []
    for batch in loader:
        batch = eval_lib.move_batch(batch, device)
        features_raw = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
        )
        delay_context = None
        first_path_delay_context = None
        los_angle_context = None
        first_path_angle_context = None
        if hasattr(model, "encode_csi_delay_context"):
            delay_context = model.encode_csi_delay_context(
                batch["tokens"],
                batch["token_mask"],
                subcarrier_spacing=batch.get("subcarrier_spacing"),
            )
        if hasattr(model, "encode_first_path_delay_context"):
            first_path_delay_context = model.encode_first_path_delay_context(
                batch["tokens"],
                batch["token_mask"],
                beam_positions=batch.get("beam_positions"),
                freq_bin=batch.get("freq_bin"),
                bw_bin=batch.get("bw_bin"),
                subcarrier_spacing=batch.get("subcarrier_spacing"),
            )
        if hasattr(model, "encode_los_angle_context"):
            los_angle_context = model.encode_los_angle_context(
                batch["tokens"],
                batch["beam_positions"],
                batch["token_mask"],
                batch["freq_bin"],
                batch["bw_bin"],
                batch["subcarrier_spacing"],
            )
        if hasattr(model, "encode_first_path_angle_context"):
            first_path_angle_context = model.encode_first_path_angle_context(
                batch["tokens"],
                batch["beam_positions"],
                batch["token_mask"],
                subcarrier_spacing=batch.get("subcarrier_spacing"),
            )
        power_context = None
        if use_power_branch:
            power_context = model.encode_power_context(
                batch["tokens"],
                batch["token_mask"],
                delay_power_map=batch.get("delay_power_map"),
                delay_power_profile=batch.get("delay_power_profile"),
            )
        outputs = model.predict_physics_components(
            features_raw,
            power_context=power_context,
            delay_context=delay_context,
            first_path_delay_context=first_path_delay_context,
            los_angle_context=los_angle_context,
            first_path_angle_context=first_path_angle_context,
        )
        all_features.append(F.normalize(features_raw, dim=-1).cpu())
        all_final.append(outputs["final"].cpu())
        all_base.append(outputs["base"].cpu())
        all_reflection_head.append(outputs["reflection_count_prediction"].cpu())
        all_reflection_path_head.append(outputs["reflection_path_count_prediction"].cpu())

    final = torch.cat(all_final, dim=0)
    base = torch.cat(all_base, dim=0)
    reflection_head = torch.cat(all_reflection_head, dim=0)
    reflection_path_head = torch.cat(all_reflection_path_head, dim=0)
    features = torch.cat(all_features, dim=0)
    return features, final, base, reflection_head, reflection_path_head


def _raw_physics(predictions: torch.Tensor) -> torch.Tensor:
    return predictions * PHYSICS_TARGET_SCALES + PHYSICS_TARGET_OFFSETS


def _raw_reflection_head(predictions: torch.Tensor) -> torch.Tensor:
    idx = PHYSICS_TARGET_NAMES.index("reflection_count")
    return predictions * PHYSICS_TARGET_SCALES[idx] + PHYSICS_TARGET_OFFSETS[idx]


def _majority_baseline(target: torch.Tensor) -> torch.Tensor:
    rounded = target.round().clamp(min=0).long()
    if rounded.numel() == 0:
        return torch.zeros_like(target)
    majority = int(torch.bincount(rounded).argmax().item())
    return torch.full_like(target, float(majority))


def _mean_baseline(target: torch.Tensor) -> torch.Tensor:
    return torch.full_like(target, float(target.float().mean()) if target.numel() else 0.0)


def _target_tensor(samples, field: str) -> torch.Tensor:
    return torch.tensor(
        [float(getattr(sample, field, 0)) for sample in samples],
        dtype=torch.float32,
    )


def _train_probe(
    features: torch.Tensor,
    target: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict[str, str | float | int]:
    rounded_target = target.round().clamp(min=0).long()
    num_classes = int(rounded_target.max().item()) + 1 if rounded_target.numel() else 1
    if num_classes <= 1:
        return {
            "name": "probe_reflection_path_count",
            "count": int(target.numel()),
            "MAE": 0.0,
            "RMSE": 0.0,
            "signed_mean": 0.0,
            "exact_accuracy": 1.0,
            "within_1_accuracy": 1.0,
            "pearson": 0.0,
            "target_histogram": _histogram_text(rounded_target),
            "prediction_histogram": _histogram_text(rounded_target),
        }

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    train_count = max(1, int(0.8 * features.shape[0]))
    train_idx = permutation[:train_count]
    eval_idx = permutation[train_count:]
    if eval_idx.numel() == 0:
        eval_idx = train_idx
    train_ds = TensorDataset(features[train_idx], rounded_target[train_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    probe = nn.Linear(features.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)
    for _ in range(epochs):
        probe.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(x), y)
            loss.backward()
            optimizer.step()
    probe.eval()
    with torch.no_grad():
        logits = probe(features[eval_idx].to(device)).cpu()
    pred = logits.argmax(dim=1).float()
    tgt = rounded_target[eval_idx].float()
    return _count_metrics("probe_reflection_path_count", pred, tgt)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    samples = _preprocess_samples(args)
    if not samples:
        raise ValueError("No samples available for diagnostic.")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    samples, checkpoint_prototype_keys, attribute_fields, attribute_remap = _prepare_samples(
        samples,
        checkpoint,
    )
    model, tokenizer = _build_model(
        samples,
        checkpoint,
        checkpoint_prototype_keys,
        attribute_fields,
        attribute_remap,
        device,
    )
    features, final, base, reflection_head, reflection_path_head = _collect_predictions(
        model,
        samples,
        tokenizer,
        args.batch_size,
        device,
    )
    final_raw = _raw_physics(final)
    base_raw = _raw_physics(base)
    reflection_head_raw = _raw_reflection_head(reflection_head)
    reflection_idx = PHYSICS_TARGET_NAMES.index("reflection_count")

    reflection_path_target = _target_tensor(samples, "reflection_path_count")
    diffraction_path_target = _target_tensor(samples, "diffraction_path_count")
    direct_path_target = _target_tensor(samples, "direct_path_count")

    print(f"diagnostic_device={device}")
    print(f"diagnostic_samples={len(samples)}")
    print(f"reflection_path_count_target_histogram={_histogram_text(reflection_path_target)}")
    print(f"diffraction_path_count_target_histogram={_histogram_text(diffraction_path_target)}")
    print(f"direct_path_count_target_histogram={_histogram_text(direct_path_target)}")

    rows = [
        _count_metrics(
            "final_physics_reflection_slot_vs_reflection_path_count",
            final_raw[:, reflection_idx],
            reflection_path_target,
        ),
        _count_metrics(
            "base_physics_reflection_slot_vs_reflection_path_count",
            base_raw[:, reflection_idx],
            reflection_path_target,
        ),
        _count_metrics(
            "specialized_reflection_head_vs_reflection_path_count",
            reflection_head_raw,
            reflection_path_target,
        ),
        _count_metrics(
            "trained_reflection_path_count_head",
            reflection_path_head * 10.0,
            reflection_path_target,
        ),
        _count_metrics(
            "majority_baseline_vs_reflection_path_count",
            _majority_baseline(reflection_path_target),
            reflection_path_target,
        ),
        _count_metrics(
            "mean_baseline_vs_reflection_path_count",
            _mean_baseline(reflection_path_target),
            reflection_path_target,
        ),
        _count_metrics(
            "final_physics_diffraction_slot_vs_diffraction_path_count",
            final_raw[:, PHYSICS_TARGET_NAMES.index("diffraction_count")],
            diffraction_path_target,
        ),
        _count_metrics(
            "final_physics_n_paths_slot_vs_direct_path_count",
            final_raw[:, PHYSICS_TARGET_NAMES.index("n_paths")],
            direct_path_target,
        ),
    ]
    if args.probe_epochs > 0:
        rows.append(
            _train_probe(
                features,
                reflection_path_target,
                seed=args.seed,
                epochs=args.probe_epochs,
                batch_size=args.probe_batch_size,
                lr=args.probe_lr,
                device=device,
            )
        )

    for row in rows:
        _print_metric_row(row)

    if args.output_csv is not None:
        _write_results(Path(args.output_csv), rows)
        print(f"saved_reflection_path_diagnostic_csv={args.output_csv}")
    if args.output_json is not None:
        _write_results(Path(args.output_json), rows)
        print(f"saved_reflection_path_diagnostic_json={args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", help="Optional preprocessed .pt that already has path-count fields.")
    parser.add_argument("--d2los-root", help="Path to RayVerse/D2Los_Data raw root.")
    parser.add_argument("--scenario", help="DeepMIMO scenario or D2Los root name under --scenario-root.")
    parser.add_argument("--scenario-root", default=str(ROOT / "Raytracing_scenarios"))
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-json")
    parser.add_argument("--save-samples")
    parser.add_argument("--freq-bin", type=int, default=0)
    parser.add_argument("--rx-index", type=int, default=0)
    parser.add_argument("--env-type", choices=["indoor", "outdoor", "O2I"])
    parser.add_argument("--max-maps", type=int)
    parser.add_argument("--max-sources-per-map", type=int)
    parser.add_argument("--max-rx-per-source", type=int)
    parser.add_argument(
        "--d2los-sampling",
        choices=["sequential", "uniform", "map_uniform"],
        default="map_uniform",
    )
    parser.add_argument("--d2los-sample-seed", type=int, default=0)
    parser.add_argument("--tx-shape", type=int, nargs=2, default=[8, 8])
    parser.add_argument("--bandwidth-hz", type=float, default=100e6)
    parser.add_argument("--total-subcarriers", type=int, default=128)
    parser.add_argument("--tx-power-dbm", type=float, default=23.0)
    parser.add_argument("--target-nf", type=int, default=128)
    parser.add_argument("--patch-1d", type=int, default=4)
    parser.add_argument("--patch-2d", type=int, nargs=2, default=[2, 2])
    parser.add_argument("--include-empty-samples", action="store_true")
    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=0,
        help="Optional frozen-encoder linear probe epochs. 0 disables probe training.",
    )
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
