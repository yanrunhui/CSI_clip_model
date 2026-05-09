from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrain import (
    assert_checkpoint_prototype_compatibility,
    build_real_components,
    checkpoint_has_compatible_prototypes,
    cfg_get,
    format_attribute_remap,
    format_attribute_value_filters,
    initialize_prototypes_from_canonical_text,
    load_model_state_compatible,
    load_train_config,
    load_transfer_checkpoint,
    parse_attribute_fields,
    parse_attribute_remap,
    parse_attribute_value_filters,
    trainable_parameters,
)
from training.trainer import TrainConfig, Trainer


def infer_arg(checkpoint: dict | None, name: str, override, default):
    if override is not None:
        return override
    if checkpoint is not None:
        value = checkpoint.get("args", {}).get(name)
        if value is not None:
            return value
    return default


@torch.no_grad()
def prototype_alignment_artifacts(
    model,
    prototype_token_ids: torch.Tensor,
    prototype_token_mask: torch.Tensor,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    was_training = model.training
    model.eval()
    token_ids = prototype_token_ids.to(next(model.parameters()).device)
    token_mask = prototype_token_mask.to(next(model.parameters()).device)
    text_features = model.encode_text(token_ids, token_mask, normalize=True)
    prototype_features = model.encode_prototypes(normalize=True)
    cosine = text_features @ prototype_features.T
    logits = model.logit_scale.exp() * cosine
    labels = torch.arange(cosine.shape[0], device=cosine.device)
    top1 = (cosine.argmax(dim=1) == labels).float().mean()
    top5 = (cosine.topk(k=min(5, cosine.shape[1]), dim=1).indices == labels[:, None]).any(dim=1).float().mean()
    loss = F.cross_entropy(logits, labels)
    diag = cosine.diag()
    offdiag = cosine[~torch.eye(cosine.shape[0], device=cosine.device, dtype=torch.bool)]
    metrics = {
        "top1": float(top1),
        "top5": float(top5),
        "loss": float(loss),
        "diag_cosine_mean": float(diag.mean()),
        "diag_cosine_min": float(diag.min()),
        "offdiag_cosine_mean": float(offdiag.mean()) if offdiag.numel() else 0.0,
        "offdiag_cosine_max": float(offdiag.max()) if offdiag.numel() else 0.0,
        "max_abs_feature_diff": float((text_features - prototype_features).abs().max()),
        "logit_scale": float(model.logit_scale.exp().detach()),
    }
    artifacts = {
        "cosine": cosine.detach().cpu(),
        "text_features": text_features.detach().cpu(),
        "prototype_features": prototype_features.detach().cpu(),
        "predicted_indices": cosine.argmax(dim=1).detach().cpu(),
    }
    if was_training:
        model.train()
    return metrics, artifacts


@torch.no_grad()
def prototype_alignment_metrics(
    model,
    prototype_token_ids: torch.Tensor,
    prototype_token_mask: torch.Tensor,
) -> dict[str, float]:
    metrics, _ = prototype_alignment_artifacts(
        model,
        prototype_token_ids,
        prototype_token_mask,
    )
    return metrics


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        print(f"{prefix}_{key}={value:.6f}")


def print_index_alignment_report(
    prefix: str,
    artifacts: dict[str, torch.Tensor],
    prototype_bank,
    max_items: int,
) -> None:
    cosine = artifacts["cosine"]
    predicted_indices = artifacts["predicted_indices"]
    labels = torch.arange(cosine.shape[0], dtype=torch.long)
    mismatches = torch.where(predicted_indices != labels)[0]
    print(f"{prefix}_index_match_count={int((predicted_indices == labels).sum().item())}")
    print(f"{prefix}_index_mismatch_count={int(mismatches.numel())}")
    if mismatches.numel() == 0:
        return
    for rank, idx in enumerate(mismatches[:max_items].tolist(), start=1):
        pred = int(predicted_indices[idx].item())
        expected_key = prototype_bank["keys"][idx]
        predicted_key = prototype_bank["keys"][pred]
        expected_caption = prototype_bank["captions"][idx]
        predicted_caption = prototype_bank["captions"][pred]
        print(
            f"{prefix}_mismatch_{rank}_expected_idx={idx} predicted_idx={pred} "
            f"expected_key={expected_key} predicted_key={predicted_key} "
            f"expected_caption={expected_caption!r} predicted_caption={predicted_caption!r} "
            f"expected_cosine={float(cosine[idx, idx]):.6f} predicted_cosine={float(cosine[idx, pred]):.6f}"
        )


@torch.no_grad()
def sample_prototype_mapping_report(
    loader,
    model,
    prototype_bank,
    device: torch.device,
    max_items: int,
) -> None:
    label_map = prototype_bank["label_map"]
    prototype_keys = prototype_bank["keys"]
    sample_labels: list[int] = []
    missing_keys: list[object] = []
    roundtrip_mismatches: list[tuple[int, object, object]] = []
    sample_index = 0
    for batch in loader:
        for key in batch["semantic_keys"]:
            label = label_map.get(key)
            if label is None:
                missing_keys.append(key)
                sample_index += 1
                continue
            sample_labels.append(label)
            roundtrip_key = prototype_keys[label]
            if roundtrip_key != key:
                roundtrip_mismatches.append((sample_index, key, roundtrip_key))
            sample_index += 1

    print(f"sample_semantic_key_total={sample_index}")
    print(f"sample_semantic_key_mapped={len(sample_labels)}")
    print(f"sample_semantic_key_missing_count={len(missing_keys)}")
    print(f"sample_semantic_key_roundtrip_mismatch_count={len(roundtrip_mismatches)}")
    if sample_labels:
        labels = torch.tensor(sample_labels, dtype=torch.long)
        counts = torch.bincount(labels, minlength=len(prototype_keys))
        print(f"sample_semantic_key_unique_label_count={int((counts > 0).sum().item())}")
        print(f"sample_semantic_key_majority_count={int(counts.max().item())}")
    else:
        return

    for rank, key in enumerate(missing_keys[:max_items], start=1):
        print(f"sample_semantic_key_missing_{rank}={key}")
    for rank, (idx, expected_key, roundtrip_key) in enumerate(roundtrip_mismatches[:max_items], start=1):
        print(
            f"sample_semantic_key_roundtrip_mismatch_{rank}="
            f"sample_idx:{idx} expected_key:{expected_key} roundtrip_key:{roundtrip_key}"
        )

    was_training = model.training
    model.eval()
    token_ids = prototype_bank["token_ids"].to(device)
    token_mask = prototype_bank["token_mask"].to(device)
    prototype_text_features = model.encode_text(token_ids, token_mask, normalize=True)
    prototype_features = model.encode_prototypes(normalize=True)
    oracle_labels = labels.to(device)
    oracle_text_features = prototype_text_features.index_select(dim=0, index=oracle_labels)
    oracle_logits = model.logit_scale.exp() * oracle_text_features @ prototype_features.T
    oracle_top1 = (oracle_logits.argmax(dim=1) == oracle_labels).float().mean()
    oracle_top5 = (
        (oracle_logits.topk(k=min(5, oracle_logits.shape[1]), dim=1).indices == oracle_labels[:, None])
        .any(dim=1)
        .float()
        .mean()
    )
    oracle_loss = F.cross_entropy(oracle_logits, oracle_labels)
    print(f"oracle_sample_text_proto_top1={float(oracle_top1):.6f}")
    print(f"oracle_sample_text_proto_top5={float(oracle_top5):.6f}")
    print(f"oracle_sample_text_proto_loss={float(oracle_loss):.6f}")
    if was_training:
        model.train()


def run_text_only_diagnostic(
    loader,
    model,
    prototype_bank,
    attribute_remap,
    device: torch.device,
    steps: int,
    lr: float,
    weight_decay: float,
    text_mode: str,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=lr, weight_decay=weight_decay)
    trainer = Trainer(
        model,
        optimizer,
        device,
        prototype_token_ids=prototype_bank["token_ids"],
        prototype_token_mask=prototype_bank["token_mask"],
        prototype_label_map=prototype_bank["label_map"],
        prototype_class_counts=prototype_bank["class_counts"],
        attribute_label_maps=prototype_bank["attribute_label_maps"],
        attribute_class_counts=prototype_bank["attribute_class_counts"],
        attribute_remap=attribute_remap,
    )
    cfg = TrainConfig(
        csi_to_text_weight=0.0,
        prototype_weight=0.0,
        text_prototype_weight=1.0,
        text_mode=text_mode,
        attribute_classifier_weight=0.0,
        aux_regression_weight=0.0,
        freeze_csi=True,
        prototype_warmup_epochs=0,
    )
    model.csi.eval()
    for parameter in model.csi.parameters():
        parameter.requires_grad = False

    step = 0
    epoch = 1
    while step < steps:
        for batch in loader:
            trainer.train_step(batch, epoch=epoch, cfg=cfg)
            step += 1
            if step >= steps:
                break
        epoch += 1
    return prototype_alignment_artifacts(
        model,
        prototype_bank["token_ids"],
        prototype_bank["token_mask"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--semantic-key-mode", choices=("full", "coarse", "coarse_delay", "coarse_angle", "coarse_k", "coarse_k_angle", "coarse_interaction"))
    parser.add_argument("--min-class-size", type=int)
    parser.add_argument("--filter-attribute-values", action="append")
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-samples-by-attribute")
    parser.add_argument("--limit-samples-per-attribute-value", type=int)
    parser.add_argument("--attribute-classifier-fields", nargs="+")
    parser.add_argument(
        "--report-mismatches",
        type=int,
        default=10,
        help="Print up to this many prototype index mismatches for init-only and text-only diagnostics.",
    )
    parser.add_argument("--text-only-steps", type=int, default=100)
    parser.add_argument("--text-only-lr", type=float, default=3e-4)
    parser.add_argument("--text-only-weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--text-only-mode",
        choices=["prototype", "instance", "multipositive"],
        default="prototype",
        help="Loss form used during the short text-only diagnostic.",
    )
    args = parser.parse_args()

    train_cfg = load_train_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_transfer_checkpoint(args.checkpoint, device) if args.checkpoint else None

    batch_size = int(infer_arg(checkpoint, "batch_size", args.batch_size, cfg_get(train_cfg, "batch_size", 128)))
    semantic_key_mode = str(
        infer_arg(checkpoint, "semantic_key_mode", args.semantic_key_mode, cfg_get(train_cfg, "semantic_key_mode", "full"))
    )
    min_class_size = int(infer_arg(checkpoint, "min_class_size", args.min_class_size, cfg_get(train_cfg, "min_class_size", 1)))
    attribute_remap = parse_attribute_remap(
        infer_arg(checkpoint, "attribute_remap", None, cfg_get(train_cfg, "attribute_remap", None))
    )
    attribute_fields = parse_attribute_fields(
        infer_arg(
            checkpoint,
            "attribute_classifier_fields",
            args.attribute_classifier_fields,
            cfg_get(train_cfg, "attribute_classifier_fields", ()),
        )
    )
    filter_attribute_values = parse_attribute_value_filters(
        infer_arg(
            checkpoint,
            "filter_attribute_values",
            args.filter_attribute_values,
            cfg_get(train_cfg, "filter_attribute_values", None),
        )
    )
    limit_samples = infer_arg(checkpoint, "limit_samples", args.limit_samples, cfg_get(train_cfg, "limit_samples", None))
    limit_samples = int(limit_samples) if limit_samples is not None else None
    limit_samples_by_attribute = infer_arg(
        checkpoint,
        "limit_samples_by_attribute",
        args.limit_samples_by_attribute,
        cfg_get(train_cfg, "limit_samples_by_attribute", None),
    )
    limit_samples_per_attribute_value = infer_arg(
        checkpoint,
        "limit_samples_per_attribute_value",
        args.limit_samples_per_attribute_value,
        cfg_get(train_cfg, "limit_samples_per_attribute_value", None),
    )
    if limit_samples_per_attribute_value is not None:
        limit_samples_per_attribute_value = int(limit_samples_per_attribute_value)

    loader, model, _, prototype_bank = build_real_components(
        data_path=args.data_path,
        device=device,
        batch_size=batch_size,
        temperature=float(infer_arg(checkpoint, "temperature", None, cfg_get(train_cfg, "temperature", 0.07))),
        min_class_size=min_class_size,
        semantic_key_mode=semantic_key_mode,
        attribute_fields=attribute_fields,
        attribute_remap=attribute_remap,
        filter_attribute_values=filter_attribute_values,
        limit_samples=limit_samples,
        limit_samples_by_attribute=limit_samples_by_attribute,
        limit_samples_per_attribute_value=limit_samples_per_attribute_value,
        tokenizer_word2id=(checkpoint.get("tokenizer_word2id") if checkpoint is not None else None),
    )
    if checkpoint is not None:
        assert_checkpoint_prototype_compatibility(
            checkpoint,
            prototype_bank["keys"],
            expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
            context="diagnose checkpoint",
        )
        load_model_state_compatible(model, checkpoint)

    print(f"checkpoint={args.checkpoint}")
    print(f"data_path={args.data_path}")
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"min_class_size={min_class_size}")
    print(f"attribute_fields={','.join(attribute_fields)}")
    print(f"attribute_remap={format_attribute_remap(attribute_remap)}")
    print(f"filter_attribute_values={format_attribute_value_filters(filter_attribute_values)}")
    print(f"limit_samples={limit_samples}")
    print(f"limit_samples_by_attribute={limit_samples_by_attribute}")
    print(f"limit_samples_per_attribute_value={limit_samples_per_attribute_value}")
    print(
        f"checkpoint_has_compatible_prototypes="
        f"{int(checkpoint_has_compatible_prototypes(model, checkpoint, prototype_bank['keys']))}"
    )
    print(f"semantic_prototypes={len(prototype_bank['keys'])}")
    sample_prototype_mapping_report(
        loader,
        model,
        prototype_bank,
        device,
        args.report_mismatches,
    )

    before_init, before_artifacts = prototype_alignment_artifacts(
        model,
        prototype_bank["token_ids"],
        prototype_bank["token_mask"],
    )
    print_metrics("before_init", before_init)
    print_index_alignment_report("before_init", before_artifacts, prototype_bank, args.report_mismatches)

    initialize_prototypes_from_canonical_text(
        model,
        prototype_bank["token_ids"],
        prototype_bank["token_mask"],
    )
    after_init, after_init_artifacts = prototype_alignment_artifacts(
        model,
        prototype_bank["token_ids"],
        prototype_bank["token_mask"],
    )
    print_metrics("after_init", after_init)
    print_index_alignment_report("after_init", after_init_artifacts, prototype_bank, args.report_mismatches)

    after_text_only, after_text_only_artifacts = run_text_only_diagnostic(
        loader=loader,
        model=model,
        prototype_bank=prototype_bank,
        attribute_remap=attribute_remap,
        device=device,
        steps=args.text_only_steps,
        lr=args.text_only_lr,
        weight_decay=args.text_only_weight_decay,
        text_mode=args.text_only_mode,
    )
    print(f"text_only_steps={args.text_only_steps}")
    print(f"text_only_mode={args.text_only_mode}")
    print_metrics("after_text_only", after_text_only)
    print_index_alignment_report(
        "after_text_only",
        after_text_only_artifacts,
        prototype_bank,
        args.report_mismatches,
    )


if __name__ == "__main__":
    main()
