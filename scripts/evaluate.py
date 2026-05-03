from __future__ import annotations

import argparse
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.caption import CaptionGenerator
from data.dataset import (
    PHYSICS_TARGET_NAMES,
    PHYSICS_TARGET_OFFSETS,
    PHYSICS_TARGET_SCALES,
    PreprocessedCSIDataset,
    apply_semantic_key_mode,
    collate_fn,
    semantic_key_mode_choices,
)
from data.semantic_key import SemanticKey, default_attribute_fields, semantic_key_field_choices
from data.tokenizer import CaptionTokenizer
from models.encoder import CSIEncoder
from models.model import CSIClip
from models.text_encoder import PhysicsTextEncoder
from training.losses import cosine_alignment_loss, paired_contrastive_loss


def build_tokenizer(samples, checkpoint: dict | None) -> CaptionTokenizer:
    tokenizer = CaptionTokenizer()
    if checkpoint is not None and "tokenizer_word2id" in checkpoint:
        tokenizer.word2id = dict(checkpoint["tokenizer_word2id"])
        tokenizer.id2word = {idx: word for word, idx in tokenizer.word2id.items()}
        tokenizer.next_id = max(tokenizer.id2word) + 1
    else:
        tokenizer.build_vocab(_build_prototype_captions(samples))
        tokenizer.build_vocab(sample.prop_caption for sample in samples)
        tokenizer.build_vocab(sample.instance_caption for sample in samples)
    return tokenizer


def semantic_key_sort_key(key: SemanticKey) -> tuple[str, ...]:
    return (
        key.env_type,
        key.los_status,
        key.path_richness,
        key.ds_bin,
        key.as_az_bin,
        key.k_factor_bin,
        key.first_delay_bin,
        key.first_power_bin,
        key.first_angle_bin,
        key.reflection_bin,
        key.diffraction_bin,
    )


def _build_prototype_captions(samples) -> list[str]:
    generator = CaptionGenerator()
    unique_keys = sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
    return [generator.generate_canonical(key) for key in unique_keys]


def build_prototype_bank(
    samples,
    tokenizer: CaptionTokenizer,
    max_caption_len: int = 48,
) -> tuple[list[SemanticKey], torch.Tensor, torch.Tensor, dict[SemanticKey, int]]:
    generator = CaptionGenerator()
    unique_keys = sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
    captions = [generator.generate_canonical(key) for key in unique_keys]
    tokenized = [tokenizer.encode(caption, max_len=max_caption_len) for caption in captions]
    token_ids = torch.stack([item.ids for item in tokenized], dim=0)
    token_mask = torch.stack([item.mask for item in tokenized], dim=0)
    label_map = {key: idx for idx, key in enumerate(unique_keys)}
    return unique_keys, token_ids, token_mask, label_map


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
        else:
            moved[key] = value
    return moved


def _infer_text_mode(checkpoint: dict | None, override: str | None) -> str:
    if override is not None:
        return override
    if checkpoint is not None:
        return str(checkpoint.get("args", {}).get("text_mode", "prototype"))
    return "prototype"


def _infer_min_class_size(checkpoint: dict | None, override: int | None) -> int:
    if override is not None:
        return override
    if checkpoint is not None:
        return int(checkpoint.get("args", {}).get("min_class_size", 1))
    return 1


def _infer_semantic_key_mode(checkpoint: dict | None, override: str | None) -> str:
    if override is not None:
        return override
    if checkpoint is not None:
        return str(checkpoint.get("args", {}).get("semantic_key_mode", "full"))
    return "full"


def _infer_attribute_fields(checkpoint: dict | None, override: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if override is not None:
        return override
    if checkpoint is not None:
        fields = checkpoint.get("args", {}).get("attribute_classifier_fields")
        if fields:
            return tuple(str(field) for field in fields)
    return default_attribute_fields()


def build_attribute_label_maps(samples, fields: tuple[str, ...]) -> dict[str, dict[str, int]]:
    label_maps = {}
    for field in fields:
        values = sorted({str(getattr(sample.semantic_key, field)) for sample in samples})
        label_maps[field] = {value: idx for idx, value in enumerate(values)}
    return label_maps


def _load_model_state_compatible(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    compatible_state = {
        name: value
        for name, value in state_dict.items()
        if name in model_state and model_state[name].shape == value.shape
    }
    skipped = sorted(set(state_dict) - set(compatible_state))
    model.load_state_dict(compatible_state, strict=False)
    if skipped:
        print(f"skipped_incompatible_checkpoint_keys={','.join(skipped)}")


def filter_samples_by_min_class_size(samples, min_class_size: int):
    if min_class_size <= 1:
        return samples
    key_counts = Counter(sample.semantic_key for sample in samples)
    filtered = [sample for sample in samples if key_counts[sample.semantic_key] >= min_class_size]
    if not filtered:
        raise ValueError(
            f"No samples remain after filtering semantic classes with min_class_size={min_class_size}."
        )
    return filtered


@torch.no_grad()
def evaluate(
    data_path: str,
    checkpoint_path: str | None,
    batch_size: int,
    device: torch.device,
    text_mode_override: str | None = None,
    min_class_size_override: int | None = None,
    semantic_key_mode_override: str | None = None,
    attribute_fields_override: tuple[str, ...] | None = None,
) -> None:
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) if checkpoint_path else None
    text_mode = _infer_text_mode(checkpoint, text_mode_override)
    min_class_size = _infer_min_class_size(checkpoint, min_class_size_override)
    semantic_key_mode = _infer_semantic_key_mode(checkpoint, semantic_key_mode_override)
    attribute_fields = _infer_attribute_fields(checkpoint, attribute_fields_override)
    mode_samples = apply_semantic_key_mode(dataset.samples, semantic_key_mode)
    if semantic_key_mode != "full":
        before_counts = Counter(sample.semantic_key for sample in dataset.samples)
        after_counts = Counter(sample.semantic_key for sample in mode_samples)
        print(
            f"semantic_key_mode={semantic_key_mode}: "
            f"semantic_prototypes {len(before_counts)} -> {len(after_counts)}"
        )
    samples = filter_samples_by_min_class_size(mode_samples, min_class_size=min_class_size)
    if len(samples) != len(mode_samples):
        before_counts = Counter(sample.semantic_key for sample in mode_samples)
        after_counts = Counter(sample.semantic_key for sample in samples)
        print(
            f"filtered classes with min_class_size={min_class_size}: "
            f"samples {len(mode_samples)} -> {len(samples)}, "
            f"semantic_prototypes {len(before_counts)} -> {len(after_counts)}"
        )
    tokenizer = build_tokenizer(samples, checkpoint)
    prototype_keys, prototype_token_ids, prototype_token_mask, prototype_label_map = build_prototype_bank(
        samples,
        tokenizer,
    )
    attribute_label_maps = build_attribute_label_maps(samples, attribute_fields)
    loader = DataLoader(
        PreprocessedCSIDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
    model = CSIClip(
        CSIEncoder(d_token=8, d_model=384, d_clip=256),
        PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300)),
        num_prototypes=len(prototype_keys),
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        attribute_num_classes={
            field: len(label_map)
            for field, label_map in attribute_label_maps.items()
        },
    ).to(device)
    if checkpoint is not None:
        _load_model_state_compatible(model, checkpoint["model_state"])
    model.eval()

    all_csi_features = []
    all_attribute_logits = {field: [] for field in attribute_label_maps}
    all_attribute_labels = {field: [] for field in attribute_label_maps}
    all_instance_text_features = []
    all_physics_predictions = []
    all_physics_targets = []
    all_physics_raw_targets = []
    all_physics_masks = []
    all_labels = []
    all_text_labels = []

    for batch in loader:
        batch = move_batch(batch, device)
        csi_features = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=True,
        )
        all_csi_features.append(csi_features.cpu())
        if attribute_label_maps:
            attribute_logits = model.predict_attributes(csi_features)
            for field, label_map in attribute_label_maps.items():
                all_attribute_logits[field].append(attribute_logits[field].cpu())
                all_attribute_labels[field].extend(
                    label_map[str(getattr(key, field))]
                    for key in batch["semantic_keys"]
                )
        physics_predictions = model.predict_physics(csi_features)
        all_physics_predictions.append(physics_predictions.cpu())
        all_physics_targets.append(batch["physics_targets"].cpu())
        all_physics_raw_targets.append(batch["physics_raw_targets"].cpu())
        all_physics_masks.append(batch["physics_target_mask"].cpu())
        if text_mode in ("instance", "multipositive"):
            instance_text_features = model.encode_text(
                batch["t_instance_ids"],
                batch["t_instance_mask"],
                normalize=True,
            )
            all_instance_text_features.append(instance_text_features.cpu())
            all_text_labels.extend(prototype_label_map[key] for key in batch["semantic_keys"])
        all_labels.extend(prototype_label_map[key] for key in batch["semantic_keys"])

    prototype_text_features = model.encode_text(
        prototype_token_ids.to(device),
        prototype_token_mask.to(device),
        normalize=True,
    ).cpu()
    prototype_features = model.encode_prototypes(normalize=True).cpu()
    csi_features = torch.cat(all_csi_features, dim=0)
    attribute_logits = {
        field: torch.cat(chunks, dim=0)
        for field, chunks in all_attribute_logits.items()
        if chunks
    }
    attribute_labels = {
        field: torch.tensor(values, dtype=torch.long)
        for field, values in all_attribute_labels.items()
        if values
    }
    physics_predictions = torch.cat(all_physics_predictions, dim=0)
    physics_targets = torch.cat(all_physics_targets, dim=0)
    physics_raw_targets = torch.cat(all_physics_raw_targets, dim=0)
    physics_masks = torch.cat(all_physics_masks, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    logit_scale = float(model.logit_scale.exp().detach().cpu().item())
    prototype_logits = logit_scale * csi_features @ prototype_features.T

    if text_mode == "prototype":
        text_features = prototype_text_features
        text_labels = labels
        logits = logit_scale * csi_features @ text_features.T
        text_metric_prefix = "csi_to_text_proto"
        text_prototype_targets = prototype_features
    elif text_mode in ("instance", "multipositive"):
        text_features = torch.cat(all_instance_text_features, dim=0)
        text_labels = torch.arange(text_features.shape[0], dtype=torch.long)
        instance_semantic_labels = torch.tensor(all_text_labels, dtype=torch.long)
        logits = logit_scale * csi_features @ text_features.T
        text_metric_prefix = "csi_to_instance_text"
        text_prototype_targets = prototype_features[labels]
    else:
        raise ValueError(f"Unsupported text_mode={text_mode!r}")

    instance_loss = F.cross_entropy(logits, text_labels)
    csi_prototype_loss = F.cross_entropy(prototype_logits, labels)
    if text_mode == "prototype":
        text_prototype_loss = paired_contrastive_loss(
            text_features,
            text_prototype_targets,
            torch.tensor(logit_scale, dtype=text_features.dtype),
        )
    else:
        text_prototype_loss = cosine_alignment_loss(
            text_features,
            text_prototype_targets,
        )
    eval_loss = instance_loss + csi_prototype_loss + text_prototype_loss
    print(f"eval_learnable_prototype_loss={float(eval_loss):.4f}")
    print(f"eval_csi_to_text_loss={float(instance_loss):.4f}")
    print(f"eval_csi_to_prototype_loss={float(csi_prototype_loss):.4f}")
    print(f"eval_text_to_prototype_loss={float(text_prototype_loss):.4f}")
    print(f"logit_scale={logit_scale:.4f}")
    print(f"text_mode={text_mode}")
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"min_class_size={min_class_size}")
    print(f"semantic_prototypes={len(prototype_keys)}")
    if checkpoint is not None and float(checkpoint.get("args", {}).get("attribute_classifier_weight", 0.0)) > 0:
        print(f"attribute_classifier_fields={','.join(attribute_fields)}")
    _print_retrieval_metrics(text_metric_prefix, logits, text_labels)
    if checkpoint is not None and float(checkpoint.get("args", {}).get("attribute_classifier_weight", 0.0)) > 0:
        _print_attribute_classifier_metrics(
            attribute_logits,
            attribute_labels,
            attribute_label_maps,
        )
    _print_physics_regression_metrics(
        physics_predictions=physics_predictions,
        physics_raw_targets=physics_raw_targets,
        physics_masks=physics_masks,
    )
    if text_mode in ("instance", "multipositive"):
        _print_semantic_retrieval_metrics(
            "csi_to_instance_text_semantic",
            logits,
            query_labels=labels,
            item_labels=instance_semantic_labels,
        )
        _print_gated_exact_retrieval_metrics(
            "csi_to_instance_text_oracle_semantic_exact",
            logits,
            target_indices=text_labels,
            item_labels=instance_semantic_labels,
            gate_labels=labels,
        )
        _print_gated_exact_retrieval_metrics(
            "csi_to_instance_text_predicted_semantic_exact",
            logits,
            target_indices=text_labels,
            item_labels=instance_semantic_labels,
            gate_labels=prototype_logits.argmax(dim=1),
        )
        _print_physics_neighbor_retrieval_metrics(
            "csi_to_instance_text_physics_neighbor",
            logits,
            query_targets=physics_targets,
            query_masks=physics_masks,
            item_targets=physics_targets,
            item_masks=physics_masks,
            distance_threshold=float(
                checkpoint.get("args", {}).get("multipositive_distance_threshold", 0.25)
                if checkpoint is not None
                else 0.25
            ),
        )
    _print_retrieval_metrics("csi_to_learnable_prototype", prototype_logits, labels)

    _print_retrieval_metrics(
        "text_proto_to_learnable_prototype",
        logit_scale * prototype_text_features @ prototype_features.T,
        torch.arange(prototype_features.shape[0], dtype=torch.long),
    )


def _print_retrieval_metrics(prefix: str, logits: torch.Tensor, labels: torch.Tensor) -> None:
    ranking = logits.argsort(dim=1, descending=True)
    target_ranks = (ranking == labels.unsqueeze(1)).float().argmax(dim=1) + 1
    for k in (1, 5, 10):
        hits = (target_ranks <= min(k, logits.shape[1])).float().mean().item()
        print(f"{prefix}_R@{k}={hits:.4f}")
    print(f"{prefix}_MRR={float((1.0 / target_ranks.float()).mean()):.4f}")
    print(f"{prefix}_mean_rank={float(target_ranks.float().mean()):.2f}")


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return 0.0
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denom) == 0.0:
        return 0.0
    return float((x * y).sum() / denom)


def _print_attribute_classifier_metrics(
    attribute_logits: dict[str, torch.Tensor],
    attribute_labels: dict[str, torch.Tensor],
    attribute_label_maps: dict[str, dict[str, int]],
) -> None:
    for field, logits in attribute_logits.items():
        labels = attribute_labels[field]
        predictions = logits.argmax(dim=1)
        label_map = attribute_label_maps[field]
        id_to_value = {idx: value for value, idx in label_map.items()}
        num_classes = len(label_map)
        confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for true_label, pred_label in zip(labels.tolist(), predictions.tolist()):
            confusion[int(true_label), int(pred_label)] += 1

        class_sizes = confusion.sum(dim=1)
        class_correct = confusion.diag()
        nonempty = class_sizes > 0
        class_accuracy = torch.zeros(num_classes, dtype=torch.float32)
        class_accuracy[nonempty] = class_correct[nonempty].float() / class_sizes[nonempty].float()
        top1 = (predictions == labels).float().mean()
        macro_acc = class_accuracy[nonempty].mean() if bool(nonempty.any()) else torch.zeros(())
        majority_label = int(class_sizes.argmax().item())
        majority_acc = float(class_sizes[majority_label]) / max(int(class_sizes.sum().item()), 1)

        print(f"attribute_classifier_{field}_loss={float(F.cross_entropy(logits, labels)):.4f}")
        print(f"attribute_classifier_{field}_top1={float(top1):.4f}")
        print(f"attribute_classifier_{field}_macro_top1={float(macro_acc):.4f}")
        print(f"attribute_classifier_{field}_majority_baseline_R@1={majority_acc:.4f}")
        print(f"attribute_classifier_{field}_majority_value={id_to_value[majority_label]}")
        print(
            f"attribute_classifier_{field}_class_size_accuracy_pearson="
            f"{_safe_pearson(class_sizes[nonempty].float(), class_accuracy[nonempty]):.4f}"
        )
        for class_idx in range(num_classes):
            print(
                f"attribute_classifier_{field}_value_{id_to_value[class_idx]}="
                f"size:{int(class_sizes[class_idx])} "
                f"acc:{float(class_accuracy[class_idx]):.4f}"
            )

        offdiag = confusion.clone()
        offdiag.fill_diagonal_(0)
        flat_counts = offdiag.flatten()
        top_confusions = torch.argsort(flat_counts, descending=True)
        printed = 0
        for flat_idx_tensor in top_confusions:
            count = int(flat_counts[int(flat_idx_tensor)].item())
            if count <= 0 or printed >= min(5, num_classes * num_classes):
                break
            true_label = int(flat_idx_tensor.item() // num_classes)
            pred_label = int(flat_idx_tensor.item() % num_classes)
            printed += 1
            print(
                f"attribute_classifier_{field}_confusion_pair_rank_{printed}="
                f"true:{id_to_value[true_label]} pred:{id_to_value[pred_label]} count:{count}"
            )


def _print_semantic_retrieval_metrics(
    prefix: str,
    logits: torch.Tensor,
    query_labels: torch.Tensor,
    item_labels: torch.Tensor,
) -> None:
    ranking = logits.argsort(dim=1, descending=True)
    ranked_labels = item_labels[ranking]
    matches = ranked_labels == query_labels.unsqueeze(1)
    target_ranks = matches.float().argmax(dim=1) + 1
    positive_counts = (item_labels.unsqueeze(0) == query_labels.unsqueeze(1)).sum(dim=1).float()
    for k in (1, 5, 10):
        hits = matches[:, : min(k, logits.shape[1])].any(dim=1).float().mean().item()
        print(f"{prefix}_R@{k}={hits:.4f}")
    print(f"{prefix}_MRR={float((1.0 / target_ranks.float()).mean()):.4f}")
    print(f"{prefix}_mean_rank={float(target_ranks.float().mean()):.2f}")
    print(f"{prefix}_positive_count_mean={float(positive_counts.mean()):.2f}")


def _print_gated_exact_retrieval_metrics(
    prefix: str,
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    item_labels: torch.Tensor,
    gate_labels: torch.Tensor,
) -> None:
    ranks = []
    fallback_rank = logits.shape[1] + 1
    for row_idx in range(logits.shape[0]):
        target_idx = int(target_indices[row_idx])
        candidate_mask = item_labels == gate_labels[row_idx]
        if not bool(candidate_mask[target_idx]):
            ranks.append(fallback_rank)
            continue
        target_score = logits[row_idx, target_idx]
        candidate_scores = logits[row_idx, candidate_mask]
        ranks.append(int((candidate_scores > target_score).sum().item()) + 1)

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    for k in (1, 5, 10):
        hits = (ranks_tensor <= k).float().mean().item()
        print(f"{prefix}_R@{k}={hits:.4f}")
    print(f"{prefix}_MRR={float((1.0 / ranks_tensor).mean()):.4f}")
    print(f"{prefix}_mean_rank={float(ranks_tensor.mean()):.2f}")


def _print_physics_neighbor_retrieval_metrics(
    prefix: str,
    logits: torch.Tensor,
    query_targets: torch.Tensor,
    query_masks: torch.Tensor,
    item_targets: torch.Tensor,
    item_masks: torch.Tensor,
    distance_threshold: float,
) -> None:
    ranking = logits.argsort(dim=1, descending=True)
    ranks = []
    positive_counts = []
    for row_idx in range(logits.shape[0]):
        common_mask = query_masks[row_idx].unsqueeze(0) & item_masks
        common_count = common_mask.sum(dim=1)
        diffs = (query_targets[row_idx].unsqueeze(0) - item_targets).abs()
        distances = (diffs * common_mask.to(dtype=diffs.dtype)).sum(dim=1)
        distances = distances / common_count.clamp(min=1).to(dtype=diffs.dtype)
        positives = (common_count >= 3) & (distances <= distance_threshold)
        positives[row_idx] = True
        positive_counts.append(int(positives.sum().item()))
        ranked_positive = positives[ranking[row_idx]]
        ranks.append(int(ranked_positive.float().argmax().item()) + 1)

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    positive_counts_tensor = torch.tensor(positive_counts, dtype=torch.float32)
    for k in (1, 5, 10):
        hits = (ranks_tensor <= k).float().mean().item()
        print(f"{prefix}_R@{k}={hits:.4f}")
    print(f"{prefix}_MRR={float((1.0 / ranks_tensor).mean()):.4f}")
    print(f"{prefix}_mean_rank={float(ranks_tensor.mean()):.2f}")
    print(f"{prefix}_positive_count_mean={float(positive_counts_tensor.mean()):.2f}")


def _print_physics_regression_metrics(
    physics_predictions: torch.Tensor,
    physics_raw_targets: torch.Tensor,
    physics_masks: torch.Tensor,
) -> None:
    raw_predictions = physics_predictions * PHYSICS_TARGET_SCALES + PHYSICS_TARGET_OFFSETS
    errors = (raw_predictions - physics_raw_targets).abs()
    mae_values = []
    angle_sin_idx = PHYSICS_TARGET_NAMES.index("first_path_aoa_az_sin")
    angle_cos_idx = PHYSICS_TARGET_NAMES.index("first_path_aoa_az_cos")
    for idx, name in enumerate(PHYSICS_TARGET_NAMES):
        if name in {"first_path_aoa_az_sin", "first_path_aoa_az_cos"}:
            continue
        mask = physics_masks[:, idx]
        if not bool(mask.any()):
            continue
        mae = errors[:, idx][mask].mean()
        mae_values.append(mae)
        print(f"physics_regression_{name}_MAE={float(mae):.4f}")
    angle_mask = physics_masks[:, angle_sin_idx] & physics_masks[:, angle_cos_idx]
    if bool(angle_mask.any()):
        pred_angle = torch.atan2(
            raw_predictions[:, angle_sin_idx],
            raw_predictions[:, angle_cos_idx],
        )
        target_angle = torch.atan2(
            physics_raw_targets[:, angle_sin_idx],
            physics_raw_targets[:, angle_cos_idx],
        )
        delta = pred_angle - target_angle
        circular_errors = torch.rad2deg(torch.atan2(torch.sin(delta), torch.cos(delta)).abs())
        circular_mae = circular_errors[angle_mask].mean()
        mae_values.append(circular_mae)
        print(f"physics_regression_first_path_aoa_az_deg_MAE={float(circular_mae):.4f}")
    if mae_values:
        total_mae = torch.stack(mae_values).mean()
    else:
        total_mae = torch.zeros(())
    print(f"physics_regression_MAE_mean={float(total_mae):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-mode", choices=["prototype", "instance", "multipositive"])
    parser.add_argument(
        "--semantic-key-mode",
        choices=semantic_key_mode_choices(),
        help="Semantic key granularity for evaluation. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        help="Drop semantic classes with fewer than this many samples before evaluation. Defaults to checkpoint args.",
    )
    parser.add_argument(
        "--attribute-classifier-fields",
        nargs="+",
        choices=semantic_key_field_choices(),
        help="SemanticKey fields for attribute classifier evaluation. Defaults to checkpoint args.",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(
        args.data_path,
        args.checkpoint,
        args.batch_size,
        device,
        text_mode_override=args.text_mode,
        min_class_size_override=args.min_class_size,
        semantic_key_mode_override=args.semantic_key_mode,
        attribute_fields_override=tuple(args.attribute_classifier_fields) if args.attribute_classifier_fields else None,
    )


if __name__ == "__main__":
    main()
