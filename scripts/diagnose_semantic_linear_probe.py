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
    PreprocessedCSIDataset,
    apply_semantic_key_mode,
    collate_fn,
    semantic_key_mode_choices,
)
from data.semantic_key import SemanticKey, semantic_key_attribute_raw_value, semantic_key_attribute_value
from data.tokenizer import CaptionTokenizer
from models.encoder import CSIEncoder
from models.model import CSIClip
from models.text_encoder import PhysicsTextEncoder
from scripts.pretrain import (
    assert_checkpoint_prototype_compatibility,
    cfg_get,
    format_attribute_remap,
    format_attribute_value_filters,
    load_train_config,
    parse_attribute_remap,
    parse_attribute_value_filters,
)


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


def infer_arg(checkpoint: dict | None, name: str, override, default):
    if override is not None:
        return override
    if checkpoint is not None:
        value = checkpoint.get("args", {}).get(name)
        if value is not None:
            return value
    return default


def build_tokenizer(samples, checkpoint: dict | None) -> CaptionTokenizer:
    tokenizer = CaptionTokenizer()
    if checkpoint is not None and "tokenizer_word2id" in checkpoint:
        tokenizer.word2id = dict(checkpoint["tokenizer_word2id"])
        tokenizer.id2word = {idx: word for word, idx in tokenizer.word2id.items()}
        tokenizer.next_id = max(tokenizer.id2word) + 1
        return tokenizer
    generator = CaptionGenerator()
    unique_keys = sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
    tokenizer.build_vocab(generator.generate_canonical(key) for key in unique_keys)
    tokenizer.build_vocab(sample.prop_caption for sample in samples)
    tokenizer.build_vocab(sample.instance_caption for sample in samples)
    return tokenizer


def filter_samples_by_min_class_size(samples, min_class_size: int):
    if min_class_size <= 1:
        return samples
    key_counts = Counter(sample.semantic_key for sample in samples)
    filtered = [sample for sample in samples if key_counts[sample.semantic_key] >= min_class_size]
    if not filtered:
        raise ValueError(f"No samples remain after min_class_size={min_class_size}.")
    return filtered


def filter_samples_by_attribute_values(samples, filters: dict[str, tuple[str, ...]]):
    filtered = samples
    for field, values in filters.items():
        allowed_values = set(values)
        filtered = [
            sample
            for sample in filtered
            if semantic_key_attribute_raw_value(sample.semantic_key, field) in allowed_values
        ]
        if not filtered:
            raise ValueError(
                f"No samples remain after filtering {field} to values {','.join(values)}."
            )
    return filtered


def limit_samples_by_attribute_value(
    samples,
    attribute_field: str | None,
    samples_per_value: int | None,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]] | None = None,
):
    if attribute_field is None and samples_per_value is None:
        return samples
    if attribute_field is None or samples_per_value is None:
        raise ValueError("--limit-samples-by-attribute and --limit-samples-per-attribute-value must be used together.")
    grouped = {}
    for sample in samples:
        grouped.setdefault(
            semantic_key_attribute_value(sample.semantic_key, attribute_field, attribute_remap),
            [],
        ).append(sample)
    limited = []
    for value in sorted(grouped):
        limited.extend(grouped[value][:samples_per_value])
    if not limited:
        raise ValueError(f"No samples remain after limit_samples_by_attribute={attribute_field!r}.")
    return limited


def limit_samples(samples, limit: int | None):
    if limit is None:
        return samples
    return samples[:limit]


def build_semantic_label_map(samples) -> dict[SemanticKey, int]:
    unique_keys = sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
    return {key: idx for idx, key in enumerate(unique_keys)}


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def mean_pairwise_l2(features: torch.Tensor) -> float:
    if features.shape[0] <= 1:
        return 0.0
    distances = torch.pdist(features.float(), p=2)
    return float(distances.mean()) if distances.numel() else 0.0


def load_model(samples, tokenizer: CaptionTokenizer, checkpoint: dict | None, device: torch.device) -> CSIClip:
    unique_keys = sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
    model = CSIClip(
        CSIEncoder(d_token=8, d_model=384, d_clip=256),
        PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300)),
        num_prototypes=len(unique_keys),
        semantic_num_classes=len(unique_keys),
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        attribute_num_classes={},
    ).to(device)
    if checkpoint is not None:
        assert_checkpoint_prototype_compatibility(
            checkpoint,
            unique_keys,
            expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
            context="semantic linear probe checkpoint",
        )
        model_state = model.state_dict()
        compatible_state = {
            name: value
            for name, value in checkpoint["model_state"].items()
            if name in model_state and model_state[name].shape == value.shape
        }
        model.load_state_dict(compatible_state, strict=False)
        skipped = sorted(set(checkpoint["model_state"]) - set(compatible_state))
        if skipped:
            print(f"skipped_incompatible_checkpoint_keys={','.join(skipped)}")
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, samples, tokenizer: CaptionTokenizer, batch_size: int, device: torch.device):
    label_map = build_semantic_label_map(samples)
    loader = DataLoader(
        PreprocessedCSIDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
    features = []
    labels = []
    for batch in loader:
        batch = move_batch(batch, device)
        batch_features = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
        )
        features.append(batch_features.cpu())
        labels.extend(label_map[key] for key in batch["semantic_keys"])
    return torch.cat(features, dim=0), torch.tensor(labels, dtype=torch.long), label_map


def stratified_train_test_split(labels: torch.Tensor, train_fraction: float, seed: int):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--probe-train-fraction must be between 0 and 1.")
    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    test_indices = []
    for class_idx in sorted(labels.unique().tolist()):
        class_indices = torch.where(labels == int(class_idx))[0]
        class_indices = class_indices[torch.randperm(class_indices.numel(), generator=generator)]
        train_count = int(round(class_indices.numel() * train_fraction))
        train_count = min(max(train_count, 1), class_indices.numel() - 1)
        train_indices.append(class_indices[:train_count])
        test_indices.append(class_indices[train_count:])
    train_indices = torch.cat(train_indices)
    test_indices = torch.cat(test_indices)
    train_indices = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
    test_indices = test_indices[torch.randperm(test_indices.numel(), generator=generator)]
    return train_indices, test_indices


def print_class_distribution(label_map: dict[SemanticKey, int], labels: torch.Tensor) -> None:
    id_to_key = {idx: key for key, idx in label_map.items()}
    counts = Counter(labels.tolist())
    print(f"semantic_prototypes={len(label_map)}")
    for class_idx, count in sorted(counts.items(), key=lambda item: (-item[1], semantic_key_sort_key(id_to_key[item[0]]))):
        print(f"class_{class_idx}_count={count} key={id_to_key[class_idx]}")


def run_train_test_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    label_map: dict[SemanticKey, int],
    steps: int,
    lr: float,
    weight_decay: float,
    train_fraction: float,
    seed: int,
    top_confusions: int,
) -> None:
    train_indices, test_indices = stratified_train_test_split(labels, train_fraction, seed)
    train_features = features[train_indices]
    test_features = features[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_x = (train_features - mean) / std
    test_x = (test_features - mean) / std

    print(f"raw_feature_global_std={float(features.float().std()):.8f}")
    print(f"raw_feature_train_std={float(train_features.float().std()):.8f}")
    print(f"raw_feature_test_std={float(test_features.float().std()):.8f}")
    print(f"raw_feature_train_pairwise_l2_mean={mean_pairwise_l2(train_features):.8f}")
    print(f"raw_feature_test_pairwise_l2_mean={mean_pairwise_l2(test_features):.8f}")
    print(f"standardized_feature_train_pairwise_l2_mean={mean_pairwise_l2(train_x):.8f}")
    print(f"standardized_feature_test_pairwise_l2_mean={mean_pairwise_l2(test_x):.8f}")

    torch.manual_seed(seed)
    num_classes = int(labels.max().item()) + 1
    probe = torch.nn.Linear(features.shape[1], num_classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(train_x)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = probe(train_x)
        test_logits = probe(test_x)
        train_loss = F.cross_entropy(train_logits, train_labels)
        test_loss = F.cross_entropy(test_logits, test_labels)
        train_predictions = train_logits.argmax(dim=1)
        test_predictions = test_logits.argmax(dim=1)
        train_acc = (train_predictions == train_labels).float().mean()
        test_acc = (test_predictions == test_labels).float().mean()
        majority_label = int(torch.bincount(train_labels, minlength=num_classes).argmax().item())
        majority_acc = (test_labels == majority_label).float().mean()

    id_to_key = {idx: key for key, idx in label_map.items()}
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true_label, pred_label in zip(test_labels.tolist(), test_predictions.tolist()):
        confusion[int(true_label), int(pred_label)] += 1

    prediction_counts = Counter(test_predictions.tolist())
    class_sizes = confusion.sum(dim=1)
    class_correct = confusion.diag()
    nonempty = class_sizes > 0
    class_accuracy = torch.zeros(num_classes, dtype=torch.float32)
    class_accuracy[nonempty] = class_correct[nonempty].float() / class_sizes[nonempty].float()
    macro_acc = class_accuracy[nonempty].mean() if bool(nonempty.any()) else torch.zeros(())

    print(f"linear_probe_split_train_count={train_indices.numel()}")
    print(f"linear_probe_split_test_count={test_indices.numel()}")
    print(f"linear_probe_split_train_loss={float(train_loss):.6f}")
    print(f"linear_probe_split_test_loss={float(test_loss):.6f}")
    print(f"linear_probe_split_train_acc={float(train_acc):.6f}")
    print(f"linear_probe_split_test_acc={float(test_acc):.6f}")
    print(f"linear_probe_split_test_macro_acc={float(macro_acc):.6f}")
    print(f"linear_probe_split_test_majority_baseline={float(majority_acc):.6f}")
    print(f"linear_probe_split_test_majority_label={majority_label}")
    print(f"linear_probe_split_test_majority_key={id_to_key[majority_label]}")

    print("test_prediction_distribution")
    for class_idx, count in sorted(prediction_counts.items(), key=lambda item: (-item[1], semantic_key_sort_key(id_to_key[item[0]]))):
        ratio = 100.0 * count / max(test_indices.numel(), 1)
        print(f"  pred_class={class_idx} count={count} ratio={ratio:.2f}% key={id_to_key[class_idx]}")

    print("test_class_accuracy")
    for class_idx in range(num_classes):
        print(
            f"  class={class_idx} size={int(class_sizes[class_idx])} "
            f"acc={float(class_accuracy[class_idx]):.4f} key={id_to_key[class_idx]}"
        )

    offdiag = confusion.clone()
    offdiag.fill_diagonal_(0)
    flat_counts = offdiag.flatten()
    ranked = torch.argsort(flat_counts, descending=True)
    printed = 0
    for flat_idx in ranked.tolist():
        count = int(flat_counts[flat_idx].item())
        if count <= 0 or printed >= top_confusions:
            break
        true_label = flat_idx // num_classes
        pred_label = flat_idx % num_classes
        printed += 1
        print(
            f"confusion_pair_rank_{printed}=true:{true_label} pred:{pred_label} count:{count} "
            f"true_key:{id_to_key[true_label]} pred_key:{id_to_key[pred_label]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--semantic-key-mode", choices=semantic_key_mode_choices())
    parser.add_argument("--min-class-size", type=int)
    parser.add_argument("--filter-attribute-values", action="append")
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-samples-by-attribute")
    parser.add_argument("--limit-samples-per-attribute-value", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0)
    parser.add_argument("--probe-train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-confusions", type=int, default=10)
    args = parser.parse_args()

    train_cfg = load_train_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False) if args.checkpoint else None

    semantic_key_mode = str(
        infer_arg(checkpoint, "semantic_key_mode", args.semantic_key_mode, cfg_get(train_cfg, "semantic_key_mode", "full"))
    )
    min_class_size = int(
        infer_arg(checkpoint, "min_class_size", args.min_class_size, cfg_get(train_cfg, "min_class_size", 1))
    )
    attribute_remap = parse_attribute_remap(
        infer_arg(checkpoint, "attribute_remap", None, cfg_get(train_cfg, "attribute_remap", None))
    )
    filter_attribute_values = parse_attribute_value_filters(
        infer_arg(
            checkpoint,
            "filter_attribute_values",
            (
                parse_attribute_value_filters(args.filter_attribute_values)
                if args.filter_attribute_values is not None
                else None
            ),
            cfg_get(train_cfg, "filter_attribute_values", None),
        )
    )
    limit = infer_arg(checkpoint, "limit_samples", args.limit_samples, cfg_get(train_cfg, "limit_samples", None))
    limit = int(limit) if limit is not None else None
    limit_by_attribute = infer_arg(
        checkpoint,
        "limit_samples_by_attribute",
        args.limit_samples_by_attribute,
        cfg_get(train_cfg, "limit_samples_by_attribute", None),
    )
    limit_per_value = infer_arg(
        checkpoint,
        "limit_samples_per_attribute_value",
        args.limit_samples_per_attribute_value,
        cfg_get(train_cfg, "limit_samples_per_attribute_value", None),
    )
    limit_per_value = int(limit_per_value) if limit_per_value is not None else None

    dataset = PreprocessedCSIDataset.from_pt(args.data_path)
    samples = apply_semantic_key_mode(dataset.samples, semantic_key_mode)
    samples = filter_samples_by_min_class_size(samples, min_class_size)
    samples = filter_samples_by_attribute_values(samples, filter_attribute_values)
    samples = limit_samples_by_attribute_value(
        samples,
        limit_by_attribute,
        limit_per_value,
        attribute_remap=attribute_remap,
    )
    samples = limit_samples(samples, limit)
    if not samples:
        raise ValueError("No samples remain after filtering.")

    print(f"data_path={args.data_path}")
    print(f"checkpoint={args.checkpoint}")
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"min_class_size={min_class_size}")
    print(f"attribute_remap={format_attribute_remap(attribute_remap)}")
    print(f"filter_attribute_values={format_attribute_value_filters(filter_attribute_values)}")
    print(f"limit_samples={limit}")
    print(f"limit_samples_by_attribute={limit_by_attribute}")
    print(f"limit_samples_per_attribute_value={limit_per_value}")

    tokenizer = build_tokenizer(samples, checkpoint)
    model = load_model(samples, tokenizer, checkpoint, device)
    features, labels, label_map = extract_features(
        model,
        samples,
        tokenizer,
        args.batch_size,
        device,
    )
    print(f"num_samples={features.shape[0]}")
    print(f"feature_dim={features.shape[1]}")
    print_class_distribution(label_map, labels)
    run_train_test_linear_probe(
        features=features,
        labels=labels,
        label_map=label_map,
        steps=args.probe_steps,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        train_fraction=args.probe_train_fraction,
        seed=args.seed,
        top_confusions=args.top_confusions,
    )


if __name__ == "__main__":
    main()
