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
from data.semantic_key import (
    SemanticKey,
    implied_attribute_value_filters,
    semantic_key_attribute_raw_value,
    semantic_key_attribute_value,
    semantic_key_field_choices,
)
from data.tokenizer import CaptionTokenizer
from models.encoder import CSIEncoder
from models.model import CSIClip
from models.text_encoder import PhysicsTextEncoder
from scripts.pretrain import assert_checkpoint_prototype_compatibility


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


def parse_attribute_remap(value) -> dict[str, dict[str, tuple[str, ...]]]:
    if value is None:
        return {}
    remap: dict[str, dict[str, tuple[str, ...]]] = {}
    for field, label_map in value.items():
        remap[str(field)] = {}
        for mapped_value, source_values in label_map.items():
            if isinstance(source_values, str):
                values = (source_values,)
            else:
                values = tuple(str(source_value) for source_value in source_values)
            remap[str(field)][str(mapped_value)] = values
    return remap


def build_attribute_label_map(
    samples,
    attribute_field: str,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]] | None = None,
) -> dict[str, int]:
    values = sorted(
        {
            semantic_key_attribute_value(sample.semantic_key, attribute_field, attribute_remap)
            for sample in samples
        }
    )
    return {value: idx for idx, value in enumerate(values)}


def load_model(
    samples,
    tokenizer: CaptionTokenizer,
    checkpoint: dict | None,
    attribute_field: str,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]],
    device: torch.device,
) -> CSIClip:
    unique_keys = sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
    label_map = build_attribute_label_map(samples, attribute_field, attribute_remap)
    model = CSIClip(
        CSIEncoder(d_token=8, d_model=384, d_clip=256),
        PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300)),
        num_prototypes=len(unique_keys),
        semantic_num_classes=len(unique_keys),
        embed_dim=256,
        num_physics_targets=len(PHYSICS_TARGET_NAMES),
        attribute_num_classes={attribute_field: len(label_map)},
    ).to(device)
    if checkpoint is not None:
        assert_checkpoint_prototype_compatibility(
            checkpoint,
            unique_keys,
            expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
            context="feature separability checkpoint",
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
    return limited


def limit_samples(samples, limit: int | None):
    if limit is None:
        return samples
    return samples[:limit]


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def extract_features(
    model: CSIClip,
    samples,
    tokenizer: CaptionTokenizer,
    attribute_field: str,
    attribute_remap: dict[str, dict[str, tuple[str, ...]]],
    batch_size: int,
    device: torch.device,
):
    label_map = build_attribute_label_map(samples, attribute_field, attribute_remap)
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
        labels.extend(
            label_map[semantic_key_attribute_value(key, attribute_field, attribute_remap)]
            for key in batch["semantic_keys"]
        )
    return torch.cat(features, dim=0), torch.tensor(labels, dtype=torch.long), label_map


def print_feature_geometry(features: torch.Tensor, labels: torch.Tensor, label_map: dict[str, int]) -> None:
    id_to_value = {idx: value for value, idx in label_map.items()}
    centered = features - features.mean(dim=0, keepdim=True)
    rank = int(torch.linalg.matrix_rank(centered).item())
    singular_values = torch.linalg.svdvals(centered)
    sample_norms = features.norm(dim=1)
    print(f"num_samples={features.shape[0]}")
    print(f"feature_dim={features.shape[1]}")
    print(f"feature_rank_centered={rank}")
    print(f"feature_std_global={float(features.std()):.6f}")
    print(f"feature_sample_norm_mean={float(sample_norms.mean()):.6f}")
    print(f"feature_sample_norm_std={float(sample_norms.std()):.6f}")
    print(f"feature_top_singular_values={','.join(f'{float(value):.6f}' for value in singular_values[:5])}")

    for class_idx in range(len(label_map)):
        class_count = int((labels == class_idx).sum().item())
        print(f"class_{class_idx}_value={id_to_value[class_idx]} count={class_count}")

    centroids = torch.stack([features[labels == class_idx].mean(dim=0) for class_idx in range(len(label_map))])
    centroid_distances = torch.cdist(centroids, centroids)
    centroid_cosine = F.normalize(centroids, dim=1) @ F.normalize(centroids, dim=1).T
    offdiag = ~torch.eye(len(label_map), dtype=torch.bool)
    if len(label_map) > 1:
        print(f"centroid_distance_mean={float(centroid_distances[offdiag].mean()):.6f}")
        print(f"centroid_distance_min={float(centroid_distances[offdiag].min()):.6f}")
        print(f"centroid_cosine_mean={float(centroid_cosine[offdiag].mean()):.6f}")
        print(f"centroid_cosine_max={float(centroid_cosine[offdiag].max()):.6f}")

    distances = torch.cdist(features, features)
    eye = torch.eye(features.shape[0], dtype=torch.bool)
    same = (labels[:, None] == labels[None, :]) & ~eye
    different = labels[:, None] != labels[None, :]
    if bool(same.any()):
        print(f"pairwise_same_distance_mean={float(distances[same].mean()):.6f}")
        print(f"pairwise_same_distance_std={float(distances[same].std()):.6f}")
    if bool(different.any()):
        print(f"pairwise_diff_distance_mean={float(distances[different].mean()):.6f}")
        print(f"pairwise_diff_distance_std={float(distances[different].std()):.6f}")
        if bool(same.any()):
            ratio = distances[different].mean() / distances[same].mean().clamp(min=1e-8)
            print(f"pairwise_diff_to_same_ratio={float(ratio):.6f}")

    nearest_centroid_predictions = torch.cdist(features, centroids).argmin(dim=1)
    nearest_centroid_acc = (nearest_centroid_predictions == labels).float().mean()
    print(f"nearest_centroid_train_acc={float(nearest_centroid_acc):.6f}")

    masked_distances = distances + torch.eye(features.shape[0]) * 1e9
    nearest_neighbor_predictions = labels[masked_distances.argmin(dim=1)]
    nearest_neighbor_acc = (nearest_neighbor_predictions == labels).float().mean()
    print(f"nearest_neighbor_leave_one_out_acc={float(nearest_neighbor_acc):.6f}")


def run_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    x = (features - features.mean(dim=0, keepdim=True)) / features.std(dim=0, keepdim=True).clamp(min=1e-6)
    num_classes = int(labels.max().item()) + 1
    probe = torch.nn.Linear(x.shape[1], num_classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = probe(x)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        acc = (predictions == labels).float().mean()
        logit_std = logits.float().std()
    print(f"linear_probe_train_loss={float(loss):.6f}")
    print(f"linear_probe_train_acc={float(acc):.6f}")
    print(f"linear_probe_logit_std={float(logit_std):.6f}")


def stratified_train_test_split(
    labels: torch.Tensor,
    train_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def run_train_test_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    lr: float,
    weight_decay: float,
    train_fraction: float,
    seed: int,
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
        train_acc = (train_logits.argmax(dim=1) == train_labels).float().mean()
        test_acc = (test_logits.argmax(dim=1) == test_labels).float().mean()
        test_majority = torch.bincount(train_labels, minlength=num_classes).argmax()
        test_majority_acc = (test_labels == test_majority).float().mean()

    print(f"linear_probe_split_train_count={train_indices.numel()}")
    print(f"linear_probe_split_test_count={test_indices.numel()}")
    print(f"linear_probe_split_train_loss={float(train_loss):.6f}")
    print(f"linear_probe_split_test_loss={float(test_loss):.6f}")
    print(f"linear_probe_split_train_acc={float(train_acc):.6f}")
    print(f"linear_probe_split_test_acc={float(test_acc):.6f}")
    print(f"linear_probe_split_test_majority_baseline={float(test_majority_acc):.6f}")


def infer_arg(checkpoint: dict | None, name: str, override, default):
    if override is not None:
        return override
    if checkpoint is not None:
        value = checkpoint.get("args", {}).get(name)
        if value is not None:
            return value
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--attribute-field", choices=semantic_key_field_choices(), default="los_status")
    parser.add_argument("--semantic-key-mode", choices=semantic_key_mode_choices())
    parser.add_argument("--min-class-size", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-samples-by-attribute", choices=semantic_key_field_choices())
    parser.add_argument("--limit-samples-per-attribute-value", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=0.0)
    parser.add_argument("--probe-train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False) if args.checkpoint else None
    semantic_key_mode = str(infer_arg(checkpoint, "semantic_key_mode", args.semantic_key_mode, "full"))
    attribute_remap = parse_attribute_remap(
        checkpoint.get("args", {}).get("attribute_remap")
        if checkpoint is not None
        else None
    )
    min_class_size = int(infer_arg(checkpoint, "min_class_size", args.min_class_size, 1))
    limit = infer_arg(checkpoint, "limit_samples", args.limit_samples, None)
    limit = int(limit) if limit is not None else None
    limit_by_attribute = infer_arg(checkpoint, "limit_samples_by_attribute", args.limit_samples_by_attribute, None)
    limit_per_value = infer_arg(checkpoint, "limit_samples_per_attribute_value", args.limit_samples_per_attribute_value, None)
    limit_per_value = int(limit_per_value) if limit_per_value is not None else None

    dataset = PreprocessedCSIDataset.from_pt(args.data_path)
    samples = apply_semantic_key_mode(dataset.samples, semantic_key_mode)
    samples = filter_samples_by_min_class_size(samples, min_class_size)
    samples = filter_samples_by_attribute_values(
        samples,
        implied_attribute_value_filters((args.attribute_field,), attribute_remap),
    )
    samples = limit_samples_by_attribute_value(
        samples,
        limit_by_attribute,
        limit_per_value,
        attribute_remap=attribute_remap,
    )
    samples = limit_samples(samples, limit)
    if not samples:
        raise ValueError("No samples remain after filtering.")

    value_counts = Counter(
        semantic_key_attribute_value(sample.semantic_key, args.attribute_field, attribute_remap)
        for sample in samples
    )
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"min_class_size={min_class_size}")
    print(f"limit_samples={limit}")
    print(f"limit_samples_by_attribute={limit_by_attribute}")
    print(f"limit_samples_per_attribute_value={limit_per_value}")
    print(f"attribute_field={args.attribute_field}")
    print(f"attribute_remap_fields={','.join(sorted(attribute_remap)) if attribute_remap else 'none'}")
    print(f"attribute_value_counts={','.join(f'{value}:{count}' for value, count in sorted(value_counts.items()))}")

    tokenizer = build_tokenizer(samples, checkpoint)
    model = load_model(samples, tokenizer, checkpoint, args.attribute_field, attribute_remap, device)
    features, labels, label_map = extract_features(
        model,
        samples,
        tokenizer,
        args.attribute_field,
        attribute_remap,
        args.batch_size,
        device,
    )
    print_feature_geometry(features, labels, label_map)
    run_linear_probe(
        features,
        labels,
        steps=args.probe_steps,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        seed=args.seed,
    )
    run_train_test_linear_probe(
        features,
        labels,
        steps=args.probe_steps,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        train_fraction=args.probe_train_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
