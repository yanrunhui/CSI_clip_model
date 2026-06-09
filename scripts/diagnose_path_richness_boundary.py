from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (
    PreprocessedCSIDataset,
    collate_fn,
)
from data.semantic_key import (
    PROP_DISC,
    SemanticKey,
    discretize,
    semantic_key_attribute_value,
)
from scripts.pretrain import (
    assert_checkpoint_prototype_compatibility,
    build_real_components,
    cfg_get,
    format_attribute_remap,
    format_attribute_value_filters,
    load_model_state_compatible,
    load_train_config,
    load_transfer_checkpoint,
    parse_attribute_remap,
    parse_attribute_value_filters,
)


def infer_arg(checkpoint: dict | None, name: str, override, default):
    if override is not None:
        return override
    if checkpoint is not None:
        value = checkpoint.get("args", {}).get(name)
        if value is not None:
            return value
    return default


def raw_path_count_bin(n_paths: int) -> str:
    return discretize(float(n_paths), PROP_DISC["n_paths"])


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def format_distribution(
    counts: Counter,
    ordered_values: tuple[str, ...],
    total: int,
) -> str:
    return ",".join(
        f"{value}:{counts.get(value, 0)} ({100.0 * counts.get(value, 0) / max(total, 1):.2f}%)"
        for value in ordered_values
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--semantic-key-mode")
    parser.add_argument("--min-class-size", type=int)
    parser.add_argument("--filter-attribute-values", action="append")
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-samples-by-attribute")
    parser.add_argument("--limit-samples-per-attribute-value", type=int)
    parser.add_argument(
        "--focus-normalized-label",
        type=str,
        default="low",
        help="Normalized path_richness label to analyze in detail.",
    )
    args = parser.parse_args()

    train_cfg = load_train_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_transfer_checkpoint(args.checkpoint, device)

    batch_size = int(
        infer_arg(checkpoint, "batch_size", args.batch_size, cfg_get(train_cfg, "batch_size", 128))
    )
    semantic_key_mode = str(
        infer_arg(
            checkpoint,
            "semantic_key_mode",
            args.semantic_key_mode,
            cfg_get(train_cfg, "semantic_key_mode", "full"),
        )
    )
    min_class_size = int(
        infer_arg(
            checkpoint,
            "min_class_size",
            args.min_class_size,
            cfg_get(train_cfg, "min_class_size", 1),
        )
    )
    attribute_remap = parse_attribute_remap(
        infer_arg(checkpoint, "attribute_remap", None, cfg_get(train_cfg, "attribute_remap", None))
    )
    filter_attribute_values = parse_attribute_value_filters(
        infer_arg(
            checkpoint,
            "filter_attribute_values",
            args.filter_attribute_values,
            cfg_get(train_cfg, "filter_attribute_values", None),
        )
    )
    limit_samples = infer_arg(
        checkpoint,
        "limit_samples",
        args.limit_samples,
        cfg_get(train_cfg, "limit_samples", None),
    )
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

    base_loader, model, tokenizer, prototype_bank = build_real_components(
        data_path=args.data_path,
        device=device,
        batch_size=batch_size,
        temperature=float(
            infer_arg(checkpoint, "temperature", None, cfg_get(train_cfg, "temperature", 0.07))
        ),
        min_class_size=min_class_size,
        semantic_key_mode=semantic_key_mode,
        attribute_fields=("path_richness",),
        attribute_remap=attribute_remap,
        filter_attribute_values=filter_attribute_values,
        limit_samples=limit_samples,
        limit_samples_by_attribute=limit_samples_by_attribute,
        limit_samples_per_attribute_value=limit_samples_per_attribute_value,
        tokenizer_word2id=checkpoint.get("tokenizer_word2id"),
    )
    assert_checkpoint_prototype_compatibility(
        checkpoint,
        prototype_bank["keys"],
        expected_shape=tuple(model.prototypes.shape) if model.prototypes is not None else None,
        context="path richness boundary checkpoint",
    )
    load_model_state_compatible(model, checkpoint)
    model.eval()

    label_map = prototype_bank["attribute_label_maps"].get("path_richness")
    if not label_map:
        raise SystemExit(
            "No path_richness label map was constructed. "
            "Check that the filtered dataset still contains path_richness labels."
        )
    if "path_richness" not in model.attribute_classifiers:
        raise SystemExit(
            "Checkpoint/model does not include a path_richness attribute head. "
            "Use a checkpoint trained with path_richness in attribute_classifier_fields."
        )
    id_to_value = {idx: value for value, idx in label_map.items()}
    ordered_pred_values = tuple(id_to_value[idx] for idx in range(len(id_to_value)))

    samples = list(getattr(base_loader, "dataset", None).samples) if hasattr(base_loader, "dataset") else None
    if samples is None:
        raise RuntimeError("Failed to recover filtered samples from build_real_components loader.")
    loader = DataLoader(
        PreprocessedCSIDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )

    overall_confusion = Counter()
    raw_bin_counts = Counter()
    raw_bin_correct = Counter()
    raw_bin_predictions: dict[str, Counter] = defaultdict(Counter)
    focus_counts = Counter()
    focus_correct = Counter()
    focus_predictions: dict[str, Counter] = defaultdict(Counter)

    sample_offset = 0
    for batch in loader:
        batch_samples = samples[sample_offset : sample_offset + len(batch["semantic_keys"])]
        sample_offset += len(batch_samples)
        batch = move_batch(batch, device)
        csi_features_raw = model.encode_csi(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            normalize=False,
        )
        logits = model.predict_attributes(csi_features_raw)["path_richness"]
        predictions = logits.argmax(dim=1).cpu().tolist()

        for sample, true_key, pred_idx in zip(batch_samples, batch["semantic_keys"], predictions):
            if sample.semantic_key != true_key:
                raise ValueError(
                    "Batch/sample misalignment detected while analyzing path_richness boundary."
                )
            true_value = semantic_key_attribute_value(sample.semantic_key, "path_richness", attribute_remap)
            pred_value = id_to_value[pred_idx]
            raw_bin = raw_path_count_bin(int(sample.n_paths))

            overall_confusion[(true_value, pred_value)] += 1
            raw_bin_counts[raw_bin] += 1
            raw_bin_predictions[raw_bin][pred_value] += 1
            if pred_value == true_value:
                raw_bin_correct[raw_bin] += 1

            if true_value == args.focus_normalized_label:
                focus_counts[raw_bin] += 1
                focus_predictions[raw_bin][pred_value] += 1
                if pred_value == true_value:
                    focus_correct[raw_bin] += 1

    print(f"data_path={args.data_path}")
    print(f"checkpoint={args.checkpoint}")
    print(f"semantic_key_mode={semantic_key_mode}")
    print(f"min_class_size={min_class_size}")
    print(f"attribute_remap={format_attribute_remap(attribute_remap)}")
    print(f"filter_attribute_values={format_attribute_value_filters(filter_attribute_values)}")
    print(f"limit_samples={limit_samples}")
    print(f"limit_samples_by_attribute={limit_samples_by_attribute}")
    print(f"limit_samples_per_attribute_value={limit_samples_per_attribute_value}")
    print(f"path_richness_label_order={','.join(ordered_pred_values)}")
    print(f"focus_normalized_label={args.focus_normalized_label}")
    print(f"num_samples={len(samples)}")

    print("overall_path_richness_confusion")
    for true_value in ordered_pred_values:
        total_true = sum(overall_confusion[(true_value, pred_value)] for pred_value in ordered_pred_values)
        if total_true <= 0:
            continue
        distribution = format_distribution(
            Counter({pred: overall_confusion[(true_value, pred)] for pred in ordered_pred_values}),
            ordered_pred_values,
            total_true,
        )
        accuracy = 100.0 * overall_confusion[(true_value, true_value)] / total_true
        print(
            f"  true={true_value} count={total_true} acc={accuracy:.2f}% "
            f"pred_distribution={distribution}"
        )

    print("raw_path_bin_overall")
    raw_order = tuple(label for label in PROP_DISC["n_paths"] if raw_bin_counts[label] > 0)
    for raw_bin in raw_order:
        total = raw_bin_counts[raw_bin]
        accuracy = 100.0 * raw_bin_correct[raw_bin] / max(total, 1)
        distribution = format_distribution(
            raw_bin_predictions[raw_bin],
            ordered_pred_values,
            total,
        )
        normalized_true = semantic_key_attribute_value(
            SemanticKey(
                env_type="any",
                los_status="any",
                path_richness=raw_bin,
                ds_bin="any",
                as_az_bin="any",
                k_factor_bin="any",
            ),
            "path_richness",
            attribute_remap,
        )
        print(
            f"  raw={raw_bin} normalized_true={normalized_true} count={total} "
            f"acc={accuracy:.2f}% pred_distribution={distribution}"
        )

    focus_total = sum(focus_counts.values())
    focus_correct_total = sum(focus_correct.values())
    print(
        f"{args.focus_normalized_label}_overall=count:{focus_total} "
        f"acc:{100.0 * focus_correct_total / max(focus_total, 1):.2f}%"
    )
    print(f"{args.focus_normalized_label}_raw_breakdown")
    for raw_bin in raw_order:
        total = focus_counts[raw_bin]
        if total <= 0:
            continue
        accuracy = 100.0 * focus_correct[raw_bin] / total
        focus_to_moderate = 100.0 * focus_predictions[raw_bin].get("moderate", 0) / total
        distribution = format_distribution(
            focus_predictions[raw_bin],
            ordered_pred_values,
            total,
        )
        print(
            f"  raw={raw_bin} count={total} acc={accuracy:.2f}% "
            f"{args.focus_normalized_label}_to_moderate={focus_to_moderate:.2f}% "
            f"pred_distribution={distribution}"
        )


if __name__ == "__main__":
    main()
