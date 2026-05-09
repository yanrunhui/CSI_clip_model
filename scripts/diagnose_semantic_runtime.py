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
    cfg_get,
    format_attribute_remap,
    format_attribute_value_filters,
    load_model_state_compatible,
    load_train_config,
    load_transfer_checkpoint,
    parse_attribute_fields,
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


def format_histogram(values: torch.Tensor) -> str:
    return ",".join(str(int(value)) for value in values.tolist())


def format_keyed_values(keys, values: torch.Tensor, precision: int = 4) -> str:
    return ";".join(
        f"{key}:{float(value):.{precision}f}"
        for key, value in zip(keys, values.tolist())
    )


def format_vector(values: torch.Tensor, precision: int = 6, limit: int = 8) -> str:
    clipped = values[:limit]
    suffix = ",..." if values.numel() > limit else ""
    return ",".join(f"{float(value):.{precision}f}" for value in clipped.tolist()) + suffix


def mean_offdiag_cosine(features: torch.Tensor) -> float:
    if features.shape[0] <= 1:
        return 0.0
    normalized = F.normalize(features.detach().float(), dim=-1)
    cosine = normalized @ normalized.T
    mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
    return float(cosine[mask].mean())


def mean_pairwise_l2(features: torch.Tensor) -> float:
    if features.shape[0] <= 1:
        return 0.0
    distances = torch.pdist(features.detach().float(), p=2)
    return float(distances.mean()) if distances.numel() else 0.0


def last_linear(module: torch.nn.Module | None) -> torch.nn.Linear | None:
    if module is None:
        return None
    if isinstance(module, torch.nn.Linear):
        return module
    if isinstance(module, torch.nn.Sequential):
        for layer in reversed(module):
            if isinstance(layer, torch.nn.Linear):
                return layer
    for child in reversed(tuple(module.children())):
        linear = last_linear(child)
        if linear is not None:
            return linear
    return None


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--semantic-key-mode")
    parser.add_argument("--min-class-size", type=int)
    parser.add_argument("--filter-attribute-values", action="append")
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-samples-by-attribute")
    parser.add_argument("--limit-samples-per-attribute-value", type=int)
    parser.add_argument("--attribute-classifier-fields", nargs="+")
    parser.add_argument("--batches", type=int, default=3)
    args = parser.parse_args()

    train_cfg = load_train_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_transfer_checkpoint(args.checkpoint, device) if args.checkpoint else None

    batch_size = int(infer_arg(checkpoint, "batch_size", args.batch_size, cfg_get(train_cfg, "batch_size", 128)))
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
            context="semantic runtime checkpoint",
        )
        load_model_state_compatible(model, checkpoint)
    model.eval()

    if model.semantic_classifier is None:
        raise SystemExit("Model does not have a semantic classifier head.")

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
    print(f"semantic_prototypes={len(prototype_bank['keys'])}")

    semantic_head_linear = last_linear(model.semantic_classifier)
    if semantic_head_linear is not None and semantic_head_linear.bias is not None:
        bias = semantic_head_linear.bias.detach().cpu()
        print(f"semantic_head_bias_mean={float(bias.mean()):.6f}")
        print(f"semantic_head_bias_std={float(bias.std()):.6f}")
        print(f"semantic_head_bias_argmax={int(bias.argmax().item())}")
        print(
            "semantic_head_bias_by_key="
            + format_keyed_values(prototype_bank["keys"], bias)
        )

    prev_batch_csi_mean = None
    prev_batch_logit_mean = None
    prev_batch_tokens_mean = None

    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > args.batches:
            break
        labels = torch.tensor(
            [prototype_bank["label_map"][key] for key in batch["semantic_keys"]],
            device=device,
            dtype=torch.long,
        )
        for sample_idx, (key, label) in enumerate(zip(batch["semantic_keys"], labels.cpu().tolist())):
            roundtrip_key = prototype_bank["keys"][label]
            if roundtrip_key != key:
                raise ValueError(
                    f"Roundtrip mismatch in batch {batch_idx} sample {sample_idx}: "
                    f"key={key} label={label} roundtrip_key={roundtrip_key}."
                )

        tokens = batch["tokens"].to(device)
        beam_positions = batch["beam_positions"].to(device)
        token_mask = batch["token_mask"].to(device)
        freq_bin = batch["freq_bin"].to(device)
        bw_bin = batch["bw_bin"].to(device)
        subcarrier_spacing = batch["subcarrier_spacing"].to(device)
        csi_features_raw = model.encode_csi(
            tokens,
            beam_positions,
            token_mask,
            freq_bin,
            bw_bin,
            subcarrier_spacing,
            normalize=False,
        )
        semantic_logits = model.predict_semantic(csi_features_raw)
        predictions = semantic_logits.argmax(dim=1)
        label_histogram = torch.bincount(labels.cpu(), minlength=len(prototype_bank["keys"]))
        prediction_histogram = torch.bincount(predictions.cpu(), minlength=len(prototype_bank["keys"]))
        loss = F.cross_entropy(semantic_logits, labels)
        mean_logits_by_class = semantic_logits.detach().float().mean(dim=0).cpu()
        mean_csi_feature = csi_features_raw.detach().float().mean(dim=0).cpu()
        token_batch_mean = tokens.detach().float().mean().cpu()
        token_batch_std = tokens.detach().float().std().cpu()
        normalized_tokens = (
            model.csi._normalize_tokens(tokens, token_mask)
            if hasattr(model.csi, "_normalize_tokens")
            else tokens
        )
        normalized_token_mean = normalized_tokens.detach().float().mean().cpu()
        normalized_token_std = normalized_tokens.detach().float().std().cpu()
        normalized_token_abs_mean = normalized_tokens.detach().float().abs().mean().cpu()
        csi_feature_norms = csi_features_raw.detach().float().norm(dim=1).cpu()
        logit_row_std = semantic_logits.detach().float().std(dim=1).cpu()
        mean_prediction = prediction_histogram.argmax().item()

        print(f"batch_{batch_idx}_semantic_loss={float(loss):.6f}")
        print(f"batch_{batch_idx}_tokens_mean={float(token_batch_mean):.8f}")
        print(f"batch_{batch_idx}_tokens_std={float(token_batch_std):.8f}")
        print(f"batch_{batch_idx}_normalized_tokens_mean={float(normalized_token_mean):.8f}")
        print(f"batch_{batch_idx}_normalized_tokens_std={float(normalized_token_std):.8f}")
        print(f"batch_{batch_idx}_normalized_tokens_abs_mean={float(normalized_token_abs_mean):.8f}")
        print(
            f"batch_{batch_idx}_label_histogram="
            f"{format_histogram(label_histogram)}"
        )
        print(
            f"batch_{batch_idx}_semantic_argmax_histogram="
            f"{format_histogram(prediction_histogram)}"
        )
        print(
            f"batch_{batch_idx}_label_majority_fraction="
            f"{label_histogram.max().item() / max(int(label_histogram.sum().item()), 1):.6f}"
        )
        print(
            f"batch_{batch_idx}_semantic_argmax_majority_fraction="
            f"{prediction_histogram.max().item() / max(int(prediction_histogram.sum().item()), 1):.6f}"
        )
        print(f"batch_{batch_idx}_semantic_argmax_mode={mean_prediction}")
        print(
            f"batch_{batch_idx}_semantic_logits_mean="
            f"{float(semantic_logits.detach().float().mean()):.6f}"
        )
        print(
            f"batch_{batch_idx}_semantic_logits_std="
            f"{float(semantic_logits.detach().float().std()):.6f}"
        )
        print(
            f"batch_{batch_idx}_semantic_logits_max_mean="
            f"{float(semantic_logits.detach().float().max(dim=1).values.mean()):.6f}"
        )
        print(
            f"batch_{batch_idx}_semantic_logit_row_std_mean="
            f"{float(logit_row_std.mean()):.8f}"
        )
        print(
            f"batch_{batch_idx}_semantic_logit_row_std_min="
            f"{float(logit_row_std.min()):.8f}"
        )
        print(
            f"batch_{batch_idx}_csi_feature_mean={float(csi_features_raw.detach().float().mean()):.8f}"
        )
        print(
            f"batch_{batch_idx}_csi_feature_std={float(csi_features_raw.detach().float().std()):.8f}"
        )
        print(
            f"batch_{batch_idx}_csi_feature_norm_mean={float(csi_feature_norms.mean()):.8f}"
        )
        print(
            f"batch_{batch_idx}_csi_feature_norm_std={float(csi_feature_norms.std()):.8f}"
        )
        print(
            f"batch_{batch_idx}_csi_feature_pairwise_l2_mean={mean_pairwise_l2(csi_features_raw):.8f}"
        )
        print(
            f"batch_{batch_idx}_csi_feature_pairwise_cosine_mean={mean_offdiag_cosine(csi_features_raw):.8f}"
        )
        print(
            f"batch_{batch_idx}_csi_feature_mean_signature={format_vector(mean_csi_feature)}"
        )
        print(
            f"batch_{batch_idx}_semantic_mean_logit_signature={format_vector(mean_logits_by_class, limit=len(prototype_bank['keys']))}"
        )
        print(
            f"batch_{batch_idx}_semantic_mean_logit_by_key="
            f"{format_keyed_values(prototype_bank['keys'], mean_logits_by_class)}"
        )
        if prev_batch_tokens_mean is not None:
            print(
                f"batch_{batch_idx}_prev_batch_tokens_mean_abs_diff="
                f"{abs(float(token_batch_mean) - float(prev_batch_tokens_mean)):.8f}"
            )
        if prev_batch_csi_mean is not None:
            print(
                f"batch_{batch_idx}_prev_batch_csi_mean_vector_max_abs_diff="
                f"{float((mean_csi_feature - prev_batch_csi_mean).abs().max()):.8f}"
            )
            print(
                f"batch_{batch_idx}_prev_batch_csi_mean_vector_mean_abs_diff="
                f"{float((mean_csi_feature - prev_batch_csi_mean).abs().mean()):.8f}"
            )
        if prev_batch_logit_mean is not None:
            print(
                f"batch_{batch_idx}_prev_batch_semantic_mean_logit_max_abs_diff="
                f"{float((mean_logits_by_class - prev_batch_logit_mean).abs().max()):.8f}"
            )
            print(
                f"batch_{batch_idx}_prev_batch_semantic_mean_logit_mean_abs_diff="
                f"{float((mean_logits_by_class - prev_batch_logit_mean).abs().mean()):.8f}"
            )

        prev_batch_tokens_mean = token_batch_mean
        prev_batch_csi_mean = mean_csi_feature
        prev_batch_logit_mean = mean_logits_by_class


if __name__ == "__main__":
    main()
