from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.caption import CaptionGenerator
from data.dataset import PreprocessedCSIDataset, collate_fn
from data.semantic_key import SemanticKey
from data.tokenizer import CaptionTokenizer
from models.encoder import CSIEncoder
from models.model import CSIClip
from models.text_encoder import PhysicsTextEncoder
from training.losses import cosine_alignment_loss


def build_tokenizer(samples, checkpoint: dict | None) -> CaptionTokenizer:
    tokenizer = CaptionTokenizer()
    if checkpoint is not None and "tokenizer_word2id" in checkpoint:
        tokenizer.word2id = dict(checkpoint["tokenizer_word2id"])
        tokenizer.id2word = {idx: word for word, idx in tokenizer.word2id.items()}
        tokenizer.next_id = max(tokenizer.id2word) + 1
    else:
        tokenizer.build_vocab(_build_prototype_captions(samples))
        tokenizer.build_vocab(sample.prop_caption for sample in samples)
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


@torch.no_grad()
def evaluate(data_path: str, checkpoint_path: str | None, batch_size: int, device: torch.device) -> None:
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) if checkpoint_path else None
    tokenizer = build_tokenizer(dataset.samples, checkpoint)
    prototype_keys, prototype_token_ids, prototype_token_mask, prototype_label_map = build_prototype_bank(
        dataset.samples,
        tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
    model = CSIClip(
        CSIEncoder(d_token=8, d_model=384, d_clip=256),
        PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300)),
        num_prototypes=len(prototype_keys),
        embed_dim=256,
    ).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    all_csi_features = []
    all_labels = []

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
        all_labels.extend(prototype_label_map[key] for key in batch["semantic_keys"])

    text_features = model.encode_text(
        prototype_token_ids.to(device),
        prototype_token_mask.to(device),
        normalize=True,
    ).cpu()
    prototype_features = model.encode_prototypes(normalize=True).cpu()
    csi_features = torch.cat(all_csi_features, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    logit_scale = float(model.logit_scale.exp().detach().cpu().item())
    logits = logit_scale * csi_features @ text_features.T
    prototype_logits = logit_scale * csi_features @ prototype_features.T

    instance_loss = F.cross_entropy(logits, labels)
    csi_prototype_loss = F.cross_entropy(prototype_logits, labels)
    text_prototype_loss = cosine_alignment_loss(
        text_features,
        prototype_features,
    )
    eval_loss = instance_loss + csi_prototype_loss + text_prototype_loss
    print(f"eval_learnable_prototype_loss={float(eval_loss):.4f}")
    print(f"eval_csi_to_text_loss={float(instance_loss):.4f}")
    print(f"eval_csi_to_prototype_loss={float(csi_prototype_loss):.4f}")
    print(f"eval_text_to_prototype_loss={float(text_prototype_loss):.4f}")
    print(f"logit_scale={logit_scale:.4f}")
    print(f"semantic_prototypes={len(prototype_keys)}")
    _print_retrieval_metrics("csi_to_text_proto", logits, labels)
    _print_retrieval_metrics("csi_to_learnable_prototype", prototype_logits, labels)

    _print_retrieval_metrics(
        "text_proto_to_learnable_prototype",
        logit_scale * text_features @ prototype_features.T,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args.data_path, args.checkpoint, args.batch_size, device)


if __name__ == "__main__":
    main()
