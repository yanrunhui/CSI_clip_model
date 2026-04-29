from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.caption import CaptionGenerator
from data.dataset import PreprocessedCSIDataset, SyntheticCSIDataset, build_synthetic_samples, collate_fn
from data.semantic_key import SemanticKey
from data.tokenizer import CaptionTokenizer
from models.encoder import CSIEncoder
from models.model import CSIClip
from models.text_encoder import PhysicsTextEncoder
from training.scheduler import build_lr_scheduler
from training.trainer import TrainConfig, Trainer


def load_train_config(path: str | None) -> dict:
    config_path = Path(path) if path is not None else ROOT / "configs" / "train.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("train", data)


def cfg_get(config: dict, key: str, fallback):
    value = config.get(key, fallback)
    return fallback if value is None else value


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


def build_prototype_bank(
    samples,
    tokenizer: CaptionTokenizer,
    max_caption_len: int = 48,
) -> tuple[list[SemanticKey], list[str], torch.Tensor, torch.Tensor, dict[SemanticKey, int], torch.Tensor]:
    caption_generator = CaptionGenerator()
    unique_keys = sorted({sample.semantic_key for sample in samples}, key=semantic_key_sort_key)
    prototype_captions = [caption_generator.generate_canonical(key) for key in unique_keys]
    tokenizer.build_vocab(prototype_captions)
    tokenizer.build_vocab(sample.prop_caption for sample in samples)
    tokenizer.build_vocab(sample.instance_caption for sample in samples)
    prototype_token_ids = torch.stack(
        [tokenizer.encode(caption, max_len=max_caption_len).ids for caption in prototype_captions],
        dim=0,
    )
    prototype_token_mask = torch.stack(
        [tokenizer.encode(caption, max_len=max_caption_len).mask for caption in prototype_captions],
        dim=0,
    )
    prototype_label_map = {key: idx for idx, key in enumerate(unique_keys)}
    key_counts = Counter(sample.semantic_key for sample in samples)
    prototype_class_counts = torch.tensor(
        [key_counts[key] for key in unique_keys],
        dtype=torch.long,
    )
    return (
        unique_keys,
        prototype_captions,
        prototype_token_ids,
        prototype_token_mask,
        prototype_label_map,
        prototype_class_counts,
    )


def build_components_from_samples(
    samples,
    device: torch.device,
    batch_size: int = 128,
    temperature: float = 0.07,
):
    source_samples = samples if isinstance(samples, list) else samples.samples
    tokenizer = CaptionTokenizer()
    (
        prototype_keys,
        prototype_captions,
        prototype_token_ids,
        prototype_token_mask,
        prototype_label_map,
        prototype_class_counts,
    ) = build_prototype_bank(source_samples, tokenizer)

    dataset = SyntheticCSIDataset(source_samples) if isinstance(samples, list) else samples
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )

    csi_encoder = CSIEncoder(d_token=8, d_model=384, d_clip=256)
    text_encoder = PhysicsTextEncoder(vocab_size=max(tokenizer.next_id + 8, 300))
    model = CSIClip(
        csi_encoder,
        text_encoder,
        num_prototypes=len(prototype_keys),
        embed_dim=256,
        temperature=temperature,
    ).to(device)
    prototype_bank = {
        "keys": prototype_keys,
        "captions": prototype_captions,
        "token_ids": prototype_token_ids,
        "token_mask": prototype_token_mask,
        "label_map": prototype_label_map,
        "class_counts": prototype_class_counts,
    }
    return loader, model, tokenizer, prototype_bank


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


def build_demo_components(device: torch.device):
    caption_generator = CaptionGenerator()
    samples = build_synthetic_samples(128, caption_generator=caption_generator)
    return build_components_from_samples(samples, device=device, batch_size=32)


def build_real_components(
    data_path: str,
    device: torch.device,
    batch_size: int = 128,
    temperature: float = 0.07,
    min_class_size: int = 1,
):
    dataset = PreprocessedCSIDataset.from_pt(data_path)
    samples = filter_samples_by_min_class_size(dataset.samples, min_class_size=min_class_size)
    if len(samples) != len(dataset.samples):
        before_counts = Counter(sample.semantic_key for sample in dataset.samples)
        after_counts = Counter(sample.semantic_key for sample in samples)
        print(
            f"filtered classes with min_class_size={min_class_size}: "
            f"samples {len(dataset.samples)} -> {len(samples)}, "
            f"semantic_prototypes {len(before_counts)} -> {len(after_counts)}"
        )
    return build_components_from_samples(
        samples,
        device=device,
        batch_size=batch_size,
        temperature=temperature,
    )


def run_smoke_test(
    device: torch.device,
    text_mode: str = "prototype",
    aux_regression_weight: float = 0.0,
    multipositive_distance_threshold: float = 0.25,
    multipositive_positive_mode: str = "semantic_and_physics",
    min_class_size_for_multipositive: int = 2,
) -> None:
    loader, model, _, prototype_bank = build_demo_components(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = build_lr_scheduler(optimizer, total_epochs=2, warmup_epochs=1)
    trainer = Trainer(
        model,
        optimizer,
        device,
        prototype_token_ids=prototype_bank["token_ids"],
        prototype_token_mask=prototype_bank["token_mask"],
        prototype_label_map=prototype_bank["label_map"],
        prototype_class_counts=prototype_bank["class_counts"],
    )

    for epoch in range(1, 3):
        for batch in loader:
            metrics = trainer.train_step(
                batch,
                epoch=epoch,
                cfg=TrainConfig(
                    epochs=2,
                    text_mode=text_mode,
                    aux_regression_weight=aux_regression_weight,
                    multipositive_distance_threshold=multipositive_distance_threshold,
                    multipositive_positive_mode=multipositive_positive_mode,
                    min_class_size_for_multipositive=min_class_size_for_multipositive,
                ),
            )
            print(
                f"epoch={epoch} loss={metrics['loss_total']:.4f} "
                f"contrastive={metrics['contrastive_loss']:.4f}"
            )
            break
        scheduler.step()


def run_real_pretrain(
    data_path: str,
    device: torch.device,
    epochs: int,
    max_steps_per_epoch: int | None,
    lr: float,
    weight_decay: float,
    batch_size: int,
    temperature: float,
    warmup_epochs: int,
    min_lr: float,
    prototype_weight: float,
    text_prototype_weight: float,
    text_mode: str,
    aux_regression_weight: float,
    multipositive_distance_threshold: float,
    multipositive_positive_mode: str,
    min_class_size_for_multipositive: int,
    min_class_size: int,
    output_dir: str,
    save_every: int,
) -> None:
    loader, model, tokenizer, prototype_bank = build_real_components(
        data_path=data_path,
        device=device,
        batch_size=batch_size,
        temperature=temperature,
        min_class_size=min_class_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        total_epochs=epochs,
        warmup_epochs=min(warmup_epochs, epochs),
        min_lr_scale=min_lr / lr,
    )
    cfg = TrainConfig(
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        prototype_weight=prototype_weight,
        text_prototype_weight=text_prototype_weight,
        text_mode=text_mode,
        aux_regression_weight=aux_regression_weight,
        multipositive_distance_threshold=multipositive_distance_threshold,
        multipositive_positive_mode=multipositive_positive_mode,
        min_class_size_for_multipositive=min_class_size_for_multipositive,
    )
    trainer = Trainer(
        model,
        optimizer,
        device,
        prototype_token_ids=prototype_bank["token_ids"],
        prototype_token_mask=prototype_bank["token_mask"],
        prototype_label_map=prototype_bank["label_map"],
        prototype_class_counts=prototype_bank["class_counts"],
    )

    print(f"training on {data_path}")
    print(
        f"device={device} epochs={epochs} batch_size={batch_size} "
        f"temperature={temperature} warmup_epochs={warmup_epochs} min_lr={min_lr}"
    )
    print(
        f"semantic_prototypes={len(prototype_bank['keys'])} "
        f"prototype_weight={prototype_weight} text_prototype_weight={text_prototype_weight} "
        f"text_mode={text_mode} aux_regression_weight={aux_regression_weight} "
        f"multipositive_distance_threshold={multipositive_distance_threshold} "
        f"multipositive_positive_mode={multipositive_positive_mode} "
        f"min_class_size_for_multipositive={min_class_size_for_multipositive} "
        f"min_class_size={min_class_size}"
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "train_log.jsonl"

    for epoch in range(1, epochs + 1):
        epoch_metrics = []
        for step, batch in enumerate(loader, start=1):
            metrics = trainer.train_step(batch, epoch=epoch, cfg=cfg)
            epoch_metrics.append(metrics)
            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break
        scheduler.step()
        if not epoch_metrics:
            raise SystemExit("No batches were produced from the dataset.")

        mean_total = sum(m["loss_total"] for m in epoch_metrics) / len(epoch_metrics)
        mean_contrastive = sum(m["contrastive_loss"] for m in epoch_metrics) / len(epoch_metrics)
        mean_csi_to_text = sum(m["loss_csi_to_text"] for m in epoch_metrics) / len(epoch_metrics)
        mean_csi_to_prototype = sum(m["loss_csi_to_prototype"] for m in epoch_metrics) / len(epoch_metrics)
        mean_text_to_prototype = sum(m["loss_text_to_prototype"] for m in epoch_metrics) / len(epoch_metrics)
        mean_aux_regression = sum(m.get("loss_aux_regression", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_positive_count = sum(m.get("multipositive_positive_count_mean", 0.0) for m in epoch_metrics) / len(epoch_metrics)
        mean_logit_scale = sum(m["logit_scale"] for m in epoch_metrics) / len(epoch_metrics)
        print(
            f"epoch={epoch} steps={len(epoch_metrics)} "
            f"loss={mean_total:.4f} contrastive={mean_contrastive:.4f} "
            f"csi_to_text={mean_csi_to_text:.4f} csi_to_proto={mean_csi_to_prototype:.4f} "
            f"text_to_proto={mean_text_to_prototype:.4f} aux_reg={mean_aux_regression:.4f} "
            f"mp_pos={mean_positive_count:.1f} logit_scale={mean_logit_scale:.4f}"
        )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "steps": len(epoch_metrics),
                        "loss_total": mean_total,
                        "contrastive_loss": mean_contrastive,
                        "loss_csi_to_text": mean_csi_to_text,
                        "loss_csi_to_prototype": mean_csi_to_prototype,
                        "loss_text_to_prototype": mean_text_to_prototype,
                        "loss_aux_regression": mean_aux_regression,
                        "logit_scale": mean_logit_scale,
                        "text_mode": text_mode,
                        "aux_regression_weight": aux_regression_weight,
                        "multipositive_distance_threshold": multipositive_distance_threshold,
                        "multipositive_positive_mode": multipositive_positive_mode,
                        "min_class_size_for_multipositive": min_class_size_for_multipositive,
                        "min_class_size": min_class_size,
                        "multipositive_positive_count_mean": mean_positive_count,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )
                + "\n"
            )
        if epoch % save_every == 0 or epoch == epochs:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "tokenizer_word2id": tokenizer.word2id,
                "prototype_captions": prototype_bank["captions"],
                "args": {
                    "data_path": data_path,
                    "epochs": epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "temperature": temperature,
                    "warmup_epochs": warmup_epochs,
                    "min_lr": min_lr,
                    "prototype_weight": prototype_weight,
                    "text_prototype_weight": text_prototype_weight,
                    "text_mode": text_mode,
                    "aux_regression_weight": aux_regression_weight,
                    "multipositive_distance_threshold": multipositive_distance_threshold,
                    "multipositive_positive_mode": multipositive_positive_mode,
                    "min_class_size_for_multipositive": min_class_size_for_multipositive,
                    "min_class_size": min_class_size,
                    "phase": f"csi_clip_{text_mode}_text",
                },
            }
            ckpt_path = output_path / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, ckpt_path)
            torch.save(checkpoint, output_path / "checkpoint_last.pt")
            print(f"saved checkpoint to {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a synthetic end-to-end training step.")
    parser.add_argument("--data-path", type=str, help="Path to preprocessed .pt samples.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--prototype-weight", type=float)
    parser.add_argument("--text-prototype-weight", type=float)
    parser.add_argument("--text-mode", choices=["prototype", "instance", "multipositive"])
    parser.add_argument("--aux-regression-weight", type=float)
    parser.add_argument("--multipositive-distance-threshold", type=float)
    parser.add_argument(
        "--multipositive-positive-mode",
        choices=["semantic_and_physics", "semantic_or_physics", "semantic", "physics"],
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        help="Drop semantic classes with fewer than this many samples before training.",
    )
    parser.add_argument("--min-class-size-for-multipositive", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--save-every", type=int)
    args = parser.parse_args()

    train_cfg = load_train_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs if args.epochs is not None else int(cfg_get(train_cfg, "epochs", 3))
    lr = args.lr if args.lr is not None else float(cfg_get(train_cfg, "lr", 3e-4))
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(cfg_get(train_cfg, "weight_decay", 1e-2))
    )
    batch_size = args.batch_size if args.batch_size is not None else int(cfg_get(train_cfg, "batch_size", 128))
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(cfg_get(train_cfg, "temperature", 0.07))
    )
    warmup_epochs = (
        args.warmup_epochs
        if args.warmup_epochs is not None
        else int(cfg_get(train_cfg, "warmup_epochs", 5))
    )
    min_lr = args.min_lr if args.min_lr is not None else float(cfg_get(train_cfg, "min_lr", 1e-5))
    prototype_weight = (
        args.prototype_weight
        if args.prototype_weight is not None
        else float(cfg_get(train_cfg, "prototype_weight", 1.0))
    )
    text_prototype_weight = (
        args.text_prototype_weight
        if args.text_prototype_weight is not None
        else float(cfg_get(train_cfg, "text_prototype_weight", 1.0))
    )
    text_mode = args.text_mode if args.text_mode is not None else str(cfg_get(train_cfg, "text_mode", "prototype"))
    aux_regression_weight = (
        args.aux_regression_weight
        if args.aux_regression_weight is not None
        else float(cfg_get(train_cfg, "aux_regression_weight", 0.0))
    )
    multipositive_distance_threshold = (
        args.multipositive_distance_threshold
        if args.multipositive_distance_threshold is not None
        else float(cfg_get(train_cfg, "multipositive_distance_threshold", 0.25))
    )
    multipositive_positive_mode = (
        args.multipositive_positive_mode
        if args.multipositive_positive_mode is not None
        else str(cfg_get(train_cfg, "multipositive_positive_mode", "semantic_and_physics"))
    )
    min_class_size_for_multipositive = (
        args.min_class_size_for_multipositive
        if args.min_class_size_for_multipositive is not None
        else int(cfg_get(train_cfg, "min_class_size_for_multipositive", 2))
    )
    min_class_size = (
        args.min_class_size
        if args.min_class_size is not None
        else int(cfg_get(train_cfg, "min_class_size", 1))
    )
    output_dir = args.output_dir if args.output_dir is not None else str(cfg_get(train_cfg, "output_dir", "artifacts/pretrain_csi_clip"))
    save_every = args.save_every if args.save_every is not None else int(cfg_get(train_cfg, "save_every", 1))

    if args.smoke_test:
        run_smoke_test(
            device,
            text_mode=text_mode,
            aux_regression_weight=aux_regression_weight,
            multipositive_distance_threshold=multipositive_distance_threshold,
            multipositive_positive_mode=multipositive_positive_mode,
            min_class_size_for_multipositive=min_class_size_for_multipositive,
        )
        return

    data_path = args.data_path if args.data_path is not None else train_cfg.get("data_path")
    if data_path:
        run_real_pretrain(
            data_path=data_path,
            device=device,
            epochs=epochs,
            max_steps_per_epoch=args.max_steps_per_epoch,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            temperature=temperature,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
            prototype_weight=prototype_weight,
            text_prototype_weight=text_prototype_weight,
            text_mode=text_mode,
            aux_regression_weight=aux_regression_weight,
            multipositive_distance_threshold=multipositive_distance_threshold,
            multipositive_positive_mode=multipositive_positive_mode,
            min_class_size_for_multipositive=min_class_size_for_multipositive,
            min_class_size=min_class_size,
            output_dir=output_dir,
            save_every=save_every,
        )
        return

    raise SystemExit("Use --smoke-test, provide --data-path, or set train.data_path in the config.")


if __name__ == "__main__":
    main()
