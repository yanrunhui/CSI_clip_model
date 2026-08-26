from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from data.dataset import CONFIG_FEATURE_DIM, _sample_configuration_features
from models.encoder import CSIEncoder
from scripts.qwen_csi_text_common import OUTPUT_FIELDS, target_response


MAPPED_SYSTEM_PROMPT = (
    "You analyze channel state information encoded in learned continuous prefix "
    "embeddings. Return exactly one JSON object and no markdown or explanation. "
    "Do not invent unsupported values. Use null when a value cannot be inferred."
)


def mapped_csi_prompt(sample) -> str:
    source_n_freq = int(
        getattr(sample, "source_n_freq", 0) or sample.tokens.shape[-1]
    )
    return (
        "Infer the physical channel facts from the learned CSI embedding prefix.\n"
        f"array_type={getattr(sample, 'array_type', '')}\n"
        f"array_rows={int(getattr(sample, 'array_rows', 0) or 0)}\n"
        f"array_cols={int(getattr(sample, 'array_cols', 0) or 0)}\n"
        f"source_num_subcarriers={source_n_freq}\n"
        f"bandwidth_hz={getattr(sample, 'bandwidth_hz', None)}\n"
        f"subcarrier_spacing_hz={getattr(sample, 'subcarrier_spacing_hz', None)}\n"
        "Return exactly one JSON object with these keys:\n"
        f"{json.dumps(OUTPUT_FIELDS, ensure_ascii=True, separators=(',', ':'))}\n"
        "environment and los_status must be inferred categorical strings or null. "
        "los_status, when known, must be exactly los or nlos. Numeric fields must "
        "be JSON numbers or null. description must be a concise string supported "
        "by the inferred fields. Do not emit markdown or explanations."
    )


def mapped_chat_text(tokenizer, sample, *, add_generation_prompt: bool = True) -> str:
    messages = [
        {"role": "system", "content": MAPPED_SYSTEM_PROMPT},
        {"role": "user", "content": mapped_csi_prompt(sample)},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def target_json_text(sample, tokenizer) -> str:
    response = target_response(sample)
    text = json.dumps(
        response,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    return text + (tokenizer.eos_token or "")


def collate_csi_samples(samples: list[Any]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty CSI batch.")
    n_freqs = {int(sample.tokens.shape[-1]) for sample in samples}
    d_tokens = {int(sample.tokens.shape[1]) for sample in samples}
    if len(n_freqs) != 1 or len(d_tokens) != 1:
        raise ValueError(
            "All samples in a CSI-prefix batch must share n_freq and d_token, "
            f"got n_freq={sorted(n_freqs)} d_token={sorted(d_tokens)}."
        )

    batch_size = len(samples)
    max_tokens = max(int(sample.n_tokens) for sample in samples)
    d_token = next(iter(d_tokens))
    n_freq = next(iter(n_freqs))
    max_antennas = max(
        int(sample.antenna_coordinates_wavelengths.shape[0]) for sample in samples
    )

    tokens = torch.zeros(batch_size, max_tokens, d_token, n_freq, dtype=torch.float32)
    beam_positions = torch.zeros(batch_size, max_tokens, 2, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    freq_bin = torch.zeros(batch_size, dtype=torch.long)
    bw_bin = torch.zeros(batch_size, dtype=torch.long)
    subcarrier_spacing = torch.zeros(batch_size, dtype=torch.float32)
    config_features = torch.zeros(batch_size, CONFIG_FEATURE_DIM, dtype=torch.float32)
    antenna_coordinates = torch.zeros(batch_size, max_antennas, 3, dtype=torch.float32)
    antenna_mask = torch.zeros(batch_size, max_antennas, dtype=torch.bool)

    for row, sample in enumerate(samples):
        count = int(sample.n_tokens)
        tokens[row, :count] = sample.tokens[:count].float()
        beam_positions[row, :count] = sample.beam_positions[:count].float()
        token_mask[row, :count] = True
        freq_bin[row] = int(sample.freq_bin)
        bw_bin[row] = int(sample.bw_bin)
        subcarrier_spacing[row] = float(sample.subcarrier_spacing_hz)
        config_features[row] = _sample_configuration_features(sample)
        coordinates = sample.antenna_coordinates_wavelengths.float()
        antenna_coordinates[row, : coordinates.shape[0]] = coordinates
        antenna_mask[row, : coordinates.shape[0]] = True

    return {
        "samples": samples,
        "tokens": tokens,
        "beam_positions": beam_positions,
        "token_mask": token_mask,
        "freq_bin": freq_bin,
        "bw_bin": bw_bin,
        "subcarrier_spacing": subcarrier_spacing,
        "config_features": config_features,
        "antenna_coordinates": antenna_coordinates,
        "antenna_mask": antenna_mask,
    }


def move_csi_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class CSIPrefixMapper(nn.Module):
    def __init__(
        self,
        *,
        d_token: int,
        qwen_hidden_size: int,
        prefix_length: int = 16,
        d_model: int = 384,
        d_clip: int = 256,
        projector_hidden_dim: int = 1024,
        token_norm_mode: str = "std",
        use_continuous_config_encoding: bool = False,
    ):
        super().__init__()
        if prefix_length <= 0:
            raise ValueError("prefix_length must be positive.")
        self.prefix_length = int(prefix_length)
        self.qwen_hidden_size = int(qwen_hidden_size)
        self.use_continuous_config_encoding = bool(use_continuous_config_encoding)
        self.csi_encoder = CSIEncoder(
            d_token=d_token,
            d_model=d_model,
            d_clip=d_clip,
            token_norm_mode=token_norm_mode,
            use_continuous_config_encoding=use_continuous_config_encoding,
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(d_clip),
            nn.Linear(d_clip, projector_hidden_dim),
            nn.GELU(),
            nn.Linear(projector_hidden_dim, prefix_length * qwen_hidden_size),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        features = self.csi_encoder(
            batch["tokens"],
            batch["beam_positions"],
            batch["token_mask"],
            batch["freq_bin"],
            batch["bw_bin"],
            batch["subcarrier_spacing"],
            config_features=(
                batch["config_features"]
                if self.use_continuous_config_encoding
                else None
            ),
            antenna_coordinates=(
                batch["antenna_coordinates"]
                if self.use_continuous_config_encoding
                else None
            ),
            antenna_mask=(
                batch["antenna_mask"]
                if self.use_continuous_config_encoding
                else None
            ),
        )
        prefix = self.projector(features)
        return prefix.reshape(
            features.shape[0], self.prefix_length, self.qwen_hidden_size
        )


def load_csi_encoder_checkpoint(
    mapper: CSIPrefixMapper,
    checkpoint_path: str,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"No state dictionary found in {checkpoint_path}.")

    prefixes = ("csi.", "module.csi.", "csi_encoder.", "module.csi_encoder.")
    extracted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix) :]] = value
                break
    if not extracted:
        raise ValueError(
            f"No CSI encoder keys found in {checkpoint_path}; tried {prefixes}."
        )
    incompatible = mapper.csi_encoder.load_state_dict(extracted, strict=False)
    return {
        "checkpoint": checkpoint_path,
        "loaded_key_count": len(extracted),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def build_training_embeddings(
    *,
    qwen,
    tokenizer,
    prefix_embeddings: torch.Tensor,
    samples: list[Any],
    max_length: int,
    assistant_prefill: str = "",
    response_prefix: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding_layer = qwen.get_input_embeddings()
    sequences = []
    labels = []
    for row, sample in enumerate(samples):
        prompt_ids = tokenizer(
            mapped_chat_text(tokenizer, sample) + assistant_prefill,
            add_special_tokens=False,
        )["input_ids"]
        answer_text = target_json_text(sample, tokenizer)
        if response_prefix:
            if not answer_text.startswith(response_prefix):
                raise ValueError(
                    "The target JSON does not start with --response-prefix: "
                    f"{response_prefix!r}."
                )
            answer_text = answer_text[len(response_prefix) :]
        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
        )["input_ids"]
        if len(answer_ids) + prefix_embeddings.shape[1] >= max_length:
            raise ValueError(
                "Answer and CSI prefix exceed max_length: "
                f"answer={len(answer_ids)} prefix={prefix_embeddings.shape[1]} "
                f"max_length={max_length}."
            )
        prompt_budget = max_length - prefix_embeddings.shape[1] - len(answer_ids)
        prompt_ids = prompt_ids[:prompt_budget]
        token_ids = torch.tensor(
            prompt_ids + answer_ids,
            dtype=torch.long,
            device=prefix_embeddings.device,
        )
        token_embeddings = embedding_layer(token_ids).to(prefix_embeddings.dtype)
        sequence = torch.cat([prefix_embeddings[row], token_embeddings], dim=0)
        sequence_labels = (
            [-100] * (prefix_embeddings.shape[1] + len(prompt_ids)) + answer_ids
        )
        sequences.append(sequence)
        labels.append(
            torch.tensor(sequence_labels, dtype=torch.long, device=sequence.device)
        )

    max_sequence = max(sequence.shape[0] for sequence in sequences)
    hidden_size = prefix_embeddings.shape[-1]
    batch_size = len(sequences)
    inputs_embeds = torch.zeros(
        batch_size,
        max_sequence,
        hidden_size,
        dtype=prefix_embeddings.dtype,
        device=prefix_embeddings.device,
    )
    attention_mask = torch.zeros(
        batch_size, max_sequence, dtype=torch.long, device=prefix_embeddings.device
    )
    padded_labels = torch.full(
        (batch_size, max_sequence),
        -100,
        dtype=torch.long,
        device=prefix_embeddings.device,
    )
    for row, (sequence, sequence_labels) in enumerate(zip(sequences, labels)):
        length = sequence.shape[0]
        inputs_embeds[row, :length] = sequence
        attention_mask[row, :length] = 1
        padded_labels[row, :length] = sequence_labels
    return inputs_embeds, attention_mask, padded_labels


@dataclass(frozen=True)
class MapperConfig:
    d_token: int
    qwen_hidden_size: int
    prefix_length: int
    d_model: int
    d_clip: int
    projector_hidden_dim: int
    token_norm_mode: str
    use_continuous_config_encoding: bool

    @classmethod
    def from_args(cls, args, qwen_hidden_size: int, d_token: int) -> "MapperConfig":
        return cls(
            d_token=d_token,
            qwen_hidden_size=qwen_hidden_size,
            prefix_length=args.prefix_length,
            d_model=args.encoder_d_model,
            d_clip=args.encoder_d_clip,
            projector_hidden_dim=args.projector_hidden_dim,
            token_norm_mode=args.token_norm_mode,
            use_continuous_config_encoding=args.use_continuous_config_encoding,
        )

    def build(self) -> CSIPrefixMapper:
        return CSIPrefixMapper(**self.__dict__)
