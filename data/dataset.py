from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.utils.data import Dataset

from .caption import CaptionGenerator
from .semantic_key import SemanticKey
from .tokenizer import CaptionTokenizer

PHYSICS_TARGET_NAMES = (
    "n_paths",
    "delay_spread_ns",
    "azimuth_spread_deg",
    "k_factor_db",
    "first_path_delay_ns",
    "first_path_power_dbw",
    "first_path_aoa_az_deg",
    "reflection_count",
    "diffraction_count",
)

PHYSICS_TARGET_SCALES = torch.tensor(
    [10.0, 200.0, 90.0, 20.0, 3000.0, 30.0, 180.0, 10.0, 10.0],
    dtype=torch.float32,
)

PHYSICS_TARGET_OFFSETS = torch.tensor(
    [0.0, 0.0, 0.0, 0.0, 0.0, -100.0, 0.0, 0.0, 0.0],
    dtype=torch.float32,
)


@dataclass
class PreprocessedSample:
    tokens: torch.Tensor
    beam_positions: torch.Tensor
    config_key: str
    n_tokens: int
    freq_bin: int
    bw_bin: int
    subcarrier_spacing_hz: float
    group_id: str
    semantic_key: SemanticKey
    config_label: int
    obs_array_label: int
    obs_ant_label: int
    obs_freq_label: int
    obs_bw_label: int
    prop_caption: str
    instance_caption: str = ""
    n_paths: int = 0
    delay_spread_s: float = 0.0
    azimuth_spread_deg: float = 0.0
    k_factor_db: float = 0.0
    first_path_delay_s: float = math.nan
    first_path_power_dbw: float = math.nan
    first_path_aoa_az_deg: float = math.nan
    reflection_count: int = 0
    diffraction_count: int = 0


def _ensure_continuous_fields(sample: PreprocessedSample) -> PreprocessedSample:
    defaults = {
        "n_paths": 0,
        "delay_spread_s": 0.0,
        "azimuth_spread_deg": 0.0,
        "k_factor_db": 0.0,
        "first_path_delay_s": math.nan,
        "first_path_power_dbw": math.nan,
        "first_path_aoa_az_deg": math.nan,
        "reflection_count": 0,
        "diffraction_count": 0,
    }
    for field_name, default_value in defaults.items():
        if not hasattr(sample, field_name):
            setattr(sample, field_name, default_value)
    if not getattr(sample, "instance_caption", ""):
        setattr(sample, "instance_caption", CaptionGenerator().generate_instance_from_sample(sample))
    return sample


def _finite_or_nan(value: float | int) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def physics_raw_values(sample: PreprocessedSample) -> torch.Tensor:
    return torch.tensor(
        [
            _finite_or_nan(sample.n_paths),
            _finite_or_nan(sample.delay_spread_s) * 1e9,
            _finite_or_nan(sample.azimuth_spread_deg),
            _finite_or_nan(sample.k_factor_db),
            _finite_or_nan(sample.first_path_delay_s) * 1e9,
            _finite_or_nan(sample.first_path_power_dbw),
            _finite_or_nan(sample.first_path_aoa_az_deg),
            _finite_or_nan(sample.reflection_count),
            _finite_or_nan(sample.diffraction_count),
        ],
        dtype=torch.float32,
    )


def normalize_physics_targets(raw_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.isfinite(raw_targets)
    safe_targets = torch.where(mask, raw_targets, PHYSICS_TARGET_OFFSETS)
    normalized = (safe_targets - PHYSICS_TARGET_OFFSETS) / PHYSICS_TARGET_SCALES
    normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
    return normalized, mask


class SyntheticCSIDataset(Dataset[PreprocessedSample]):
    def __init__(self, samples: list[PreprocessedSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreprocessedSample:
        return self.samples[index]


class PreprocessedCSIDataset(Dataset[PreprocessedSample]):
    def __init__(self, samples: list[PreprocessedSample]):
        self.samples = samples

    @classmethod
    def from_pt(cls, path: str) -> "PreprocessedCSIDataset":
        samples = torch.load(path, weights_only=False)
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Expected a non-empty list of PreprocessedSample in {path}")
        return cls([_ensure_continuous_fields(sample) for sample in samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreprocessedSample:
        return self.samples[index]


def collate_fn(
    batch: list[PreprocessedSample],
    tokenizer: CaptionTokenizer,
    max_caption_len: int = 48,
) -> dict[str, torch.Tensor | list[str] | list[SemanticKey]]:
    k_max = max(sample.n_tokens for sample in batch)
    batch_size = len(batch)
    d_token = batch[0].tokens.shape[1]
    n_freq = batch[0].tokens.shape[2]

    tokens = torch.zeros(batch_size, k_max, d_token, n_freq, dtype=batch[0].tokens.dtype)
    beam_pos = torch.zeros(batch_size, k_max, 2, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, k_max, dtype=torch.bool)

    t_prop_ids = torch.zeros(batch_size, max_caption_len, dtype=torch.long)
    t_prop_mask = torch.zeros(batch_size, max_caption_len, dtype=torch.bool)
    t_instance_ids = torch.zeros(batch_size, max_caption_len, dtype=torch.long)
    t_instance_mask = torch.zeros(batch_size, max_caption_len, dtype=torch.bool)
    physics_raw_targets = torch.zeros(batch_size, len(PHYSICS_TARGET_NAMES), dtype=torch.float32)
    physics_targets = torch.zeros_like(physics_raw_targets)
    physics_target_mask = torch.zeros_like(physics_raw_targets, dtype=torch.bool)

    for i, sample in enumerate(batch):
        n_tokens = sample.n_tokens
        tokens[i, :n_tokens] = sample.tokens
        beam_pos[i, :n_tokens] = sample.beam_positions
        token_mask[i, :n_tokens] = True
        encoded = tokenizer.encode(sample.prop_caption, max_len=max_caption_len)
        t_prop_ids[i] = encoded.ids
        t_prop_mask[i] = encoded.mask
        instance_encoded = tokenizer.encode(sample.instance_caption, max_len=max_caption_len)
        t_instance_ids[i] = instance_encoded.ids
        t_instance_mask[i] = instance_encoded.mask
        raw_targets = physics_raw_values(sample)
        normalized_targets, target_mask = normalize_physics_targets(raw_targets)
        physics_raw_targets[i] = raw_targets
        physics_targets[i] = normalized_targets
        physics_target_mask[i] = target_mask

    return {
        "tokens": tokens,
        "beam_positions": beam_pos,
        "token_mask": token_mask,
        "freq_bin": torch.tensor([sample.freq_bin for sample in batch], dtype=torch.long),
        "bw_bin": torch.tensor([sample.bw_bin for sample in batch], dtype=torch.long),
        "subcarrier_spacing": torch.tensor(
            [sample.subcarrier_spacing_hz for sample in batch],
            dtype=torch.float32,
        ),
        "t_prop_ids": t_prop_ids,
        "t_prop_mask": t_prop_mask,
        "t_instance_ids": t_instance_ids,
        "t_instance_mask": t_instance_mask,
        "physics_targets": physics_targets,
        "physics_target_mask": physics_target_mask,
        "physics_raw_targets": physics_raw_targets,
        "instance_captions": [sample.instance_caption for sample in batch],
        "semantic_keys": [sample.semantic_key for sample in batch],
        "group_ids": [sample.group_id for sample in batch],
        "config_keys": [sample.config_key for sample in batch],
    }


def build_synthetic_samples(
    count: int,
    caption_generator: CaptionGenerator,
) -> list[PreprocessedSample]:
    samples: list[PreprocessedSample] = []
    keys = [
        SemanticKey("indoor", "los", "four", "low", "narrow", "strong", "short", "strong", "front", "none", "none"),
        SemanticKey("outdoor", "nlos", "seven_plus", "high", "wide", "weak", "long", "weak", "left", "heavy", "light"),
        SemanticKey("O2I", "nlos", "two", "moderate", "moderate", "moderate", "medium", "moderate", "right", "light", "none"),
    ]
    configs = [
        ("ULA-16", 4, 0, 1, 1),
        ("ULA-32", 8, 0, 1, 1),
        ("UPA-4x4", 4, 1, 1, 1),
        ("UPA-8x8", 16, 1, 2, 1),
    ]
    for idx in range(count):
        key = keys[idx % len(keys)]
        config_key, n_tokens, array_label, ant_label, bw_label = configs[idx % len(configs)]
        tokens = torch.randn(n_tokens, 8, 128)
        beam_positions = torch.rand(n_tokens, 2)
        caption = caption_generator.generate(key)
        instance_caption = caption_generator.generate_instance(
            key,
            n_paths=max(idx % 8, 1),
            delay_spread_s=float((idx % 3 + 1) * 50e-9),
            azimuth_spread_deg=float((idx % 3 + 1) * 15.0),
            k_factor_db=float(12.0 - (idx % 3) * 5.0),
            first_path_delay_s=float((idx % 3 + 1) * 800e-9),
            first_path_power_dbw=float(-80.0 - (idx % 3) * 10.0),
            first_path_aoa_az_deg=float((-60.0, 0.0, 60.0)[idx % 3]),
            reflection_count=idx % 5,
            diffraction_count=idx % 4,
            config_key=config_key,
            subcarrier_spacing_hz=float(15e3 * (2 ** (idx % 5))),
        )
        samples.append(
            PreprocessedSample(
                tokens=tokens,
                beam_positions=beam_positions,
                config_key=config_key,
                n_tokens=n_tokens,
                freq_bin=idx % 3,
                bw_bin=bw_label,
                subcarrier_spacing_hz=float(15e3 * (2 ** (idx % 5))),
                group_id=f"group-{idx // 4}",
                semantic_key=key,
                config_label=idx % len(configs),
                obs_array_label=array_label,
                obs_ant_label=ant_label,
                obs_freq_label=idx % 3,
                obs_bw_label=bw_label,
                prop_caption=caption,
                instance_caption=instance_caption,
                n_paths=max(idx % 8, 1),
                delay_spread_s=float((idx % 3 + 1) * 50e-9),
                azimuth_spread_deg=float((idx % 3 + 1) * 15.0),
                k_factor_db=float(12.0 - (idx % 3) * 5.0),
                first_path_delay_s=float((idx % 3 + 1) * 800e-9),
                first_path_power_dbw=float(-80.0 - (idx % 3) * 10.0),
                first_path_aoa_az_deg=float((-60.0, 0.0, 60.0)[idx % 3]),
                reflection_count=idx % 5,
                diffraction_count=idx % 4,
            )
        )
    return samples
