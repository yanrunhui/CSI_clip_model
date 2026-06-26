from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import torch
from torch.utils.data import Dataset

from .caption import CaptionGenerator
from .semantic_key import (
    PROP_DISC,
    SemanticKey,
    discretize,
    normalize_k_factor_bin,
    normalize_path_richness,
)
from .tokenizer import CaptionTokenizer

SEMANTIC_KEY_MODES = (
    "full",
    "coarse",
    "coarse_delay",
    "coarse_angle",
    "coarse_k",
    "coarse_k_angle",
    "coarse_interaction",
)

PHYSICS_TARGET_NAMES = (
    "n_paths",
    "delay_spread_ns",
    "azimuth_spread_deg",
    "k_factor_db",
    "first_path_delay_ns",
    "first_path_power_dbw",
    "first_path_aoa_az_sin",
    "first_path_aoa_az_cos",
    "reflection_count",
    "diffraction_count",
)

PHYSICS_AUX_TARGET_ALIASES = {
    "first_path_aoa_az_deg": ("first_path_aoa_az_sin", "first_path_aoa_az_cos"),
}

PHYSICS_AUX_TARGET_CHOICES = ("all", *PHYSICS_TARGET_NAMES, *PHYSICS_AUX_TARGET_ALIASES)

PHYSICS_TARGET_SCALES = torch.tensor(
    [10.0, 200.0, 90.0, 20.0, 3000.0, 30.0, 1.0, 1.0, 10.0, 10.0],
    dtype=torch.float32,
)

PHYSICS_TARGET_OFFSETS = torch.tensor(
    [0.0, 0.0, 0.0, 0.0, 0.0, -100.0, 0.0, 0.0, 0.0, 0.0],
    dtype=torch.float32,
)

DELAY_POWER_MAP_SHAPE = (32, 32)
DELAY_POWER_PROFILE_BINS = 64


def empty_delay_power_map() -> torch.Tensor:
    return torch.zeros(DELAY_POWER_MAP_SHAPE, dtype=torch.float32)


def empty_delay_power_profile() -> torch.Tensor:
    return torch.zeros(DELAY_POWER_PROFILE_BINS, dtype=torch.float32)


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
    los_delay_s: float = math.nan
    los_aoa_az_deg: float = math.nan
    first_path_power_dbw: float = math.nan
    first_path_aoa_az_deg: float = math.nan
    reflection_count: int = 0
    diffraction_count: int = 0
    delay_power_map: torch.Tensor = field(default_factory=empty_delay_power_map)
    delay_power_profile: torch.Tensor = field(default_factory=empty_delay_power_profile)


def _ensure_continuous_fields(sample: PreprocessedSample) -> PreprocessedSample:
    defaults = {
        "n_paths": 0,
        "delay_spread_s": 0.0,
        "azimuth_spread_deg": 0.0,
        "k_factor_db": 0.0,
        "first_path_delay_s": math.nan,
        "los_delay_s": math.nan,
        "los_aoa_az_deg": math.nan,
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
    delay_power_map = getattr(sample, "delay_power_map", None)
    if not isinstance(delay_power_map, torch.Tensor) or delay_power_map.shape != DELAY_POWER_MAP_SHAPE:
        setattr(sample, "delay_power_map", empty_delay_power_map())
    delay_power_profile = getattr(sample, "delay_power_profile", None)
    if not isinstance(delay_power_profile, torch.Tensor) or delay_power_profile.shape != (DELAY_POWER_PROFILE_BINS,):
        setattr(sample, "delay_power_profile", empty_delay_power_profile())
    return sample


def _finite_or_nan(value: float | int) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def physics_raw_values(sample: PreprocessedSample) -> torch.Tensor:
    first_path_aoa_az_deg = _finite_or_nan(sample.first_path_aoa_az_deg)
    first_path_aoa_az_rad = (
        math.radians(first_path_aoa_az_deg)
        if math.isfinite(first_path_aoa_az_deg)
        else math.nan
    )
    return torch.tensor(
        [
            _finite_or_nan(sample.n_paths),
            _finite_or_nan(sample.delay_spread_s) * 1e9,
            _finite_or_nan(sample.azimuth_spread_deg),
            _finite_or_nan(sample.k_factor_db),
            _finite_or_nan(sample.first_path_delay_s) * 1e9,
            _finite_or_nan(sample.first_path_power_dbw),
            math.sin(first_path_aoa_az_rad) if math.isfinite(first_path_aoa_az_rad) else math.nan,
            math.cos(first_path_aoa_az_rad) if math.isfinite(first_path_aoa_az_rad) else math.nan,
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


def semantic_key_mode_choices() -> tuple[str, ...]:
    return SEMANTIC_KEY_MODES


def physics_aux_target_choices() -> tuple[str, ...]:
    return PHYSICS_AUX_TARGET_CHOICES


def expand_physics_aux_targets(targets: tuple[str, ...]) -> tuple[str, ...]:
    if targets == ("all",):
        return targets
    expanded: list[str] = []
    for target in targets:
        expanded.extend(PHYSICS_AUX_TARGET_ALIASES.get(target, (target,)))
    return tuple(dict.fromkeys(expanded))


def semantic_key_for_mode(
    key: SemanticKey,
    mode: str,
    n_paths: int | None = None,
) -> SemanticKey:
    if mode == "full":
        return key
    path_richness = (
        normalize_path_richness(discretize(float(n_paths), PROP_DISC["n_paths"]))
        if n_paths is not None
        else normalize_path_richness(key.path_richness)
    )
    if mode in {
        "coarse",
        "coarse_delay",
        "coarse_angle",
        "coarse_k",
        "coarse_k_angle",
        "coarse_interaction",
    }:
        return SemanticKey(
            env_type=key.env_type,
            los_status=key.los_status,
            path_richness=path_richness,
            ds_bin=key.ds_bin if mode == "coarse_delay" else "any",
            as_az_bin=key.as_az_bin if mode in {"coarse_angle", "coarse_k_angle"} else "any",
            k_factor_bin=normalize_k_factor_bin(key.k_factor_bin) if mode in {"coarse_k", "coarse_k_angle"} else "any",
            first_delay_bin=key.first_delay_bin if mode not in {"coarse_k", "coarse_k_angle"} else "any",
            first_power_bin=key.first_power_bin,
            first_angle_bin=key.first_angle_bin if mode != "coarse_k" else "any",
            reflection_bin=key.reflection_bin if mode == "coarse_interaction" else "any",
            diffraction_bin=key.diffraction_bin if mode == "coarse_interaction" else "any",
        )
    raise ValueError(
        f"Unsupported semantic_key_mode={mode!r}. "
        f"Choose from: {', '.join(semantic_key_mode_choices())}"
    )


def apply_semantic_key_mode(samples: list[PreprocessedSample], mode: str) -> list[PreprocessedSample]:
    if mode == "full":
        return samples
    generator = CaptionGenerator()
    remapped = []
    for sample in samples:
        key = semantic_key_for_mode(sample.semantic_key, mode, n_paths=sample.n_paths)
        remapped_sample = replace(
            sample,
            semantic_key=key,
            prop_caption=generator.generate(key),
        )
        remapped_sample = replace(
            remapped_sample,
            instance_caption=generator.generate_instance_from_sample(remapped_sample),
        )
        remapped.append(remapped_sample)
    return remapped


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
    los_delay_raw_target = torch.zeros(batch_size, dtype=torch.float32)
    los_delay_target = torch.zeros(batch_size, dtype=torch.float32)
    los_delay_target_mask = torch.zeros(batch_size, dtype=torch.bool)
    los_angle_target = torch.zeros(batch_size, 2, dtype=torch.float32)
    los_angle_target_mask = torch.zeros(batch_size, dtype=torch.bool)
    delay_power_map = torch.zeros(
        batch_size,
        DELAY_POWER_MAP_SHAPE[0],
        DELAY_POWER_MAP_SHAPE[1],
        dtype=torch.float32,
    )
    delay_power_profile = torch.zeros(
        batch_size,
        DELAY_POWER_PROFILE_BINS,
        dtype=torch.float32,
    )

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
        los_delay_ns = _finite_or_nan(sample.los_delay_s) * 1e9
        if math.isfinite(los_delay_ns):
            los_delay_raw_target[i] = float(los_delay_ns)
            los_delay_target[i] = float(los_delay_ns) / 3000.0
            los_delay_target_mask[i] = True
        los_aoa_az_deg = _finite_or_nan(getattr(sample, "los_aoa_az_deg", math.nan))
        if math.isfinite(los_aoa_az_deg):
            los_aoa_az_rad = math.radians(los_aoa_az_deg)
            los_angle_target[i, 0] = math.sin(los_aoa_az_rad)
            los_angle_target[i, 1] = math.cos(los_aoa_az_rad)
            los_angle_target_mask[i] = True
        delay_power_map[i] = sample.delay_power_map.to(dtype=torch.float32)
        delay_power_profile[i] = sample.delay_power_profile.to(dtype=torch.float32)

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
        "los_delay_target": los_delay_target,
        "los_delay_target_mask": los_delay_target_mask,
        "los_delay_raw_target": los_delay_raw_target,
        "los_angle_target": los_angle_target,
        "los_angle_target_mask": los_angle_target_mask,
        "delay_power_map": delay_power_map,
        "delay_power_profile": delay_power_profile,
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
        SemanticKey("indoor", "los", "high", "low", "narrow", "strong", "short", "strong", "front", "none", "none"),
        SemanticKey("outdoor", "nlos", "high", "high", "wide", "weak", "long", "weak", "left", "heavy", "light"),
        SemanticKey("O2I", "nlos", "low", "moderate", "moderate", "weak", "medium", "moderate", "right", "light", "none"),
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
