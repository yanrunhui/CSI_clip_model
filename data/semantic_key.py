from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

PATH_RICHNESS_LABELS = (
    "low",
    "moderate",
    "high",
)

FIRST_POWER_DBW_BINS = (
    ("very_weak", float("-inf"), -180),
    ("weak", -180, -140),
    ("moderate", -140, -100),
    ("strong", -100, float("inf")),
)
FIRST_POWER_DBW_BIN_LABELS = tuple(label for label, _, _ in FIRST_POWER_DBW_BINS)
FIRST_POWER_DBW_POSITION_BINS = (
    ("very_weak", -220.0, -180.0),
    ("weak", -180.0, -140.0),
    ("moderate", -140.0, -100.0),
    ("strong", -100.0, -60.0),
)

PROP_DISC = {
    "n_paths": {
        "zero": (0, 1),
        "one": (1, 2),
        "two": (2, 3),
        "three": (3, 4),
        "four": (4, 5),
        "five": (5, 6),
        "six": (6, 7),
        "seven_plus": (7, float("inf")),
    },
    "delay_spread_ns": {
        "low": (0, 50),
        "moderate": (50, 200),
        "high": (200, float("inf")),
    },
    "first_delay_ns": {
        "unknown": (float("nan"), float("nan")),
        "short": (0, 1500),
        "medium": (1500, 3000),
        "long": (3000, float("inf")),
    },
    "first_power_dbw": {
        "unknown": (float("nan"), float("nan")),
        **{
            label: (lower, upper)
            for label, lower, upper in FIRST_POWER_DBW_BINS
        },
    },
    "azimuth_spread_deg": {
        "narrow": (0, 10),
        "moderate": (10, 40),
        "wide": (40, float("inf")),
    },
    "k_factor_db": {
        "strong": (3, float("inf")),
        "weak": (float("-inf"), 3),
    },
    "interaction_count": {
        "unknown": (float("nan"), float("nan")),
        "none": (0, 1),
        "light": (1, 4),
        "heavy": (4, float("inf")),
    },
}


@dataclass(frozen=True)
class SemanticKey:
    env_type: str
    los_status: str
    path_richness: str
    ds_bin: str
    as_az_bin: str
    k_factor_bin: str
    first_delay_bin: str = "unknown"
    first_power_bin: str = "unknown"
    first_angle_bin: str = "unknown"
    reflection_bin: str = "unknown"
    diffraction_bin: str = "unknown"


CONTRAST_KEY_FIELDS = (
    "los_status",
    "path_richness",
    "ds_bin",
    "as_az_bin",
    "k_factor_bin",
    "first_power_bin",
    "first_angle_bin",
)

SEMANTIC_FIELD_KEY_FIELDS = {
    "all": CONTRAST_KEY_FIELDS,
    "los": ("los_status",),
    "n_paths": ("path_richness",),
    "delay_spread": ("ds_bin",),
    "angle_spread": ("as_az_bin",),
    "k_factor": ("k_factor_bin",),
    "first_delay": ("first_delay_bin",),
    "first_power": ("first_power_bin",),
    "first_angle": ("first_angle_bin",),
    "reflection": ("reflection_bin",),
    "diffraction": ("diffraction_bin",),
    "interaction": ("reflection_bin", "diffraction_bin"),
}

SEMANTIC_KEY_FIELDS = tuple(SemanticKey.__dataclass_fields__)
DERIVED_SEMANTIC_KEY_FIELDS = ("k_factor_binary", "first_power_binary")

DEFAULT_ATTRIBUTE_FIELDS = (
    "los_status",
    "path_richness",
    "k_factor_bin",
    "first_delay_bin",
    "first_power_bin",
    "first_angle_bin",
)


def semantic_key_field_choices() -> tuple[str, ...]:
    return (*SEMANTIC_KEY_FIELDS, *DERIVED_SEMANTIC_KEY_FIELDS)


AttributeRemap = Mapping[str, Mapping[str, tuple[str, ...] | list[str]]]


def normalize_path_richness(value: str) -> str:
    value = str(value)
    if value in {"zero", "one", "low"}:
        return "low"
    if value in {"two", "three", "four", "moderate"}:
        return "moderate"
    return "high"


def normalize_k_factor_bin(value: str) -> str:
    return "weak" if str(value) == "weak" else "strong"


def semantic_key_attribute_raw_value(key: SemanticKey, field: str) -> str:
    if field == "path_richness":
        return normalize_path_richness(key.path_richness)
    if field == "k_factor_bin":
        return normalize_k_factor_bin(key.k_factor_bin)
    if field == "k_factor_binary":
        return normalize_k_factor_bin(key.k_factor_bin)
    if field == "first_power_binary":
        if key.first_power_bin in {"very_weak", "weak"}:
            return "weak"
        return key.first_power_bin if key.first_power_bin == "strong" else "moderate"
    return str(getattr(key, field))


def semantic_key_attribute_value(
    key: SemanticKey,
    field: str,
    attribute_remap: AttributeRemap | None = None,
) -> str:
    raw_value = semantic_key_attribute_raw_value(key, field)
    field_remap = (attribute_remap or {}).get(field, {})
    for mapped_value, source_values in field_remap.items():
        if raw_value in set(source_values):
            return str(mapped_value)
    return raw_value


def implied_attribute_value_filters(
    fields: tuple[str, ...],
    attribute_remap: AttributeRemap | None = None,
) -> dict[str, tuple[str, ...]]:
    filters: dict[str, tuple[str, ...]] = {}
    if "k_factor_binary" in fields:
        filters["k_factor_binary"] = ("weak", "strong")
    if "first_power_binary" in fields:
        filters["first_power_binary"] = ("weak", "strong")
    for field in fields:
        field_remap = (attribute_remap or {}).get(field)
        if field_remap:
            filters[field] = tuple(
                dict.fromkeys(
                    str(source_value)
                    for source_values in field_remap.values()
                    for source_value in source_values
                )
            )
    return filters


def default_attribute_fields() -> tuple[str, ...]:
    return DEFAULT_ATTRIBUTE_FIELDS


def semantic_field_choices() -> tuple[str, ...]:
    return tuple(SEMANTIC_FIELD_KEY_FIELDS)


def contrast_key(key: object, semantic_field: str = "all") -> tuple:
    if semantic_field not in SEMANTIC_FIELD_KEY_FIELDS:
        raise ValueError(
            f"Unknown semantic_field={semantic_field!r}. "
            f"Choose from: {', '.join(semantic_field_choices())}"
        )
    fields = SEMANTIC_FIELD_KEY_FIELDS[semantic_field]
    return tuple(getattr(key, field, None) for field in fields)


def discretize(value: float, bins_dict: Mapping[str, tuple[float, float]]) -> str:
    if not np.isfinite(value):
        return "unknown" if "unknown" in bins_dict else next(reversed(bins_dict))
    for label, (lo, hi) in bins_dict.items():
        if label == "unknown":
            continue
        if lo <= value < hi:
            return label
    return "unknown" if "unknown" in bins_dict else next(reversed(bins_dict))


def discretize_angle_sector(angle_deg: float) -> str:
    if not np.isfinite(angle_deg):
        return "unknown"
    wrapped = ((angle_deg + 180.0) % 360.0) - 180.0
    if -45.0 <= wrapped < 45.0:
        return "front"
    if 45.0 <= wrapped < 135.0:
        return "left"
    if -135.0 <= wrapped < -45.0:
        return "right"
    return "back"


def _get(sample: Mapping[str, float] | object, key: str, default=None):
    if isinstance(sample, Mapping):
        return sample.get(key, default)
    return getattr(sample, key, default)


def build_semantic_key(sample: Mapping[str, float] | object) -> SemanticKey:
    return SemanticKey(
        env_type=_get(sample, "environment_type"),
        los_status="los" if bool(_get(sample, "los_status")) else "nlos",
        path_richness=normalize_path_richness(
            discretize(float(_get(sample, "n_paths")), PROP_DISC["n_paths"])
        ),
        ds_bin=discretize(
            float(_get(sample, "delay_spread")) * 1e9,
            PROP_DISC["delay_spread_ns"],
        ),
        as_az_bin=discretize(
            float(np.degrees(_get(sample, "azimuth_spread_aoa"))),
            PROP_DISC["azimuth_spread_deg"],
        ),
        k_factor_bin=normalize_k_factor_bin(
            discretize(
                float(_get(sample, "k_factor_db")),
                PROP_DISC["k_factor_db"],
            )
        ),
        first_delay_bin=discretize(
            float(_get(sample, "first_path_delay", float("nan"))) * 1e9,
            PROP_DISC["first_delay_ns"],
        ),
        first_power_bin=discretize(
            float(_get(sample, "first_path_power_dbw", float("nan"))),
            PROP_DISC["first_power_dbw"],
        ),
        first_angle_bin=discretize_angle_sector(
            float(_get(sample, "first_path_aoa_az_deg", float("nan"))),
        ),
        reflection_bin=discretize(
            float(_get(sample, "reflection_count", float("nan"))),
            PROP_DISC["interaction_count"],
        ),
        diffraction_bin=discretize(
            float(_get(sample, "diffraction_count", float("nan"))),
            PROP_DISC["interaction_count"],
        ),
    )
