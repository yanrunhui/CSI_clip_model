from __future__ import annotations

import math
import random
from typing import Any

from .semantic_key import SemanticKey, semantic_field_choices

PROP_S1 = [
    "{env} environment with {los} propagation and {richness} multipath.",
    "{los} channel in an {env} setting with {richness} resolvable paths.",
    "{env}, {los}, {richness} multipath components.",
]

PROP_S2 = [
    "The delay spread is {ds} and the azimuth spread is {as_az}, with {kf} direct path.",
    "{ds} delay spread, {as_az} angular spread, {kf} dominant component.",
    "Temporal dispersion is {ds}, spatial dispersion is {as_az}, K-factor is {kf}.",
]

PROP_S3 = [
    "First path has {first_power} power and {first_angle} arrival.",
    "Earliest path: {first_power} power with {first_angle} arrival.",
    "Dominant arrival is {first_angle} with {first_power} first-path power.",
]

FOCUSED_PROP_TEMPLATES = {
    "los": [
        "The channel is {los}.",
        "Propagation condition: {los}.",
    ],
    "n_paths": [
        "The channel has {richness} multipath components.",
        "Multipath richness is {richness}.",
        "Resolvable path count is {richness}.",
    ],
    "delay_spread": [
        "The delay spread is {ds}.",
        "Temporal dispersion is {ds}.",
    ],
    "angle_spread": [
        "The azimuth angular spread is {as_az}.",
        "Arrival angle spread is {as_az}.",
    ],
    "k_factor": [
        "The direct path component is {kf}.",
        "K-factor strength is {kf}.",
    ],
    "first_delay": [
        "The first path delay is {first_delay}.",
        "Earliest path timing is {first_delay}.",
    ],
    "first_power": [
        "The first path power is {first_power}.",
        "Earliest path power is {first_power}.",
    ],
    "first_angle": [
        "The first path arrives from the {first_angle} sector.",
        "Earliest arrival direction is {first_angle}.",
    ],
    "reflection": [
        "The ray path has {reflection} reflections.",
        "Reflection count is {reflection}.",
    ],
    "diffraction": [
        "The ray path has {diffraction} diffractions.",
        "Diffraction count is {diffraction}.",
    ],
    "interaction": [
        "The ray path has {reflection} reflections and {diffraction} diffractions.",
        "Interaction count is {reflection} reflection and {diffraction} diffraction.",
    ],
}

FOCUSED_PROP_CANONICAL_TEMPLATES = {
    "los": "Propagation condition: {los}.",
    "n_paths": "Multipath richness: {richness}.",
    "delay_spread": "Delay spread: {ds}.",
    "angle_spread": "Azimuth spread: {as_az}.",
    "k_factor": "K-factor strength: {kf}.",
    "first_delay": "First path delay: {first_delay}.",
    "first_power": "First path power: {first_power}.",
    "first_angle": "First path arrival sector: {first_angle}.",
    "reflection": "Reflection count: {reflection}.",
    "diffraction": "Diffraction count: {diffraction}.",
    "interaction": "Interaction count: reflection {reflection}, diffraction {diffraction}.",
}

FOCUSED_PROP_CONTEXT = {
    "los": [
        "It shows {richness} multipath, {ds} delay spread, and a {kf} direct component.",
        "The channel exhibits {richness} path richness with {ds} temporal dispersion.",
    ],
    "n_paths": [
        "Propagation is {los} with {ds} delay spread and a {kf} direct path.",
        "The channel is {los}, with {ds} temporal dispersion and {as_az} angular spread.",
    ],
    "delay_spread": [
        "It is a {los} channel with {richness} multipath components and {as_az} angular spread.",
        "The channel has {richness} paths, {kf} direct-path strength, and {first_power} first-path power.",
    ],
    "angle_spread": [
        "The link is {los} with {richness} multipath components and {ds} delay spread.",
        "Temporal dispersion is {ds}, and the direct-path strength is {kf}.",
    ],
    "k_factor": [
        "The channel is {los} with {richness} multipath and {ds} delay spread.",
        "Angular spread is {as_az}, and the first path has {first_power} power.",
    ],
    "first_delay": [
        "The channel is {los} with {richness} multipath and {ds} delay spread.",
        "The first path has {first_power} power and arrives from the {first_angle} sector.",
    ],
    "first_power": [
        "The link is {los} with {richness} multipath and {ds} delay spread.",
        "The earliest arrival comes from the {first_angle} sector with {first_delay} timing.",
    ],
    "first_angle": [
        "The channel is {los} with {richness} multipath and {ds} delay spread.",
        "The earliest path has {first_power} power and {first_delay} delay.",
    ],
    "reflection": [
        "The channel is {los} with {richness} multipath and {ds} delay spread.",
        "The ray path has {diffraction} diffractions and {kf} direct-path strength.",
    ],
    "diffraction": [
        "The channel is {los} with {richness} multipath and {ds} delay spread.",
        "The ray path has {reflection} reflections and {kf} direct-path strength.",
    ],
    "interaction": [
        "The link is {los} with {richness} multipath and {ds} delay spread.",
        "The direct path is {kf}, and the first path arrives from the {first_angle} sector.",
    ],
}

SLOT_VOCAB = {
    "env": {
        "indoor": ["indoor", "indoor office"],
        "outdoor": ["outdoor", "outdoor urban"],
        "O2I": ["outdoor-to-indoor"],
    },
    "los": {
        "los": ["line-of-sight", "LoS"],
        "nlos": ["non-line-of-sight", "NLoS", "obstructed"],
    },
    "richness": {
        "low": ["low", "sparse", "few"],
        "high": ["high", "rich", "many"],
    },
    "ds": {
        "any": ["any"],
        "low": ["low", "small"],
        "moderate": ["moderate", "medium"],
        "high": ["high", "large"],
    },
    "as_az": {
        "any": ["any"],
        "narrow": ["narrow", "concentrated"],
        "moderate": ["moderate"],
        "wide": ["wide", "broad"],
    },
    "kf": {
        "any": ["any"],
        "strong": ["strong", "dominant"],
        "weak": ["weak", "absent"],
    },
    "first_delay": {
        "any": ["any"],
        "unknown": ["unknown"],
        "short": ["short"],
        "medium": ["medium"],
        "long": ["long"],
    },
    "first_power": {
        "unknown": ["unknown"],
        "weak": ["weak"],
        "moderate": ["moderate"],
        "strong": ["strong"],
    },
    "first_angle": {
        "any": ["any"],
        "unknown": ["unknown"],
        "front": ["front-side"],
        "left": ["left-side"],
        "back": ["back-side"],
        "right": ["right-side"],
    },
    "interaction": {
        "any": ["any"],
        "unknown": ["unknown"],
        "none": ["no"],
        "light": ["light"],
        "heavy": ["heavy"],
    },
}


class CaptionGenerator:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def _slots(self, key: SemanticKey) -> dict[str, str]:
        return {
            "env": self.rng.choice(SLOT_VOCAB["env"][key.env_type]),
            "los": self.rng.choice(SLOT_VOCAB["los"][key.los_status]),
            "richness": self.rng.choice(SLOT_VOCAB["richness"][key.path_richness]),
            "ds": self.rng.choice(SLOT_VOCAB["ds"][key.ds_bin]),
            "as_az": self.rng.choice(SLOT_VOCAB["as_az"][key.as_az_bin]),
            "kf": self.rng.choice(SLOT_VOCAB["kf"][key.k_factor_bin]),
            "first_delay": self.rng.choice(SLOT_VOCAB["first_delay"][key.first_delay_bin]),
            "first_power": self.rng.choice(SLOT_VOCAB["first_power"][key.first_power_bin]),
            "first_angle": self.rng.choice(SLOT_VOCAB["first_angle"][key.first_angle_bin]),
            "reflection": self.rng.choice(SLOT_VOCAB["interaction"][key.reflection_bin]),
            "diffraction": self.rng.choice(SLOT_VOCAB["interaction"][key.diffraction_bin]),
        }

    def _canonical_slots(self, key: SemanticKey) -> dict[str, str]:
        return {
            "env": key.env_type,
            "los": key.los_status,
            "richness": key.path_richness,
            "ds": key.ds_bin,
            "as_az": key.as_az_bin,
            "kf": key.k_factor_bin,
            "first_delay": key.first_delay_bin,
            "first_power": key.first_power_bin,
            "first_angle": key.first_angle_bin,
            "reflection": key.reflection_bin,
            "diffraction": key.diffraction_bin,
        }

    def generate_canonical(self, key: SemanticKey) -> str:
        slots = self._canonical_slots(key)
        return (
            "{env} environment with {los} propagation and {richness} multipath. "
            "Delay spread is {ds}, azimuth spread is {as_az}, and K-factor is {kf}. "
            "First path delay is {first_delay}, first path power is {first_power}, "
            "and first path arrival sector is {first_angle}. "
            "Reflection count is {reflection} and diffraction count is {diffraction}."
        ).format(**slots)

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _format_value(value: Any, unit: str = "", scale: float = 1.0, decimals: int = 1) -> str:
        finite = CaptionGenerator._finite_float(value)
        if finite is None:
            return "unknown"
        scaled = finite * scale
        formatted = f"{scaled:.{decimals}f}".rstrip("0").rstrip(".")
        return f"{formatted} {unit}".strip()

    @staticmethod
    def _format_count(value: Any) -> str:
        finite = CaptionGenerator._finite_float(value)
        if finite is None:
            return "unknown"
        return str(max(int(round(finite)), 0))

    def generate_instance(
        self,
        key: SemanticKey,
        *,
        n_paths: int,
        delay_spread_s: float,
        azimuth_spread_deg: float,
        k_factor_db: float,
        first_path_delay_s: float,
        first_path_power_dbw: float,
        first_path_aoa_az_deg: float,
        reflection_count: int,
        diffraction_count: int,
        config_key: str | None = None,
        subcarrier_spacing_hz: float | None = None,
    ) -> str:
        config_clause = f", {config_key}" if config_key else ""
        spacing_clause = ""
        if subcarrier_spacing_hz is not None:
            spacing_clause = f" SCS {self._format_value(subcarrier_spacing_hz, 'Hz', decimals=1)}."
        return (
            f"Instance channel: {key.env_type} {key.los_status}{config_clause}, "
            f"{self._format_count(n_paths)} paths. "
            f"Delay spread {self._format_value(delay_spread_s, 'ns', scale=1e9)}; "
            f"azimuth spread {self._format_value(azimuth_spread_deg, 'deg')}; "
            f"K-factor {self._format_value(k_factor_db, 'dB')}. "
            f"First path delay {self._format_value(first_path_delay_s, 'ns', scale=1e9)}, "
            f"power {self._format_value(first_path_power_dbw, 'dBW')}, "
            f"AoA {self._format_value(first_path_aoa_az_deg, 'deg')}. "
            f"Interactions: {self._format_count(reflection_count)} reflections, "
            f"{self._format_count(diffraction_count)} diffractions."
            f"{spacing_clause}"
        )

    def generate_instance_from_sample(self, sample: Any) -> str:
        return self.generate_instance(
            sample.semantic_key,
            n_paths=getattr(sample, "n_paths", 0),
            delay_spread_s=getattr(sample, "delay_spread_s", 0.0),
            azimuth_spread_deg=getattr(sample, "azimuth_spread_deg", 0.0),
            k_factor_db=getattr(sample, "k_factor_db", 0.0),
            first_path_delay_s=getattr(sample, "first_path_delay_s", math.nan),
            first_path_power_dbw=getattr(sample, "first_path_power_dbw", math.nan),
            first_path_aoa_az_deg=getattr(sample, "first_path_aoa_az_deg", math.nan),
            reflection_count=getattr(sample, "reflection_count", 0),
            diffraction_count=getattr(sample, "diffraction_count", 0),
            config_key=getattr(sample, "config_key", None),
            subcarrier_spacing_hz=getattr(sample, "subcarrier_spacing_hz", None),
        )

    def generate(self, key: SemanticKey, semantic_field: str = "all") -> str:
        if semantic_field not in semantic_field_choices():
            raise ValueError(
                f"Unknown semantic_field={semantic_field!r}. "
                f"Choose from: {', '.join(semantic_field_choices())}"
            )
        slots = {
            **self._slots(key),
        }
        if semantic_field != "all":
            # Focused semantic training works better with low-variance captions:
            # the text should name the target property directly instead of mixing
            # in randomly phrased side-context from other attributes.
            slots = self._canonical_slots(key)
            template = FOCUSED_PROP_CANONICAL_TEMPLATES[semantic_field]
            return template.format(**slots)
        s1 = self.rng.choice(PROP_S1).format(**slots)
        s2 = self.rng.choice(PROP_S2).format(**slots)
        s3 = self.rng.choice(PROP_S3).format(**slots)
        return f"{s1} {s2} {s3}"
