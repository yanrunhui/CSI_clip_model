from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


GROUP_ID_RE = re.compile(
    r"^(?P<dataset>.+)-map_(?P<map>\d+)-source_(?P<source>\d+)-rx_(?P<rx>\d+)$"
)

NUMERIC_FIELDS = (
    "n_paths",
    "delay_spread_s",
    "azimuth_spread_deg",
    "k_factor_db",
    "first_path_delay_s",
    "los_delay_s",
    "los_aoa_az_deg",
    "first_path_power_dbw",
    "first_path_aoa_az_deg",
    "reflection_count",
    "diffraction_count",
    "reflection_path_count",
    "diffraction_path_count",
    "direct_path_count",
)

INTEGER_FIELDS = {
    "n_paths",
    "reflection_count",
    "diffraction_count",
    "reflection_path_count",
    "diffraction_path_count",
    "direct_path_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a serialized PreprocessedSample dataset without modifying it."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--compare",
        type=Path,
        action="append",
        default=[],
        help="Optional dataset to compare for map/link/observation overlap.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-text", type=Path)
    parser.add_argument("--show-examples", type=int, default=3)
    parser.add_argument(
        "--scan-tensors",
        action="store_true",
        help="Scan every token tensor for NaN/Inf values; this can take time.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_samples(path: Path) -> list[Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty list in {path}, got {type(payload).__name__}.")
    return payload


def instance_has_field(sample: Any, field: str) -> bool:
    values = getattr(sample, "__dict__", None)
    return isinstance(values, dict) and field in values


def finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def normalized_float_key(value: Any, digits: int) -> float | str:
    converted = finite_float(value)
    return "missing" if converted is None else round(converted, digits)


def normalized_link_key(group_id: str) -> tuple[int, int, int] | None:
    match = GROUP_ID_RE.fullmatch(group_id)
    if match is None:
        return None
    return (
        int(match.group("map")),
        int(match.group("source")),
        int(match.group("rx")),
    )


def sample_status(sample: Any) -> str:
    return str(getattr(getattr(sample, "semantic_key", None), "los_status", "unknown"))


def sample_environment(sample: Any) -> str:
    return str(getattr(getattr(sample, "semantic_key", None), "env_type", "unknown"))


def observation_key(sample: Any) -> tuple[Any, ...] | None:
    group_id = str(getattr(sample, "group_id", "")).strip()
    link = normalized_link_key(group_id)
    if link is None:
        return None
    return (
        *link,
        str(getattr(sample, "config_key", "")),
        int(getattr(sample, "source_n_freq", 0) or 0),
        normalized_float_key(getattr(sample, "bandwidth_hz", math.nan), 3),
        normalized_float_key(getattr(sample, "subcarrier_spacing_hz", math.nan), 6),
    )


def quantile_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "count": int(array.size),
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "max": float(quantiles[4]),
        "mean": float(array.mean()),
    }


def sorted_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def numeric_field_summary(samples: list[Any], field: str) -> dict[str, Any]:
    explicit_count = sum(instance_has_field(sample, field) for sample in samples)
    finite_values: list[float] = []
    for sample in samples:
        if not instance_has_field(sample, field):
            continue
        value = finite_float(sample.__dict__[field])
        if value is not None:
            finite_values.append(value)

    summary: dict[str, Any] = {
        "serialized_field_count": explicit_count,
        "serialized_field_rate": explicit_count / len(samples),
        "finite_serialized_value_count": len(finite_values),
        "nonfinite_or_missing_count": len(samples) - len(finite_values),
        "all_finite_values_zero": (
            all(value == 0 for value in finite_values) if finite_values else None
        ),
        "distribution": quantile_summary(finite_values),
    }
    if field in INTEGER_FIELDS:
        histogram = Counter(int(round(value)) for value in finite_values)
        summary["histogram"] = sorted_counter(histogram)
    return summary


def inspect_samples(samples: list[Any], *, scan_tensors: bool, show_examples: int) -> dict[str, Any]:
    group_ids = [str(getattr(sample, "group_id", "")).strip() for sample in samples]
    group_counter = Counter(group_ids)
    parsed_links = [normalized_link_key(group_id) for group_id in group_ids]
    valid_links = [link for link in parsed_links if link is not None]
    observation_keys = [observation_key(sample) for sample in samples]
    valid_observations = [key for key in observation_keys if key is not None]

    token_shapes: Counter[str] = Counter()
    token_dtypes: Counter[str] = Counter()
    token_nonfinite_samples = 0
    token_nonfinite_values = 0
    for sample in samples:
        tokens = getattr(sample, "tokens", None)
        if not isinstance(tokens, torch.Tensor):
            token_shapes["missing"] += 1
            continue
        token_shapes[str(tuple(tokens.shape))] += 1
        token_dtypes[str(tokens.dtype)] += 1
        if scan_tensors:
            nonfinite = int((~torch.isfinite(tokens)).sum().item())
            token_nonfinite_values += nonfinite
            token_nonfinite_samples += int(nonfinite > 0)

    status_histogram = Counter(sample_status(sample) for sample in samples)
    environment_histogram = Counter(sample_environment(sample) for sample in samples)
    config_histogram = Counter(str(getattr(sample, "config_key", "")) for sample in samples)
    n_tokens_histogram = Counter(int(getattr(sample, "n_tokens", 0) or 0) for sample in samples)
    source_nf_histogram = Counter(int(getattr(sample, "source_n_freq", 0) or 0) for sample in samples)
    scs_histogram = Counter(
        normalized_float_key(getattr(sample, "subcarrier_spacing_hz", math.nan), 6)
        for sample in samples
    )
    bandwidth_histogram = Counter(
        normalized_float_key(getattr(sample, "bandwidth_hz", math.nan), 3)
        for sample in samples
    )
    geometry_histogram = Counter(
        (
            str(getattr(sample, "array_type", "")),
            int(getattr(sample, "array_rows", 0) or 0),
            int(getattr(sample, "array_cols", 0) or 0),
        )
        for sample in samples
    )

    los_samples = [sample for sample in samples if sample_status(sample).lower() == "los"]
    nlos_samples = [sample for sample in samples if sample_status(sample).lower() == "nlos"]

    def finite_explicit(sample: Any, field: str) -> float | None:
        if not instance_has_field(sample, field):
            return None
        return finite_float(sample.__dict__[field])

    los_delay_values = [finite_explicit(sample, "los_delay_s") for sample in los_samples]
    nlos_delay_values = [finite_explicit(sample, "los_delay_s") for sample in nlos_samples]
    los_angle_values = [finite_explicit(sample, "los_aoa_az_deg") for sample in los_samples]
    nlos_angle_values = [finite_explicit(sample, "los_aoa_az_deg") for sample in nlos_samples]

    equal_delay_count = 0
    comparable_delay_count = 0
    for sample in los_samples:
        first = finite_explicit(sample, "first_path_delay_s")
        direct = finite_explicit(sample, "los_delay_s")
        if first is None or direct is None:
            continue
        comparable_delay_count += 1
        equal_delay_count += int(math.isclose(first, direct, rel_tol=0.0, abs_tol=1e-12))

    constraint_counts = Counter()
    for sample in samples:
        values = {
            field: finite_explicit(sample, field)
            for field in (
                "n_paths",
                "delay_spread_s",
                "azimuth_spread_deg",
                "reflection_count",
                "reflection_path_count",
                "direct_path_count",
            )
        }
        if values["n_paths"] is not None and values["n_paths"] < 1:
            constraint_counts["n_paths_lt_1"] += 1
        if values["delay_spread_s"] is not None and values["delay_spread_s"] < 0:
            constraint_counts["delay_spread_negative"] += 1
        if values["azimuth_spread_deg"] is not None and values["azimuth_spread_deg"] < 0:
            constraint_counts["azimuth_spread_negative"] += 1
        if values["reflection_count"] is not None and values["reflection_count"] < 0:
            constraint_counts["reflection_count_negative"] += 1
        if values["reflection_path_count"] is not None and values["reflection_path_count"] < 0:
            constraint_counts["reflection_path_count_negative"] += 1
        if (
            values["reflection_path_count"] is not None
            and values["n_paths"] is not None
            and values["reflection_path_count"] > values["n_paths"]
        ):
            constraint_counts["reflection_path_count_gt_n_paths"] += 1

    examples = []
    for index, sample in enumerate(samples[: max(show_examples, 0)]):
        examples.append(
            {
                "index": index,
                "group_id": str(getattr(sample, "group_id", "")),
                "parsed_link_key": normalized_link_key(str(getattr(sample, "group_id", ""))),
                "config_key": str(getattr(sample, "config_key", "")),
                "los_status": sample_status(sample),
                "token_shape": list(getattr(sample, "tokens", torch.empty(0)).shape),
                "n_paths": finite_explicit(sample, "n_paths"),
                "first_path_delay_ns": (
                    None
                    if finite_explicit(sample, "first_path_delay_s") is None
                    else finite_explicit(sample, "first_path_delay_s") * 1e9
                ),
                "reflection_count": finite_explicit(sample, "reflection_count"),
                "reflection_path_count": finite_explicit(sample, "reflection_path_count"),
            }
        )

    reflection_path = numeric_field_summary(samples, "reflection_path_count")
    reflection_path_usable = (
        reflection_path["serialized_field_rate"] == 1.0
        and reflection_path["finite_serialized_value_count"] == len(samples)
        and not reflection_path["all_finite_values_zero"]
    )

    return {
        "sample_count": len(samples),
        "sample_class_histogram": sorted_counter(Counter(type(sample).__name__ for sample in samples)),
        "environment_histogram": sorted_counter(environment_histogram),
        "los_status_histogram": sorted_counter(status_histogram),
        "configuration": {
            "config_key_histogram": sorted_counter(config_histogram),
            "geometry_histogram": sorted_counter(geometry_histogram),
            "source_n_freq_histogram": sorted_counter(source_nf_histogram),
            "subcarrier_spacing_hz_histogram": sorted_counter(scs_histogram),
            "bandwidth_hz_histogram": sorted_counter(bandwidth_histogram),
            "n_tokens_histogram": sorted_counter(n_tokens_histogram),
            "token_shape_histogram": sorted_counter(token_shapes),
            "token_dtype_histogram": sorted_counter(token_dtypes),
        },
        "identity": {
            "missing_group_id_count": sum(not group_id for group_id in group_ids),
            "invalid_group_id_format_count": sum(link is None for link in parsed_links),
            "unique_group_id_count": len(group_counter),
            "duplicate_group_id_value_count": sum(count > 1 for count in group_counter.values()),
            "duplicate_group_id_sample_count": sum(max(count - 1, 0) for count in group_counter.values()),
            "unique_normalized_link_count": len(set(valid_links)),
            "unique_observation_count": len(set(valid_observations)),
            "duplicate_observation_sample_count": len(valid_observations) - len(set(valid_observations)),
            "map_count": len({link[0] for link in valid_links}),
            "map_ids": sorted({link[0] for link in valid_links}),
            "source_count": len({(link[0], link[1]) for link in valid_links}),
        },
        "numeric_fields": {
            field: numeric_field_summary(samples, field) for field in NUMERIC_FIELDS
        },
        "conditional_label_checks": {
            "los_sample_count": len(los_samples),
            "nlos_sample_count": len(nlos_samples),
            "los_delay_finite_on_los_count": sum(value is not None for value in los_delay_values),
            "los_delay_finite_on_nlos_count": sum(value is not None for value in nlos_delay_values),
            "los_angle_finite_on_los_count": sum(value is not None for value in los_angle_values),
            "los_angle_finite_on_nlos_count": sum(value is not None for value in nlos_angle_values),
            "first_los_delay_comparable_count": comparable_delay_count,
            "first_los_delay_equal_within_1e-3ns_count": equal_delay_count,
            "first_los_delay_equal_within_1e-3ns_rate": (
                equal_delay_count / comparable_delay_count if comparable_delay_count else None
            ),
        },
        "constraint_violation_counts": sorted_counter(constraint_counts),
        "reflection_path_label_audit": {
            "usable": reflection_path_usable,
            "reason": (
                "serialized, finite, and nonconstant-zero"
                if reflection_path_usable
                else "missing/nonfinite in some records or constant zero; inspect provenance before use"
            ),
        },
        "tensor_scan": {
            "enabled": scan_tensors,
            "nonfinite_sample_count": token_nonfinite_samples if scan_tensors else None,
            "nonfinite_value_count": token_nonfinite_values if scan_tensors else None,
        },
        "examples": examples,
    }


def identity_sets(samples: Iterable[Any]) -> dict[str, set[Any]]:
    group_ids = {
        str(getattr(sample, "group_id", "")).strip()
        for sample in samples
        if str(getattr(sample, "group_id", "")).strip()
    }
    links = {
        link
        for group_id in group_ids
        if (link := normalized_link_key(group_id)) is not None
    }
    observations = {
        key for sample in samples if (key := observation_key(sample)) is not None
    }
    return {
        "group_ids": group_ids,
        "links": links,
        "observations": observations,
        "maps": {link[0] for link in links},
    }


def compare_datasets(reference: list[Any], candidate: list[Any]) -> dict[str, int]:
    left = identity_sets(reference)
    right = identity_sets(candidate)
    return {
        "exact_group_id_overlap_count": len(left["group_ids"] & right["group_ids"]),
        "normalized_physical_link_overlap_count": len(left["links"] & right["links"]),
        "exact_observation_overlap_count": len(left["observations"] & right["observations"]),
        "map_overlap_count": len(left["maps"] & right["maps"]),
    }


def format_report(payload: dict[str, Any]) -> str:
    audit = payload["audit"]
    identity = audit["identity"]
    config = audit["configuration"]
    conditional = audit["conditional_label_checks"]
    reflection = audit["numeric_fields"]["reflection_path_count"]
    lines = [
        f"dataset={payload['dataset']}",
        f"file_size_bytes={payload['file_size_bytes']}",
        f"sha256={payload['sha256']}",
        f"sample_count={audit['sample_count']}",
        f"los_status_histogram={json.dumps(audit['los_status_histogram'], sort_keys=True)}",
        f"environment_histogram={json.dumps(audit['environment_histogram'], sort_keys=True)}",
        f"config_key_histogram={json.dumps(config['config_key_histogram'], sort_keys=True)}",
        f"token_shape_histogram={json.dumps(config['token_shape_histogram'], sort_keys=True)}",
        f"source_n_freq_histogram={json.dumps(config['source_n_freq_histogram'], sort_keys=True)}",
        f"subcarrier_spacing_hz_histogram={json.dumps(config['subcarrier_spacing_hz_histogram'], sort_keys=True)}",
        f"bandwidth_hz_histogram={json.dumps(config['bandwidth_hz_histogram'], sort_keys=True)}",
        f"map_count={identity['map_count']}",
        f"source_count={identity['source_count']}",
        f"unique_group_id_count={identity['unique_group_id_count']}",
        f"duplicate_group_id_sample_count={identity['duplicate_group_id_sample_count']}",
        f"duplicate_observation_sample_count={identity['duplicate_observation_sample_count']}",
        f"invalid_group_id_format_count={identity['invalid_group_id_format_count']}",
        f"reflection_path_count_serialized_field_rate={reflection['serialized_field_rate']:.6f}",
        f"reflection_path_count_all_zero={reflection['all_finite_values_zero']}",
        f"reflection_path_count_histogram={json.dumps(reflection.get('histogram', {}), sort_keys=True)}",
        f"reflection_path_count_usable={audit['reflection_path_label_audit']['usable']}",
        f"los_delay_finite_on_los={conditional['los_delay_finite_on_los_count']}/{conditional['los_sample_count']}",
        f"los_delay_finite_on_nlos={conditional['los_delay_finite_on_nlos_count']}/{conditional['nlos_sample_count']}",
        f"first_los_delay_equal_rate={conditional['first_los_delay_equal_within_1e-3ns_rate']}",
        "constraint_violation_counts="
        + json.dumps(audit["constraint_violation_counts"], sort_keys=True),
    ]
    for comparison in payload["comparisons"]:
        lines.extend(
            [
                "",
                f"compare_dataset={comparison['dataset']}",
                f"compare_sha256={comparison['sha256']}",
                "overlap=" + json.dumps(comparison["overlap"], sort_keys=True),
            ]
        )
    lines.extend(["", "examples=" + json.dumps(audit["examples"], ensure_ascii=True)])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    samples = load_samples(dataset_path)
    payload: dict[str, Any] = {
        "dataset": str(dataset_path),
        "file_size_bytes": dataset_path.stat().st_size,
        "sha256": sha256_file(dataset_path),
        "audit": inspect_samples(
            samples,
            scan_tensors=args.scan_tensors,
            show_examples=args.show_examples,
        ),
        "comparisons": [],
    }

    for compare_path_arg in args.compare:
        compare_path = compare_path_arg.resolve()
        compare_samples = load_samples(compare_path)
        payload["comparisons"].append(
            {
                "dataset": str(compare_path),
                "file_size_bytes": compare_path.stat().st_size,
                "sha256": sha256_file(compare_path),
                "sample_count": len(compare_samples),
                "overlap": compare_datasets(samples, compare_samples),
            }
        )

    report = format_report(payload)
    print(report, end="")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved_json={args.output_json}")
    if args.output_text is not None:
        args.output_text.parent.mkdir(parents=True, exist_ok=True)
        args.output_text.write_text(report, encoding="utf-8")
        print(f"saved_text={args.output_text}")


if __name__ == "__main__":
    main()
