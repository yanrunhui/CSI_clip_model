from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess_all import load_d2los_dataset, preprocess_deepmimo_dataset


DEFAULT_D2LOS_ROOT = Path(
    "/home/yrh/CSI_model/CSI_model/deepmimo_scenarios/D2Los_Data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate native SISO DeepMIMO/RayVerse samples with one spatial token. "
            "The saved token layout is [1, 2, Nf]: real and imaginary parts of one "
            "physical 1x1 channel, with no virtual-array zero padding."
        )
    )
    parser.add_argument("--d2los-root", type=Path, default=DEFAULT_D2LOS_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        required=True,
        help="Required safety limit because a complete D2Los conversion is very large.",
    )
    parser.add_argument("--max-maps", type=int)
    parser.add_argument("--max-sources-per-map", type=int)
    parser.add_argument("--max-rx-per-source", type=int)
    parser.add_argument(
        "--sampling",
        choices=("sequential", "uniform", "map_uniform"),
        default="map_uniform",
    )
    parser.add_argument("--sample-seed", type=int, default=23421)
    parser.add_argument("--bandwidth-hz", type=float, default=100e6)
    parser.add_argument("--subcarriers", type=int, default=128)
    parser.add_argument(
        "--target-nf",
        type=int,
        default=128,
        help="Frequency points stored in every model token.",
    )
    parser.add_argument("--tx-power-dbm", type=float, default=23.0)
    parser.add_argument("--antenna-spacing", type=float, default=0.5)
    parser.add_argument("--freq-bin", type=int, default=0)
    parser.add_argument("--env-type", choices=("indoor", "outdoor", "O2I"), default="outdoor")
    parser.add_argument("--include-empty-samples", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.bandwidth_hz <= 0.0:
        raise ValueError("--bandwidth-hz must be positive")
    if args.subcarriers <= 0:
        raise ValueError("--subcarriers must be positive")
    if args.target_nf <= 0:
        raise ValueError("--target-nf must be positive")
    if args.subcarriers != args.target_nf:
        raise ValueError(
            "For measured-data matching, --subcarriers and --target-nf must be equal "
            f"(got {args.subcarriers} and {args.target_nf})."
        )
    if args.max_maps is not None and args.max_maps <= 0:
        raise ValueError("--max-maps must be positive")
    if args.max_sources_per_map is not None and args.max_sources_per_map <= 0:
        raise ValueError("--max-sources-per-map must be positive")
    if args.max_rx_per_source is not None and args.max_rx_per_source <= 0:
        raise ValueError("--max-rx-per-source must be positive")


def validate_siso_samples(samples: list, target_nf: int, bandwidth_hz: float) -> None:
    expected_shape = (1, 2, target_nf)
    expected_spacing_hz = bandwidth_hz / target_nf
    for index, sample in enumerate(samples):
        if tuple(sample.tokens.shape) != expected_shape:
            raise ValueError(
                f"sample {index} has tokens {tuple(sample.tokens.shape)}; "
                f"expected native SISO {expected_shape}"
            )
        if sample.n_tokens != 1:
            raise ValueError(f"sample {index} has n_tokens={sample.n_tokens}; expected 1")
        if sample.array_rows != 1 or sample.array_cols != 1:
            raise ValueError(
                f"sample {index} has array {sample.array_rows}x{sample.array_cols}; expected 1x1"
            )
        if sample.source_n_freq != target_nf:
            raise ValueError(
                f"sample {index} has source_n_freq={sample.source_n_freq}; expected {target_nf}"
            )
        if abs(float(sample.bandwidth_hz) - bandwidth_hz) > max(1e-6 * bandwidth_hz, 1.0):
            raise ValueError(
                f"sample {index} has bandwidth_hz={sample.bandwidth_hz}; expected {bandwidth_hz}"
            )
        if abs(float(sample.subcarrier_spacing_hz) - expected_spacing_hz) > max(
            1e-6 * expected_spacing_hz, 1e-3
        ):
            raise ValueError(
                f"sample {index} has subcarrier_spacing_hz={sample.subcarrier_spacing_hz}; "
                f"expected {expected_spacing_hz}"
            )


def atomic_torch_save(samples: list, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    torch.save(samples, temporary)
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    validate_args(args)

    dataset = load_d2los_dataset(
        d2los_root=args.d2los_root,
        max_samples=args.max_samples,
        max_maps=args.max_maps,
        max_sources_per_map=args.max_sources_per_map,
        max_rx_per_source=args.max_rx_per_source,
        tx_shape=(1, 1),
        bandwidth_hz=args.bandwidth_hz,
        total_subcarriers=args.subcarriers,
        tx_power_dbm=args.tx_power_dbm,
        tx_spacing=args.antenna_spacing,
        sampling=args.sampling,
        sample_seed=args.sample_seed,
    )
    samples = preprocess_deepmimo_dataset(
        dataset=dataset,
        scenario=args.d2los_root.name,
        freq_bin=args.freq_bin,
        rx_index=0,
        env_type=args.env_type,
        max_samples=args.max_samples,
        patch_1d=1,
        patch_2d=(1, 1),
        target_nf=args.target_nf,
        include_empty_samples=args.include_empty_samples,
    )
    if not samples:
        raise ValueError("No usable SISO samples were produced")
    validate_siso_samples(
        samples,
        target_nf=args.target_nf,
        bandwidth_hz=args.bandwidth_hz,
    )
    atomic_torch_save(samples, args.output)

    first = samples[0]
    print(f"saved_samples={len(samples)}")
    print(f"output={args.output.resolve()}")
    print(f"tokens_per_sample={tuple(first.tokens.shape)}")
    print(f"array={first.array_rows}x{first.array_cols}")
    print(f"bandwidth_hz={first.bandwidth_hz}")
    print(f"source_n_freq={first.source_n_freq}")
    print(f"target_n_freq={first.tokens.shape[-1]}")
    print(f"subcarrier_spacing_hz={first.subcarrier_spacing_hz}")
    print("zero_padded_virtual_antennas=0")


if __name__ == "__main__":
    main()
