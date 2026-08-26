from __future__ import annotations

from dataclasses import replace
import math

import torch

from .dataset import PreprocessedSample


def add_complex_awgn(
    h: torch.Tensor,
    snr_db: float,
    *,
    generator: torch.Generator | None = None,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Add sample-power-scaled circular complex AWGN to one CSI tensor.

    ``valid_mask`` marks physical complex coefficients. It is useful for tokenized
    CSI whose final beam patch contains zero padding. Signal and noise power are
    both measured only over valid coefficients.
    """
    if not h.is_complex():
        raise TypeError("h must be a complex tensor")
    if not math.isfinite(float(snr_db)):
        raise ValueError("snr_db must be finite")

    if valid_mask is None:
        mask = torch.ones(h.shape, dtype=torch.bool, device=h.device)
    else:
        mask = valid_mask.to(device=h.device, dtype=torch.bool)
        try:
            mask = torch.broadcast_to(mask, h.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} is not broadcastable "
                f"to CSI shape {tuple(h.shape)}"
            ) from exc
    if not bool(mask.any()):
        raise ValueError("valid_mask contains no physical CSI coefficients")

    valid_h = h[mask]
    signal_power = valid_h.abs().square().mean()
    if not bool(torch.isfinite(signal_power)) or float(signal_power) <= 0.0:
        raise ValueError("CSI signal power must be finite and positive")

    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = signal_power / snr_linear
    real_dtype = h.real.dtype
    noise_real = torch.randn(
        h.shape,
        dtype=real_dtype,
        device=h.device,
        generator=generator,
    )
    noise_imag = torch.randn(
        h.shape,
        dtype=real_dtype,
        device=h.device,
        generator=generator,
    )
    noise = torch.complex(noise_real, noise_imag)
    noise = noise * torch.sqrt(noise_power / 2.0)
    noise = torch.where(mask, noise, torch.zeros_like(noise))

    realized_noise_power = noise[mask].abs().square().mean()
    actual_snr_db = 10.0 * torch.log10(signal_power / realized_noise_power)
    return h + noise, float(actual_snr_db)


def _valid_patch_mask(
    sample: PreprocessedSample,
    *,
    patch_1d: int,
    patch_2d: tuple[int, int],
) -> torch.Tensor:
    """Return [n_tokens, complex_channels] mask for non-padding beam entries."""
    tokens = sample.tokens
    if tokens.ndim != 3 or tokens.shape[1] % 2 != 0:
        raise ValueError(
            "sample.tokens must have shape [tokens, 2*complex_channels, frequency], "
            f"got {tuple(tokens.shape)}"
        )
    n_tokens, doubled_channels, _ = tokens.shape
    complex_channels = doubled_channels // 2
    rows = int(sample.array_rows)
    cols = int(sample.array_cols)
    if rows <= 0 or cols <= 0:
        raise ValueError(
            f"Invalid array geometry rows={rows}, cols={cols} for {sample.group_id!r}"
        )

    mask = torch.zeros(n_tokens, complex_channels, dtype=torch.bool)
    if sample.array_type == "ULA":
        if patch_1d != complex_channels:
            raise ValueError(
                f"patch_1d={patch_1d} does not match token complex width "
                f"{complex_channels}"
            )
        antenna_count = rows * cols
        expected_tokens = math.ceil(antenna_count / patch_1d)
        if n_tokens != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} ULA tokens for {antenna_count} antennas, "
                f"found {n_tokens}"
            )
        for token_idx in range(n_tokens):
            valid_count = min(
                patch_1d,
                max(antenna_count - token_idx * patch_1d, 0),
            )
            mask[token_idx, :valid_count] = True
        return mask

    if sample.array_type != "UPA":
        raise ValueError(f"Unsupported array_type={sample.array_type!r}")
    patch_rows, patch_cols = patch_2d
    if patch_rows <= 0 or patch_cols <= 0:
        raise ValueError("patch_2d dimensions must be positive")
    if patch_rows * patch_cols != complex_channels:
        raise ValueError(
            f"patch_2d={patch_2d} has {patch_rows * patch_cols} entries but "
            f"tokens contain {complex_channels} complex channels"
        )
    expected_tokens = math.ceil(rows / patch_rows) * math.ceil(cols / patch_cols)
    if n_tokens != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} UPA tokens for {rows}x{cols} geometry and "
            f"patch {patch_2d}, found {n_tokens}"
        )

    token_idx = 0
    for row_start in range(0, rows, patch_rows):
        for col_start in range(0, cols, patch_cols):
            valid_rows = min(patch_rows, rows - row_start)
            valid_cols = min(patch_cols, cols - col_start)
            patch_mask = torch.zeros(patch_rows, patch_cols, dtype=torch.bool)
            patch_mask[:valid_rows, :valid_cols] = True
            mask[token_idx] = patch_mask.reshape(-1)
            token_idx += 1
    return mask


def add_awgn_to_preprocessed_sample(
    sample: PreprocessedSample,
    snr_db: float,
    *,
    generator: torch.Generator,
    patch_1d: int = 4,
    patch_2d: tuple[int, int] = (2, 2),
) -> tuple[PreprocessedSample, float]:
    """Add AWGN to the unnormalized complex beamspace CSI stored in a sample.

    The serialized token layout stores all real beam coefficients followed by
    all imaginary coefficients within each patch. Padding coefficients remain
    exactly zero and are excluded from both power estimation and noise.
    """
    tokens = sample.tokens
    complex_channels = tokens.shape[1] // 2
    h = torch.complex(
        tokens[:, :complex_channels],
        tokens[:, complex_channels:],
    )
    valid_patch = _valid_patch_mask(
        sample,
        patch_1d=patch_1d,
        patch_2d=patch_2d,
    )
    valid_mask = valid_patch.unsqueeze(-1).expand_as(h)
    noisy_h, actual_snr_db = add_complex_awgn(
        h,
        snr_db,
        generator=generator,
        valid_mask=valid_mask,
    )
    noisy_tokens = torch.cat([noisy_h.real, noisy_h.imag], dim=1).to(
        dtype=tokens.dtype
    )
    return replace(sample, tokens=noisy_tokens), actual_snr_db


def sample_noise_seed(noise_seed: int, sample_index: int) -> int:
    """Derive a stable independent PyTorch seed for one sample."""
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    modulus = 2**63 - 1
    return (int(noise_seed) * 1_000_003 + int(sample_index)) % modulus
