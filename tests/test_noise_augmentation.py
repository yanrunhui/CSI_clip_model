from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.noise import add_complex_awgn_to_token_batch


def test_token_batch_awgn_hits_target_snr_and_preserves_padding() -> None:
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randn(2, 3, 8, 128, generator=generator)
    token_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ]
    )
    tokens = tokens * token_mask[:, :, None, None]

    noisy, actual_snr_db = add_complex_awgn_to_token_batch(
        tokens,
        token_mask,
        20.0,
        generator=torch.Generator().manual_seed(11),
    )

    assert noisy.shape == tokens.shape
    assert torch.allclose(actual_snr_db, torch.full((2,), 20.0), atol=0.5)
    assert torch.equal(noisy[0, 2], torch.zeros_like(noisy[0, 2]))
    assert torch.equal(noisy[1, 1:], torch.zeros_like(noisy[1, 1:]))


def test_token_batch_awgn_supports_per_sample_snr() -> None:
    tokens = torch.ones(2, 1, 8, 256)
    token_mask = torch.ones(2, 1, dtype=torch.bool)

    _, actual_snr_db = add_complex_awgn_to_token_batch(
        tokens,
        token_mask,
        torch.tensor([10.0, 30.0]),
        generator=torch.Generator().manual_seed(13),
    )

    assert torch.allclose(
        actual_snr_db,
        torch.tensor([10.0, 30.0]),
        atol=0.5,
    )
