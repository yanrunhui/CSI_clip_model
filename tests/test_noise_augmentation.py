from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.noise import add_complex_awgn_to_token_batch
from training.trainer import Trainer


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


def test_noisy_physics_forward_passes_both_delay_contexts() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.delay_context = torch.tensor([[1.0, 2.0]])
            self.first_path_delay_context = torch.tensor([[3.0, 4.0]])
            self.received: dict[str, torch.Tensor] = {}

        def encode_csi(self, tokens: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return torch.tensor([[5.0, 6.0]])

        def encode_csi_delay_context(self, *args, **kwargs) -> torch.Tensor:
            return self.delay_context

        def encode_first_path_delay_context(self, *args, **kwargs) -> torch.Tensor:
            return self.first_path_delay_context

        def predict_physics_components(
            self,
            features: torch.Tensor,
            **kwargs,
        ) -> dict[str, torch.Tensor]:
            self.received = kwargs
            return {"final": features}

    model = RecordingModel()
    trainer = object.__new__(Trainer)
    trainer.model = model
    batch = {
        "beam_positions": torch.zeros(1, 1, 3),
        "token_mask": torch.ones(1, 1, dtype=torch.bool),
        "freq_bin": torch.zeros(1, dtype=torch.long),
        "bw_bin": torch.zeros(1, dtype=torch.long),
        "subcarrier_spacing": torch.ones(1),
        "config_features": torch.zeros(1, 4),
        "antenna_coordinates": torch.zeros(1, 1, 3),
        "antenna_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    Trainer._noise_augmented_physics_outputs(
        trainer,
        batch,
        torch.zeros(1, 1, 2, 256),
    )

    assert model.received["delay_context"] is model.delay_context
    assert (
        model.received["first_path_delay_context"]
        is model.first_path_delay_context
    )
