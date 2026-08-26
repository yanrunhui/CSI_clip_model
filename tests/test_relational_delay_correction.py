from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import _apply_signal_description_correction
from scripts.evaluate_signal_descriptions import consistency_violations
from data.dataset import PHYSICS_TARGET_NAMES, PHYSICS_TARGET_SCALES
from training.trainer import Trainer


def _record(first_delay: float, los_delay: float, status: str = "los") -> dict:
    return {
        "los_status": status,
        "first_path_delay_ns": first_delay,
        "los_delay_ns": los_delay,
        "delay_spread_ns": 1.0,
        "angle_spread_deg": 1.0,
        "reflection_count": 0.0,
        "reflection_path_count": 0.0,
        "path_count": 1.0,
        "first_path_angle_deg": 0.0,
        "los_angle_deg": 0.0,
    }


class RelationalDelayCorrectionTest(unittest.TestCase):
    def test_equality_calibration_handles_first_delay_below_los_delay(self) -> None:
        corrected = _apply_signal_description_correction(
            _record(first_delay=10.0, los_delay=20.0), "relational"
        )

        self.assertEqual(corrected["first_path_delay_ns"], 20.0)
        self.assertEqual(corrected["los_delay_ns"], 20.0)

    def test_equality_calibration_handles_first_delay_above_los_delay(self) -> None:
        corrected = _apply_signal_description_correction(
            _record(first_delay=30.0, los_delay=20.0), "relational"
        )

        self.assertEqual(corrected["first_path_delay_ns"], 20.0)
        self.assertEqual(corrected["los_delay_ns"], 20.0)

    def test_relational_correction_does_not_change_nlos_delays(self) -> None:
        corrected = _apply_signal_description_correction(
            _record(first_delay=30.0, los_delay=20.0, status="nlos"),
            "relational",
        )

        self.assertEqual(corrected["first_path_delay_ns"], 30.0)
        self.assertEqual(corrected["los_delay_ns"], 20.0)

    def test_consistency_diagnostic_flags_both_directions(self) -> None:
        common = {
            "los_delay_tolerance_ns": 50.0,
            "los_angle_tolerance_deg": 15.0,
        }

        below = consistency_violations(_record(10.0, 20.0), **common)
        above = consistency_violations(_record(30.0, 20.0), **common)

        self.assertIn("los_first_delay_unequal", below)
        self.assertIn("los_first_delay_unequal", above)

    def test_training_regularizer_penalizes_both_directions_equally(self) -> None:
        target_count = len(PHYSICS_TARGET_NAMES)
        physics_target_mask = torch.zeros((2, target_count), dtype=torch.bool)
        los_delay_mask = torch.ones(2, dtype=torch.bool)
        scale = float(
            PHYSICS_TARGET_SCALES[
                PHYSICS_TARGET_NAMES.index("first_path_delay_ns")
            ]
        )
        physics_outputs = {
            "final": torch.zeros((2, target_count)),
            "first_path_delay_bin_soft_fused_raw": torch.tensor([10.0, 30.0]),
            "los_delay_context": torch.tensor([20.0 / scale, 20.0 / scale]),
        }
        batch = {
            "physics_target_mask": physics_target_mask,
            "los_delay_target_mask": los_delay_mask,
            "semantic_keys": [
                SimpleNamespace(los_status="los"),
                SimpleNamespace(los_status="los"),
            ],
        }

        losses = Trainer._physics_relational_losses(
            object.__new__(Trainer),
            physics_outputs,
            batch,
        )

        expected = torch.tensor(10.0 / scale)
        self.assertTrue(
            torch.isclose(
                losses["loss_physics_relational_first_path_delay_eq_los_delay"],
                expected,
            )
        )


if __name__ == "__main__":
    unittest.main()
