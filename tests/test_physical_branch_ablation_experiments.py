from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ablation_experiments import ABLATIONS, selected_specs


def _spec(name: str):
    return next(spec for spec in ABLATIONS if spec.name == name)


class PhysicalBranchAblationExperimentTest(unittest.TestCase):
    def test_requested_ablation_overrides(self) -> None:
        self.assertEqual(
            _spec("no_power_branch").train_overrides,
            ("--disable-power-branch",),
        )
        self.assertEqual(
            _spec("no_los_angle_context_encoder").train_overrides,
            ("--no-use-los-angle-context-encoder",),
        )
        self.assertEqual(
            _spec("no_first_path_angle_context_encoder").train_overrides,
            ("--no-use-first-path-angle-context-encoder",),
        )
        self.assertEqual(
            _spec("no_los_consistency").train_overrides,
            (
                "--no-use-physics-calibration-loss",
                "--los-delay-consistency-weight",
                "0",
            ),
        )
        self.assertEqual(
            _spec("no_reflection_aux_head").train_overrides,
            (
                "--reflection-count-classifier-weight",
                "0",
                "--reflection-count-regression-weight",
                "0",
                "--reflection-path-count-regression-weight",
                "0",
                "--interaction-count-classifier-weight",
                "0",
                "--interaction-count-regression-weight",
                "0",
            ),
        )

    def test_group_has_exactly_the_requested_ablations(self) -> None:
        self.assertEqual(
            [spec.name for spec in selected_specs(("physical_branches",))],
            [
                "no_power_branch",
                "no_los_angle_context_encoder",
                "no_first_path_angle_context_encoder",
                "no_reflection_aux_head",
                "no_los_consistency",
            ],
        )


if __name__ == "__main__":
    unittest.main()
