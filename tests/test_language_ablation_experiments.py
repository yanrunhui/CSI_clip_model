from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ablation_experiments import (
    ABLATIONS,
    build_text_evaluate_command,
    format_mean_std,
    selected_specs,
)


def _spec(name: str):
    return next(spec for spec in ABLATIONS if spec.name == name)


class LanguageAblationExperimentTest(unittest.TestCase):
    def test_requested_language_ablations_have_exact_loss_overrides(self) -> None:
        self.assertEqual(
            _spec("no_csi_text_alignment").train_overrides,
            ("--csi-to-text-weight", "0"),
        )
        self.assertEqual(
            _spec("no_text_prototype_alignment").train_overrides,
            ("--text-prototype-weight", "0"),
        )
        self.assertEqual(
            _spec("physics_only_same_verbalizer").train_overrides,
            (
                "--csi-to-text-weight",
                "0",
                "--prototype-weight",
                "0",
                "--text-prototype-weight",
                "0",
                "--prototype-warmup-epochs",
                "0",
                "--freeze-text-prototypes",
            ),
        )

    def test_physics_only_uses_the_shared_verbalizer_contract(self) -> None:
        output_dir = Path("experiment")
        spec = _spec("physics_only_same_verbalizer")
        command = build_text_evaluate_command(output_dir)

        self.assertEqual(spec.csi_text_alignment, "off")
        self.assertEqual(spec.csi_prototype_alignment, "off")
        self.assertEqual(spec.text_prototype_alignment, "off")
        self.assertEqual(spec.verbalizer, "deterministic_signal_description")
        self.assertEqual(command[1], "scripts/evaluate_signal_descriptions.py")
        self.assertIn(str(output_dir / "signal_descriptions.pt"), command)

    def test_language_group_contains_only_requested_controls(self) -> None:
        self.assertEqual(
            [spec.name for spec in selected_specs(("language",))],
            [
                "no_csi_text_alignment",
                "no_text_prototype_alignment",
                "physics_only_same_verbalizer",
            ],
        )

    def test_ablation_aggregation_reports_sample_standard_deviation(self) -> None:
        mean, std = format_mean_std([1.0, 2.0, 3.0])

        self.assertEqual(float(mean), 2.0)
        self.assertTrue(math.isclose(float(std), 1.0))


if __name__ == "__main__":
    unittest.main()
