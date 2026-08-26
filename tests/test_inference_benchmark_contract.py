from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_inference_cost_benchmark import measured_sample_identities


class InferenceBenchmarkIdentityTest(unittest.TestCase):
    def write_latency(self, rows: list[dict[str, object]]) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        cost_path = directory / "cost_summary.json"
        latency_path = directory / "per_sample_latency.csv"
        with latency_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("repeat", "sample_index", "group_id", "config_key"),
            )
            writer.writeheader()
            writer.writerows(rows)
        return cost_path

    def test_accepts_identical_identity_triples_across_repeats(self) -> None:
        rows = [
            {
                "repeat": repeat,
                "sample_index": index,
                "group_id": f"group-{index}",
                "config_key": "UPA-8x8",
            }
            for repeat in range(3)
            for index in range(2)
        ]
        identities = measured_sample_identities(self.write_latency(rows))
        self.assertEqual(
            identities,
            [(0, "group-0", "UPA-8x8"), (1, "group-1", "UPA-8x8")],
        )

    def test_rejects_config_mismatch_inside_one_repeat(self) -> None:
        rows = [
            {
                "repeat": repeat,
                "sample_index": index,
                "group_id": f"group-{index}",
                "config_key": (
                    "UPA-4x4" if repeat == 1 and index == 1 else "UPA-8x8"
                ),
            }
            for repeat in range(2)
            for index in range(2)
        ]
        with self.assertRaisesRegex(ValueError, "changed between repeats"):
            measured_sample_identities(self.write_latency(rows))

    def test_rejects_missing_sample_index(self) -> None:
        rows = [
            {
                "repeat": 0,
                "sample_index": index,
                "group_id": f"group-{index}",
                "config_key": "UPA-8x8",
            }
            for index in (0, 2)
        ]
        with self.assertRaisesRegex(ValueError, "Missing or duplicate sample_index"):
            measured_sample_identities(self.write_latency(rows))


if __name__ == "__main__":
    unittest.main()
