from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_noise_robustness_by_model_seed import (  # noqa: E402
    PRIMARY_METRICS,
    aggregate_by_model_seed,
)


def make_run(model_seed: str, noise_seed: int, value: float) -> dict[str, object]:
    run: dict[str, object] = {
        "model_seed": model_seed,
        "condition": "snr_20_db",
        "target_snr_db": 20.0,
        "noise_seed": noise_seed,
        "actual_snr_mean_db": 20.0,
        "numeric_accuracy_drop_percentage_points": value,
    }
    for metric in PRIMARY_METRICS:
        run[metric] = value
        if metric != "numeric_accuracy":
            run[f"{metric}_relative_degradation_pct"] = value
    return run


def test_averages_noise_first_then_computes_model_seed_std() -> None:
    runs = [
        make_run("0", 100, 1.0),
        make_run("0", 101, 3.0),
        make_run("1", 100, 5.0),
        make_run("1", 101, 7.0),
    ]

    model_means, summary = aggregate_by_model_seed(runs)

    assert [row["first_delay_mae_ns"] for row in model_means] == [2.0, 6.0]
    assert summary[0]["model_seed_count"] == 2
    assert summary[0]["noise_runs_per_model_min"] == 2
    assert summary[0]["noise_runs_per_model_max"] == 2
    assert summary[0]["first_delay_mae_ns_mean"] == 4.0
    assert math.isclose(
        summary[0]["first_delay_mae_ns_std"],
        math.sqrt(8.0),
    )
