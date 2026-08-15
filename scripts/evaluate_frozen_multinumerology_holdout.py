from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_multinumerology_generalization import (  # noqa: E402
    FINAL_OUTPUT_METHOD,
    PairSpec,
    PeriodConditionedDelayFusion,
    evaluate_pair,
    make_loaders,
    parse_pair,
)


FROZEN_SHARED_BASELINE_METHODS = (
    "paired_fused_direct",
    "paired_expert_soft",
    FINAL_OUTPUT_METHOD,
)
FIXED_MAX_PERIOD_METHOD = "fixed_max_period_branch"


def frozen_methods(spec: PairSpec) -> tuple[str, ...]:
    return (
        f"single_{spec.name_a}",
        f"single_{spec.name_b}",
        *FROZEN_SHARED_BASELINE_METHODS,
        FIXED_MAX_PERIOD_METHOD,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--test-pair",
        action="append",
        nargs=6,
        required=True,
        metavar=(
            "NAME_A",
            "PERIOD_A_NS",
            "PATH_A",
            "NAME_B",
            "PERIOD_B_NS",
            "PATH_B",
        ),
    )
    parser.add_argument("--held-out-name", default="nf128")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--report-frozen-baselines",
        action="store_true",
        help="Report the pre-registered single, direct, soft, hard, and max-period baselines.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_held_out = checkpoint_args.get("held_out_name")
    if checkpoint_held_out not in (None, args.held_out_name):
        raise ValueError(
            f"Checkpoint held_out_name={checkpoint_held_out!r}, "
            f"requested={args.held_out_name!r}."
        )

    test_specs: list[PairSpec] = [parse_pair(values) for values in args.test_pair]
    if any(args.held_out_name not in (spec.name_a, spec.name_b) for spec in test_specs):
        raise ValueError(f"Every final test pair must contain {args.held_out_name!r}.")
    max_delay_ns = float(checkpoint_args.get("max_delay_ns", 1920.0))
    max_delay_spread_ns = checkpoint_args.get("max_delay_spread_ns", 400.0)
    max_delay_spread_ns = (
        None if max_delay_spread_ns is None else float(max_delay_spread_ns)
    )
    model = PeriodConditionedDelayFusion(
        max_delay_ns=max_delay_ns,
        hidden_dim=int(checkpoint_args.get("hidden_dim", 256)),
        residual_scale_ns=float(checkpoint_args.get("residual_scale_ns", 50.0)),
        period_prior_strength=float(checkpoint_args.get("period_prior_strength", 0.0)),
        fallback_confidence_threshold=float(
            checkpoint_args.get("fallback_confidence_threshold", 0.0)
        ),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sample_cache: dict[str, tuple[list[object], dict[str, object]]] = {}
    test_loaders, test_info = make_loaders(
        test_specs,
        max_delay_ns=max_delay_ns,
        max_delay_spread_ns=max_delay_spread_ns,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=int(checkpoint_args.get("seed", 0)),
        shuffle=False,
        sample_cache=sample_cache,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")
    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_seed={checkpoint_args.get('seed', 'unknown')}")
    print(f"final_output_method={FINAL_OUTPUT_METHOD}")
    print(f"report_frozen_baselines={args.report_frozen_baselines}")
    print(f"held_out_numerology={args.held_out_name}")
    all_rows = []
    all_summary = {}
    for spec, loader in test_loaders:
        print(
            f"test_pair_info_{spec.name}={json.dumps(test_info[spec.name], sort_keys=True)}"
        )
        rows, summary = evaluate_pair(model, spec, loader, device)
        selected_methods = (
            frozen_methods(spec)
            if args.report_frozen_baselines
            else (FINAL_OUTPUT_METHOD,)
        )
        selected_rows = [row for row in rows if row["method"] in selected_methods]
        if args.report_frozen_baselines:
            max_period_method = f"single_{spec.name_b}"
            selected_rows.extend(
                {
                    **row,
                    "method": FIXED_MAX_PERIOD_METHOD,
                }
                for row in rows
                if row["method"] == max_period_method
            )
        all_rows.extend(selected_rows)
        all_summary[spec.name] = {}
        for target_name, target_summary in summary.items():
            final_metrics = target_summary["final_metrics"]
            all_summary[spec.name][target_name] = {
                "count": target_summary["count"],
                "final_output_method": FINAL_OUTPUT_METHOD,
                "final_metrics": final_metrics,
            }
            if args.report_frozen_baselines:
                reported_methods = {
                    method: target_summary["methods"][method]
                    for method in selected_methods
                    if method != FIXED_MAX_PERIOD_METHOD
                }
                reported_methods[FIXED_MAX_PERIOD_METHOD] = {
                    **target_summary["methods"][f"single_{spec.name_b}"],
                    "method": FIXED_MAX_PERIOD_METHOD,
                }
                all_summary[spec.name][target_name][
                    "frozen_baselines"
                ] = reported_methods
            print(f"{spec.name}_{target_name}_count={target_summary['count']}")
            methods_to_print = (
                reported_methods
                if args.report_frozen_baselines
                else {FINAL_OUTPUT_METHOD: final_metrics}
            )
            for method, metrics in methods_to_print.items():
                print(
                    f"{spec.name}_{target_name}_{method}_MAE="
                    f"{float(metrics['MAE']):.4f}"
                )
                print(
                    f"{spec.name}_{target_name}_{method}_accuracy_at_50ns="
                    f"{float(metrics['accuracy_at_50ns']):.4f}"
                )

    metrics_path = args.output_dir / "final_test_metrics.csv"
    fieldnames = [
        "pair",
        "field",
        "method",
        "target_range",
        "count",
        "MAE",
        "RMSE",
        "signed_mean",
        "pearson",
        "accuracy_at_50ns",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    summary_path = args.output_dir / "final_test_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_seed": checkpoint_args.get("seed"),
                "final_output_method": FINAL_OUTPUT_METHOD,
                "held_out_numerology": args.held_out_name,
                "report_frozen_baselines": args.report_frozen_baselines,
                "frozen_baseline_methods": (
                    {spec.name: frozen_methods(spec) for spec in test_specs}
                    if args.report_frozen_baselines
                    else {}
                ),
                "test_info": test_info,
                "summary": all_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved_final_metrics_csv={metrics_path}")
    print(f"saved_final_summary_json={summary_path}")


if __name__ == "__main__":
    main()
