from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_multinumerology_delay_fusion import safe_pearson  # noqa: E402
from scripts.train_multinumerology_delay_fusion import (  # noqa: E402
    TARGET_NAMES,
    move_nested,
)
from scripts.train_multinumerology_generalization import (  # noqa: E402
    FINAL_OUTPUT_METHOD,
    PairSpec,
    PeriodConditionedDelayFusion,
    make_loaders,
    parse_pair,
)


METHOD_KEYS = (
    ("single_a", "single_a"),
    ("single_b", "single_b"),
    ("fused_direct", "paired_fused_direct"),
    ("fused_expert_soft", "paired_expert_soft"),
    ("fused_gate_ab_hard", FINAL_OUTPUT_METHOD),
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SEED=CHECKPOINT_PATH.")
    seed, raw_path = value.split("=", 1)
    seed = seed.strip()
    path = Path(raw_path.strip())
    if not seed or not str(path):
        raise argparse.ArgumentTypeError("Expected non-empty SEED=CHECKPOINT_PATH.")
    return seed, path


def metric_record(
    *,
    seed: str,
    pair: str,
    target_name: str,
    method: str,
    los_status: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, str | int | float]:
    count = int(targets.numel())
    record: dict[str, str | int | float] = {
        "seed": seed,
        "pair": pair,
        "target": target_name,
        "method": method,
        "los_status": los_status,
        "count": count,
        "MAE_ns": math.nan,
        "RMSE_ns": math.nan,
        "signed_mean_ns": math.nan,
        "pearson": math.nan,
        "accuracy_at_50ns": math.nan,
    }
    if count == 0:
        return record
    predictions = predictions.float()
    targets = targets.float()
    errors = predictions - targets
    record.update(
        {
            "MAE_ns": float(errors.abs().mean()),
            "RMSE_ns": float(torch.sqrt(errors.square().mean())),
            "signed_mean_ns": float(errors.mean()),
            "pearson": safe_pearson(predictions, targets),
            "accuracy_at_50ns": float((errors.abs() <= 50.0).float().mean()),
        }
    )
    return record


@torch.no_grad()
def evaluate_checkpoint(
    *,
    seed: str,
    checkpoint_path: Path,
    test_specs: list[PairSpec],
    held_out_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[dict[str, str | int | float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_held_out = checkpoint_args.get("held_out_name")
    if checkpoint_held_out not in (None, held_out_name):
        raise ValueError(
            f"Checkpoint {checkpoint_path} held_out_name={checkpoint_held_out!r}, "
            f"requested={held_out_name!r}."
        )

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
    model = model.to(device).eval()

    sample_cache: dict[str, tuple[list[object], dict[str, object]]] = {}
    loaders, _ = make_loaders(
        test_specs,
        max_delay_ns=max_delay_ns,
        max_delay_spread_ns=max_delay_spread_ns,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=int(checkpoint_args.get("seed", 0)),
        shuffle=False,
        sample_cache=sample_cache,
    )

    rows: list[dict[str, str | int | float]] = []
    for spec, loader in loaders:
        grouped_a = sample_cache[spec.path_a][1]
        collected = {
            target_name: {
                "targets": {"los": [], "nlos": []},
                "predictions": {
                    method: {"los": [], "nlos": []} for _, method in METHOD_KEYS
                },
            }
            for target_name in TARGET_NAMES
        }
        for batch in loader:
            statuses = [
                str(grouped_a[group_id].semantic_key.los_status)
                for group_id in batch["group_ids"]
            ]
            moved = move_nested(batch, device)
            outputs = model(
                moved["view_a"],
                moved["view_b"],
                moved["period_a_ns"],
                moved["period_b_ns"],
                moved["availability_a"],
                moved["availability_b"],
            )
            for target_name in TARGET_NAMES:
                valid_mask = moved["masks"][target_name]
                for los_status in ("los", "nlos"):
                    status_mask = torch.tensor(
                        [status == los_status for status in statuses],
                        dtype=torch.bool,
                        device=device,
                    )
                    mask = valid_mask & status_mask
                    if not bool(mask.any()):
                        continue
                    collected[target_name]["targets"][los_status].append(
                        moved["targets"][target_name][mask].cpu()
                    )
                    for output_key, method in METHOD_KEYS:
                        collected[target_name]["predictions"][method][
                            los_status
                        ].append(outputs[target_name][output_key][mask].cpu())

        for target_name in TARGET_NAMES:
            for method in [method for _, method in METHOD_KEYS]:
                for los_status in ("los", "nlos"):
                    target_parts = collected[target_name]["targets"][los_status]
                    prediction_parts = collected[target_name]["predictions"][method][
                        los_status
                    ]
                    targets = (
                        torch.cat(target_parts)
                        if target_parts
                        else torch.empty(0, dtype=torch.float32)
                    )
                    predictions = (
                        torch.cat(prediction_parts)
                        if prediction_parts
                        else torch.empty(0, dtype=torch.float32)
                    )
                    rows.append(
                        metric_record(
                            seed=seed,
                            pair=spec.name,
                            target_name=target_name,
                            method=method,
                            los_status=los_status,
                            predictions=predictions,
                            targets=targets,
                        )
                    )

    del model, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def finite_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean_sample_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, math.nan
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def summarize(rows: list[dict]) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (
            str(row["pair"]),
            str(row["target"]),
            str(row["method"]),
            str(row["los_status"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (pair, target, method, los_status), group_rows in sorted(grouped.items()):
        summary: dict[str, str | int | float] = {
            "pair": pair,
            "target": target,
            "method": method,
            "los_status": los_status,
            "seeds": ",".join(str(row["seed"]) for row in group_rows),
            "n": len(group_rows),
            "count_per_seed": int(group_rows[0]["count"]),
        }
        for metric in (
            "MAE_ns",
            "RMSE_ns",
            "signed_mean_ns",
            "pearson",
            "accuracy_at_50ns",
        ):
            mean, sample_std = mean_sample_std(finite_values(group_rows, metric))
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sample_std"] = sample_std
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_named_path,
        required=True,
        metavar="SEED=PATH",
    )
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    test_specs = [parse_pair(values) for values in args.test_pair]
    if any(args.held_out_name not in (spec.name_a, spec.name_b) for spec in test_specs):
        raise ValueError(f"Every test pair must contain {args.held_out_name!r}.")
    for seed, checkpoint_path in args.checkpoint:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"final_output_method={FINAL_OUTPUT_METHOD}")
    all_rows = []
    for seed, checkpoint_path in args.checkpoint:
        print(f"evaluating_seed={seed}")
        all_rows.extend(
            evaluate_checkpoint(
                seed=seed,
                checkpoint_path=checkpoint_path,
                test_specs=test_specs,
                held_out_name=args.held_out_name,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
        )

    summary_rows = summarize(all_rows)
    per_seed_path = args.output_dir / "los_nlos_metrics_per_seed.csv"
    summary_path = args.output_dir / "los_nlos_metrics_3seed_summary.csv"
    write_csv(per_seed_path, all_rows)
    write_csv(summary_path, summary_rows)
    (args.output_dir / "los_nlos_metrics_3seed_summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for row in summary_rows:
        if row["method"] != FINAL_OUTPUT_METHOD:
            continue
        print(
            f"{row['pair']}_{row['target']}_{row['los_status']}_"
            f"{FINAL_OUTPUT_METHOD}_count={row['count_per_seed']}"
        )
        print(
            f"{row['pair']}_{row['target']}_{row['los_status']}_"
            f"{FINAL_OUTPUT_METHOD}_MAE="
            f"{float(row['MAE_ns_mean']):.4f} +/- "
            f"{float(row['MAE_ns_sample_std']):.4f} ns"
        )
        print(
            f"{row['pair']}_{row['target']}_{row['los_status']}_"
            f"{FINAL_OUTPUT_METHOD}_accuracy_at_50ns="
            f"{float(row['accuracy_at_50ns_mean']):.4f} +/- "
            f"{float(row['accuracy_at_50ns_sample_std']):.4f}"
        )
    print(f"saved_per_seed_csv={per_seed_path}")
    print(f"saved_3seed_summary_csv={summary_path}")


if __name__ == "__main__":
    main()
