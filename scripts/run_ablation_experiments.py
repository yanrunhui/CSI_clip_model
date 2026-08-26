from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KEY_METRICS = (
    "first_path_delay_context_MAE",
    "first_path_delay_los_MAE",
    "first_path_delay_nlos_MAE",
    "first_path_delay_bin_fused_MAE",
    "first_path_delay_bin_soft_fused_MAE",
    "delay_spread_MAE",
    "k_factor_db_MAE",
    "strong_k_MAE",
    "base_first_power_MAE",
    "enhanced_first_power_MAE",
    "final_first_power_MAE",
    "los_first_power_MAE",
    "nlos_first_power_MAE",
    "los_delay_context_MAE",
    "los_angle_MAE",
    "first_path_angle_los_MAE",
    "first_path_angle_nlos_MAE",
    "reflection_count_head_MAE",
    "reflection_count_head_accuracy",
    "reflection_count_head_adjacent_accuracy",
    "reflection_count_head_pearson",
    "n_paths_MAE",
    "azimuth_spread_MAE",
)

TRAIN_METADATA_KEYS = (
    "gpu_name",
    "gpu_count",
    "cuda_device_capability",
    "cuda_version",
    "cudnn_version",
    "torch_version",
    "device",
    "seed",
    "amp_enabled",
    "model_parameters",
    "model_trainable_parameters",
    "training_start_time",
    "training_end_time",
    "training_elapsed_seconds",
    "training_elapsed_hms",
)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    purpose: str
    train_overrides: tuple[str, ...] = ()
    kind: str = "pretrain"
    csi_text_alignment: str = "on"
    csi_prototype_alignment: str = "on"
    text_prototype_alignment: str = "on"
    verbalizer: str = "deterministic_signal_description"


ABLATIONS = (
    AblationSpec(
        name="full_multitask",
        purpose="Reference full multi-task physics model.",
    ),
    AblationSpec(
        name="no_csi_text_alignment",
        purpose="Remove only CSI-to-text alignment while keeping the other objectives fixed.",
        train_overrides=("--csi-to-text-weight", "0"),
        csi_text_alignment="off",
    ),
    AblationSpec(
        name="no_text_prototype_alignment",
        purpose="Remove only text-to-prototype alignment while keeping the other objectives fixed.",
        train_overrides=("--text-prototype-weight", "0"),
        text_prototype_alignment="off",
    ),
    AblationSpec(
        name="physics_only_same_verbalizer",
        purpose=(
            "Remove all three language/prototype alignment losses, train from structured "
            "physical supervision, and evaluate with the same deterministic verbalizer."
        ),
        train_overrides=(
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
        csi_text_alignment="off",
        csi_prototype_alignment="off",
        text_prototype_alignment="off",
    ),
    AblationSpec(
        name="no_shared_physics_token",
        purpose="Verify whether the residual shared physics representation helps.",
        train_overrides=("--no-use-shared-physics-token",),
    ),
    AblationSpec(
        name="no_delay_specific_encoder",
        purpose="Verify whether the delay-specific encoder branch helps delay tasks.",
        train_overrides=("--no-use-delay-specific-encoder",),
    ),
    AblationSpec(
        name="no_los_consistency",
        purpose="Verify LoS first-path-delay/LoS-delay consistency supervision.",
        train_overrides=(
            "--no-use-physics-calibration-loss",
            "--los-delay-consistency-weight",
            "0",
        ),
    ),
    AblationSpec(
        name="no_reflection_aux_head",
        purpose="Verify whether reflection-count auxiliary supervision helps shared representations.",
        train_overrides=(
            "--reflection-count-classifier-weight",
            "0",
            "--reflection-count-regression-weight",
            "0",
            "--interaction-count-classifier-weight",
            "0",
            "--interaction-count-regression-weight",
            "0",
        ),
    ),
    AblationSpec(
        name="single_task_csi_encoder",
        purpose="Single-task CSI encoder baseline for multi-task vs single-task comparison.",
        kind="single_task_baseline",
        csi_text_alignment="n/a",
        csi_prototype_alignment="n/a",
        text_prototype_alignment="n/a",
        verbalizer="none",
    ),
)

ABLATION_GROUPS = {
    "language": (
        "no_csi_text_alignment",
        "no_text_prototype_alignment",
        "physics_only_same_verbalizer",
    ),
    "architecture": (
        "full_multitask",
        "no_shared_physics_token",
        "no_delay_specific_encoder",
        "no_los_consistency",
        "no_reflection_aux_head",
        "single_task_csi_encoder",
    ),
}


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_metric_lines(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics
    pattern = re.compile(r"^([A-Za-z0-9_@.-]+)=([^=\n]+)$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            metrics[match.group(1)] = match.group(2)
    return metrics


def latest_checkpoint(output_dir: Path, preferred_epoch: int | None) -> Path | None:
    if preferred_epoch is not None:
        checkpoint = output_dir / f"checkpoint_epoch_{preferred_epoch}.pt"
        if checkpoint.exists():
            return checkpoint
    last_checkpoint = output_dir / "checkpoint_last.pt"
    if last_checkpoint.exists():
        return last_checkpoint
    checkpoints = sorted(
        output_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def build_pretrain_command(spec: AblationSpec, args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/pretrain.py",
        "--config",
        args.config,
        "--data-path",
        args.train_data,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--save-every",
        str(args.save_every),
        "--seed",
        str(args.current_seed),
    ]
    if args.max_steps_per_epoch is not None:
        command.extend(["--max-steps-per-epoch", str(args.max_steps_per_epoch)])
    command.extend(spec.train_overrides)
    return command


def build_evaluate_command(
    checkpoint: Path,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    power_gate_mode: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--data-path",
        args.test_data,
        "--checkpoint",
        str(checkpoint),
        "--batch-size",
        str(args.eval_batch_size),
        "--save-signal-descriptions",
        str(output_dir / "signal_descriptions.pt"),
        "--signal-description-correction",
        args.signal_description_correction,
        "--verbose-diagnostics",
    ]
    if power_gate_mode is not None:
        command.extend(["--first-path-power-gate-mode", power_gate_mode])
    return command


def build_text_evaluate_command(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_signal_descriptions.py",
        "--payload",
        str(output_dir / "signal_descriptions.pt"),
        "--output-dir",
        str(output_dir),
    ]


def build_single_task_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/train_physics_baselines.py",
        "--train-data",
        args.train_data,
        "--test-data",
        args.test_data,
        "--model",
        "csi_encoder_single_task",
        "--target",
        "all",
        "--epochs",
        str(args.baseline_epochs),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.current_seed),
    ]
    if args.max_steps_per_epoch is not None:
        limit_train = args.max_steps_per_epoch * args.batch_size
        command.extend(["--limit-train", str(limit_train)])
    return command


def selected_specs(names: tuple[str, ...]) -> list[AblationSpec]:
    by_name = {spec.name: spec for spec in ABLATIONS}
    if names == ("all",):
        return list(ABLATIONS)
    expanded_names = []
    for name in names:
        expanded_names.extend(ABLATION_GROUPS.get(name, (name,)))
    deduplicated_names = tuple(dict.fromkeys(expanded_names))
    unknown = [name for name in deduplicated_names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown ablation(s): {', '.join(unknown)}")
    return [by_name[name] for name in deduplicated_names]


def write_commands(commands: list[tuple[str, list[str]]], path: Path) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for name, command in commands:
        lines.append(f"echo '=== {name} ==='")
        lines.append(shell_join(command))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def format_mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    return f"{mean:.6g}", f"{math.sqrt(variance):.6g}"


def parse_text_metrics(path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if not path.exists():
        return metrics
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            metric = str(row.get("metric", "")).strip()
            field = str(row.get("field", "")).strip()
            value = str(row.get("value", "")).strip()
            if not metric or not field:
                continue
            metrics[f"text_{metric}_{field}"] = value
    return metrics


def run_output_dir(output_root: Path, spec: AblationSpec, seed: int, multi_seed: bool) -> Path:
    base = output_root / spec.name
    return base / f"seed_{seed}" if multi_seed else base


def summarize(
    output_root: Path,
    specs: list[AblationSpec],
    seeds: tuple[int, ...],
    summary_csv: Path,
) -> None:
    rows: list[dict[str, str]] = []
    multi_seed = len(seeds) > 1
    for spec in specs:
        for seed in seeds:
            output_dir = run_output_dir(output_root, spec, seed, multi_seed)
            row = {
                "ablation": spec.name,
                "seed": str(seed),
                "purpose": spec.purpose,
                "kind": spec.kind,
                "csi_text_alignment": spec.csi_text_alignment,
                "csi_prototype_alignment": spec.csi_prototype_alignment,
                "text_prototype_alignment": spec.text_prototype_alignment,
                "verbalizer": spec.verbalizer,
            }
            if spec.kind == "single_task_baseline":
                baseline_csv = output_dir / "baseline_results.csv"
                if baseline_csv.exists():
                    with baseline_csv.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for baseline_row in reader:
                            target = baseline_row.get("target", "")
                            row[f"{target}_MAE"] = baseline_row.get("MAE", "")
                            row[f"{target}_RMSE"] = baseline_row.get("RMSE", "")
                            if target == "reflection_count":
                                row["reflection_count_head_accuracy"] = baseline_row.get("accuracy", "")
                                row["reflection_count_head_adjacent_accuracy"] = baseline_row.get("adjacent_accuracy", "")
                                row["reflection_count_head_pearson"] = baseline_row.get("pearson", "")
                rows.append(row)
                continue

            metrics = parse_metric_lines(output_dir / "evaluate_output.txt")
            for metric in KEY_METRICS:
                row[metric] = metrics.get(metric, "")
            train_metadata = parse_metric_lines(output_dir / "train.log")
            for key in TRAIN_METADATA_KEYS:
                row[key] = train_metadata.get(key, "")
            row.update(
                parse_text_metrics(output_dir / "signal_description_text_metrics.csv")
            )

            base_metrics = parse_metric_lines(output_dir / "evaluate_power_base_output.txt")
            enhanced_metrics = parse_metric_lines(output_dir / "evaluate_power_enhanced_output.txt")
            if base_metrics:
                row["power_base_mode_los_first_power_MAE"] = base_metrics.get("los_first_power_MAE", "")
                row["power_base_mode_nlos_first_power_MAE"] = base_metrics.get("nlos_first_power_MAE", "")
            if enhanced_metrics:
                row["power_enhanced_mode_los_first_power_MAE"] = enhanced_metrics.get("los_first_power_MAE", "")
                row["power_enhanced_mode_nlos_first_power_MAE"] = enhanced_metrics.get("nlos_first_power_MAE", "")
            rows.append(row)

    descriptor_fields = [
        "ablation",
        "seed",
        "kind",
        "purpose",
        "csi_text_alignment",
        "csi_prototype_alignment",
        "text_prototype_alignment",
        "verbalizer",
    ]
    fieldnames = list(descriptor_fields)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_json = summary_csv.with_suffix(".json")
    summary_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved_ablation_summary_csv={summary_csv}")
    print(f"saved_ablation_summary_json={summary_json}")

    if len(seeds) <= 1:
        return

    aggregate_rows: list[dict[str, str]] = []
    metric_names = [name for name in fieldnames if name not in descriptor_fields]
    for spec in specs:
        spec_rows = [row for row in rows if row["ablation"] == spec.name]
        aggregate_row = {
            "ablation": spec.name,
            "seeds": ",".join(str(seed) for seed in seeds),
            "kind": spec.kind,
            "purpose": spec.purpose,
            "csi_text_alignment": spec.csi_text_alignment,
            "csi_prototype_alignment": spec.csi_prototype_alignment,
            "text_prototype_alignment": spec.text_prototype_alignment,
            "verbalizer": spec.verbalizer,
        }
        for metric in metric_names:
            values = [
                parsed
                for row in spec_rows
                if (parsed := parse_float(row.get(metric, ""))) is not None
            ]
            mean, std = format_mean_std(values)
            aggregate_row[f"{metric}_mean"] = mean
            aggregate_row[f"{metric}_std"] = std
        aggregate_rows.append(aggregate_row)

    aggregate_csv = summary_csv.with_name(summary_csv.stem + "_aggregate.csv")
    aggregate_fields = [
        "ablation",
        "seeds",
        "kind",
        "purpose",
        "csi_text_alignment",
        "csi_prototype_alignment",
        "text_prototype_alignment",
        "verbalizer",
    ]
    for row in aggregate_rows:
        for key in row:
            if key not in aggregate_fields:
                aggregate_fields.append(key)
    with aggregate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    aggregate_json = aggregate_csv.with_suffix(".json")
    aggregate_json.write_text(
        json.dumps(aggregate_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved_ablation_aggregate_csv={aggregate_csv}")
    print(f"saved_ablation_aggregate_json={aggregate_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="artifacts/d2los_30k_los_nlos_balanced_los_angle_train.pt")
    parser.add_argument("--test-data", default="artifacts/d2los_30k_los_nlos_balanced_los_angle_test.pt")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output-root", default="artifacts/ablations_d2los_30k_los_nlos_balanced_los_angle")
    parser.add_argument(
        "--commands-name",
        default="ablation_commands.sh",
        help="Filename for the generated shell command script under --output-root.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--checkpoint-epoch", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument(
        "--signal-description-correction",
        choices=("none", "bounds", "relational"),
        default="bounds",
        help="Use this identical pre-verbalization correction for every ablation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Single seed used when --seeds is omitted.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Run each selected ablation once per seed, e.g. --seeds 0 1 2.",
    )
    parser.add_argument(
        "--ablation",
        nargs="+",
        default=("all",),
        help=(
            "Ablations to include, or a group: language, architecture, all. "
            "Use --list to print names."
        ),
    )
    parser.add_argument("--run", action="store_true", help="Actually execute commands. Without this, only commands are written.")
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate/summarize existing checkpoints.")
    parser.add_argument("--skip-eval", action="store_true", help="Only train/generate commands; do not evaluate checkpoints.")
    parser.add_argument(
        "--skip-text-eval",
        action="store_true",
        help="Skip factual evaluation of the descriptions produced by the shared verbalizer.",
    )
    parser.add_argument("--summarize-only", action="store_true", help="Only rebuild ablation_summary.csv/json from existing outputs.")
    parser.add_argument("--list", action="store_true", help="List available ablations and exit.")
    args = parser.parse_args()

    if args.list:
        for spec in ABLATIONS:
            print(f"{spec.name}\t{spec.kind}\t{spec.purpose}")
        for group, names in ABLATION_GROUPS.items():
            print(f"group:{group}\t{','.join(names)}")
        return

    specs = selected_specs(tuple(args.ablation))
    seeds = tuple(args.seeds) if args.seeds is not None else (args.seed,)
    output_root = Path(args.output_root)
    summary_csv = output_root / "ablation_summary.csv"

    if args.summarize_only:
        summarize(output_root, specs, seeds, summary_csv)
        return

    planned_commands: list[tuple[str, list[str]]] = []
    multi_seed = len(seeds) > 1
    for seed in seeds:
        args.current_seed = seed
        for spec in specs:
            output_dir = run_output_dir(output_root, spec, seed, multi_seed)
            command_prefix = f"{spec.name}:seed_{seed}" if multi_seed else spec.name
            if spec.kind == "single_task_baseline":
                baseline_command = build_single_task_command(args, output_dir)
                planned_commands.append((f"{command_prefix}:train", baseline_command))
                if args.run and not args.skip_train:
                    code = run_command(baseline_command, output_dir / "baseline_train.log")
                    if code != 0:
                        raise SystemExit(code)
                continue

            train_command = build_pretrain_command(spec, args, output_dir)
            planned_commands.append((f"{command_prefix}:train", train_command))
            if args.run and not args.skip_train:
                code = run_command(train_command, output_dir / "train.log")
                if code != 0:
                    raise SystemExit(code)

            checkpoint = latest_checkpoint(output_dir, args.checkpoint_epoch or args.epochs)
            if checkpoint is None:
                checkpoint_epoch = args.checkpoint_epoch or args.epochs
                checkpoint = output_dir / f"checkpoint_epoch_{checkpoint_epoch}.pt"
                if args.run or args.skip_train:
                    print(
                        f"warning: using expected checkpoint path for {command_prefix}: {checkpoint}",
                        file=sys.stderr,
                    )
            eval_command = build_evaluate_command(checkpoint, args, output_dir)
            planned_commands.append((f"{command_prefix}:eval", eval_command))
            if args.run and not args.skip_eval:
                code = run_command(eval_command, output_dir / "evaluate_output.txt")
                if code != 0:
                    raise SystemExit(code)

            if not args.skip_text_eval:
                text_eval_command = build_text_evaluate_command(output_dir)
                planned_commands.append(
                    (f"{command_prefix}:eval_shared_verbalizer", text_eval_command)
                )
                if args.run and not args.skip_eval:
                    code = run_command(
                        text_eval_command,
                        output_dir / "evaluate_signal_descriptions_output.txt",
                    )
                    if code != 0:
                        raise SystemExit(code)

            if spec.name == "full_multitask":
                base_eval = build_evaluate_command(
                    checkpoint,
                    args,
                    output_dir,
                    power_gate_mode="base",
                )
                enhanced_eval = build_evaluate_command(
                    checkpoint,
                    args,
                    output_dir,
                    power_gate_mode="none",
                )
                planned_commands.append((f"{command_prefix}:eval_power_base", base_eval))
                planned_commands.append((f"{command_prefix}:eval_power_enhanced", enhanced_eval))
                if args.run and not args.skip_eval:
                    code = run_command(base_eval, output_dir / "evaluate_power_base_output.txt")
                    if code != 0:
                        raise SystemExit(code)
                    code = run_command(enhanced_eval, output_dir / "evaluate_power_enhanced_output.txt")
                    if code != 0:
                        raise SystemExit(code)

    commands_path = output_root / args.commands_name
    write_commands(planned_commands, commands_path)
    print(f"saved_ablation_commands={commands_path}")
    if not args.run:
        print("dry_run=true")
        print(f"Run with --run to execute the commands in order.")

    summarize(output_root, specs, seeds, summary_csv)


if __name__ == "__main__":
    main()
