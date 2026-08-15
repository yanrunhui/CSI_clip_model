#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_ID="${GPU_ID:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
DATA_ROOT="${DATA_ROOT:-artifacts/d2los_100k_multiconfig_aligned}"
CHECKPOINT="${CHECKPOINT:-artifacts/pretrain_d2los_100k_3array_nf128_joint/seed_0/checkpoint_epoch_30.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/generalization_3array_nf128_joint_seed0}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "checkpoint_not_found=$CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

evaluate_one() {
  local name="$1"
  local test_file="$2"
  local split_type="$3"
  local data_path="$DATA_ROOT/$test_file"
  local output_dir="$OUTPUT_ROOT/$name"

  if [[ ! -f "$data_path" ]]; then
    echo "test_data_not_found=$data_path" >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  echo "=== evaluating $name ($split_type) ==="
  echo "data_path=$data_path"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate.py \
    --data-path "$data_path" \
    --checkpoint "$CHECKPOINT" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --verbose-diagnostics \
    --signal-description-correction relational \
    --save-signal-descriptions "$output_dir/signal_descriptions.pt" \
    > "$output_dir/evaluate_output.txt" 2>&1

  python scripts/evaluate_signal_descriptions.py \
    --payload "$output_dir/signal_descriptions.pt" \
    --output-dir "$output_dir" \
    > "$output_dir/text_evaluate_output.txt" 2>&1

  printf '%s\n' "$split_type" > "$output_dir/evaluation_type.txt"
  echo "completed_configuration=$name"
}

# These configurations appeared during joint training.
evaluate_one \
  "seen_upa8x8_nf128" \
  "d2los_100k_upa8x8_nf128_los50k_nlos50k_test.pt" \
  "seen_nf128_configuration"
evaluate_one \
  "seen_upa4x4_nf128" \
  "d2los_100k_upa4x4_nf128_los50k_nlos50k_test.pt" \
  "seen_nf128_configuration"
evaluate_one \
  "seen_ula64_nf128" \
  "d2los_100k_ula64_nf128_los50k_nlos50k_test.pt" \
  "seen_nf128_configuration"

# These source-nf values were not present during training.
evaluate_one \
  "unseen_upa8x8_nf64" \
  "d2los_100k_upa8x8_nf64_los50k_nlos50k_test.pt" \
  "unseen_nf_generalization"
evaluate_one \
  "unseen_upa8x8_nf96" \
  "d2los_100k_upa8x8_nf96_los50k_nlos50k_test.pt" \
  "unseen_nf_generalization"
evaluate_one \
  "unseen_upa8x8_nf192" \
  "d2los_100k_upa8x8_nf192_los50k_nlos50k_test.pt" \
  "unseen_nf_generalization"
evaluate_one \
  "unseen_upa8x8_nf256" \
  "d2los_100k_upa8x8_nf256_los50k_nlos50k_test.pt" \
  "unseen_nf_generalization"

grep -H -E \
  '^(first_path_delay_context_MAE|first_path_delay_los_MAE|first_path_delay_nlos_MAE|los_delay_context_MAE|los_angle_MAE|first_path_angle_los_MAE|first_path_angle_nlos_MAE|delay_spread_MAE|k_factor_db_MAE|azimuth_spread_MAE|base_first_power_MAE|reflection_count_head_MAE|reflection_count_head_accuracy|reflection_path_count_head_MAE|reflection_path_count_head_exact_accuracy)=' \
  "$OUTPUT_ROOT"/*/evaluate_output.txt \
  > "$OUTPUT_ROOT/generalization_physics_summary.txt"

{
  for output_dir in "$OUTPUT_ROOT"/*; do
    metrics_path="$output_dir/signal_description_text_metrics.csv"
    [[ -f "$metrics_path" ]] || continue
    echo "=== $(basename "$output_dir") ==="
    grep -E \
      'description_factual_accuracy|slot_f1,all_numeric_slots|hallucination_rate,all_numeric_slots|numerical_slot_accuracy,all_numeric_slots|physical_consistency_rate,predicted_description' \
      "$metrics_path"
  done
} > "$OUTPUT_ROOT/generalization_text_summary.txt"

echo "saved_physics_summary=$OUTPUT_ROOT/generalization_physics_summary.txt"
echo "saved_text_summary=$OUTPUT_ROOT/generalization_text_summary.txt"
echo "=== evaluation completed ==="
