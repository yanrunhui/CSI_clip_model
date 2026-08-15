#!/usr/bin/env bash
set -euo pipefail

CKPT_ROOT=${CKPT_ROOT:-artifacts/dev_holdout_nf128_expert_gate}
DATA_ROOT=${DATA_ROOT:-artifacts/d2los_6k_multiconfig_final_holdout_seed45678}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/final_test_nf128_text_oracle_seed45678}
CUDA_DEVICE=${CUDA_DEVICE:-2}
BATCH_SIZE=${BATCH_SIZE:-128}
CONTEXT_MODE=${CONTEXT_MODE:-oracle_context}
CORRECTION=${CORRECTION:-relational}
BASE_PAYLOAD_ROOT=${BASE_PAYLOAD_ROOT:-}

mkdir -p "$OUTPUT_ROOT"

run_pair() {
  local seed=$1
  local pair_name=$2
  local name_a=$3
  local period_a=$4
  local path_a=$5
  local name_b=$6
  local period_b=$7
  local path_b=$8
  local output_dir="$OUTPUT_ROOT/seed_${seed}/${pair_name}"
  local base_payload_args=()

  if [[ "$CONTEXT_MODE" == "base_prediction" ]]; then
    if [[ -z "$BASE_PAYLOAD_ROOT" ]]; then
      echo "BASE_PAYLOAD_ROOT is required when CONTEXT_MODE=base_prediction." >&2
      return 2
    fi
    base_payload_args=(
      --base-payload
      "$BASE_PAYLOAD_ROOT/seed_${seed}/${pair_name}/signal_descriptions.pt"
    )
  fi

  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/build_multinumerology_signal_descriptions.py \
    --checkpoint "$CKPT_ROOT/seed_${seed}/multinumerology_generalization.pt" \
    --test-pair \
      "$name_a" "$period_a" "$path_a" \
      "$name_b" "$period_b" "$path_b" \
    "${base_payload_args[@]}" \
    --context-mode "$CONTEXT_MODE" \
    --signal-description-correction "$CORRECTION" \
    --batch-size "$BATCH_SIZE" \
    --output "$output_dir/signal_descriptions.pt" \
    > "$output_dir/build_output.txt"

  python scripts/evaluate_signal_descriptions.py \
    --payload "$output_dir/signal_descriptions.pt" \
    --output-dir "$output_dir" \
    > "$output_dir/text_evaluate_output.txt"
}

for seed in 0 1 2; do
  run_pair \
    "$seed" \
    nf96_nf128 \
    nf96 960 "$DATA_ROOT/d2los_6k_upa8x8_nf96_final_test.pt" \
    nf128 1280 "$DATA_ROOT/d2los_6k_upa8x8_nf128_final_test.pt"

  run_pair \
    "$seed" \
    nf128_nf192 \
    nf128 1280 "$DATA_ROOT/d2los_6k_upa8x8_nf128_final_test.pt" \
    nf192 1920 "$DATA_ROOT/d2los_6k_upa8x8_nf192_final_test.pt"
done

for pair_name in nf96_nf128 nf128_nf192; do
  python scripts/summarize_signal_description_text_metrics.py \
    --input "seed_0=$OUTPUT_ROOT/seed_0/$pair_name/signal_description_text_metrics.csv" \
    --input "seed_1=$OUTPUT_ROOT/seed_1/$pair_name/signal_description_text_metrics.csv" \
    --input "seed_2=$OUTPUT_ROOT/seed_2/$pair_name/signal_description_text_metrics.csv" \
    --output-csv "$OUTPUT_ROOT/${pair_name}_text_metrics_3seed_summary.csv" \
    --output-json "$OUTPUT_ROOT/${pair_name}_text_metrics_3seed_summary.json"
done

echo "saved_multinumerology_text_evaluation=$OUTPUT_ROOT"
