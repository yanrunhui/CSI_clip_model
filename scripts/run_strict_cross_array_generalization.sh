#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_ID=${GPU_ID:-2}
EPOCHS=${EPOCHS:-30}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
DATA_ROOT=${DATA_ROOT:-artifacts/d2los_100k_multiconfig_aligned}
UPA16_ROOT=${UPA16_ROOT:-artifacts/d2los_upa16x4_nf128_aligned_test}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/strict_cross_array_upa8x8_train}

UPA8_TRAIN="$DATA_ROOT/d2los_100k_upa8x8_nf128_los50k_nlos50k_train.pt"
UPA8_TEST="$DATA_ROOT/d2los_100k_upa8x8_nf128_los50k_nlos50k_test.pt"
UPA4_TEST="$DATA_ROOT/d2los_100k_upa4x4_nf128_los50k_nlos50k_test.pt"
ULA64_TEST="$DATA_ROOT/d2los_100k_ula64_nf128_los50k_nlos50k_test.pt"
UPA16_TEST="$UPA16_ROOT/upa16x4_paired.pt"

for path in "$UPA8_TRAIN" "$UPA8_TEST" "$UPA4_TEST" "$ULA64_TEST" "$UPA16_TEST"; do
  if [[ ! -f "$path" ]]; then
    echo "required_data_not_found=$path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
python scripts/validate_cross_array_splits.py \
  --train "train_upa8x8=$UPA8_TRAIN" \
  --test "seen_upa8x8=$UPA8_TEST" \
  --test "unseen_upa4x4=$UPA4_TEST" \
  --test "unseen_upa16x4=$UPA16_TEST" \
  --test "unseen_ula64=$ULA64_TEST" \
  --seen-test seen_upa8x8 \
  --require-aligned-tests \
  --output-json "$OUTPUT_ROOT/split_validation.json" \
  > "$OUTPUT_ROOT/split_validation.txt"

evaluate_one() {
  local checkpoint=$1
  local data_path=$2
  local output_dir=$3
  local text_dir="$output_dir/text_metrics"
  mkdir -p "$output_dir" "$text_dir"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate.py \
    --data-path "$data_path" \
    --checkpoint "$checkpoint" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --verbose-diagnostics \
    --signal-description-correction relational \
    --save-signal-descriptions "$output_dir/signal_descriptions.pt" \
    > "$output_dir/evaluate_output.txt" 2>&1

  python scripts/evaluate_signal_descriptions.py \
    --payload "$output_dir/signal_descriptions.pt" \
    --output-dir "$text_dir" \
    > "$text_dir/evaluate_text_output.txt" 2>&1
}

for seed in 0 1 2; do
  seed_dir="$OUTPUT_ROOT/seed_${seed}"
  mkdir -p "$seed_dir/evaluation"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/pretrain.py \
    --config configs/train.yaml \
    --data-path "$UPA8_TRAIN" \
    --output-dir "$seed_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --save-every "$EPOCHS" \
    --seed "$seed" \
    --use-continuous-config-encoding \
    --use-array-invariant-delay-encoder \
    --array-token-dropout 0.75 \
    --array-token-min-tokens 4 \
    --reflection-path-count-regression-weight 0.01 \
    > "$seed_dir/train_output.txt"

  checkpoint="$seed_dir/checkpoint_epoch_${EPOCHS}.pt"
  evaluate_one "$checkpoint" "$UPA8_TEST" "$seed_dir/evaluation/seen_upa8x8"
  evaluate_one "$checkpoint" "$UPA4_TEST" "$seed_dir/evaluation/unseen_upa4x4"
  evaluate_one "$checkpoint" "$UPA16_TEST" "$seed_dir/evaluation/unseen_upa16x4"
  evaluate_one "$checkpoint" "$ULA64_TEST" "$seed_dir/evaluation/unseen_ula64"
done

python scripts/summarize_cross_array_results.py \
  --root "$OUTPUT_ROOT" \
  --config seen_upa8x8=seen_reference \
  --config unseen_upa4x4=unseen_array \
  --config unseen_upa16x4=unseen_array \
  --config unseen_ula64=unseen_array \
  --output-prefix "$OUTPUT_ROOT/cross_array_3seed_summary"

echo "training_arrays=UPA-8x8"
echo "unseen_test_arrays=UPA-4x4,UPA-16x4,ULA-64"
echo "saved_cross_array_results=$OUTPUT_ROOT"
