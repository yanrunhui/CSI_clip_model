#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-artifacts/d2los_80k_multiconfig_fit70k_val10k}
FINAL_DATA_DIR=${FINAL_DATA_DIR:-artifacts/d2los_6k_multiconfig_final_holdout_seed45678}
TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT:-artifacts/dev_both_unseen_nf96_nf128_expert_gate}
FINAL_OUTPUT_ROOT=${FINAL_OUTPUT_ROOT:-artifacts/final_test_both_unseen_nf96_nf128_seed45678}
CUDA_DEVICE=${CUDA_DEVICE:-2}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-128}

mkdir -p "$TRAIN_OUTPUT_ROOT" "$FINAL_OUTPUT_ROOT"

for seed in 0 1 2; do
  train_output="$TRAIN_OUTPUT_ROOT/seed_${seed}"
  final_output="$FINAL_OUTPUT_ROOT/seed_${seed}"
  mkdir -p "$train_output" "$final_output"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/train_multinumerology_generalization.py \
    --train-pair \
      nf64 640 "$DATA_DIR/d2los_100k_upa8x8_nf64_los50k_nlos50k_fit.pt" \
      nf192 1920 "$DATA_DIR/d2los_100k_upa8x8_nf192_los50k_nlos50k_fit.pt" \
    --train-pair \
      nf64 640 "$DATA_DIR/d2los_100k_upa8x8_nf64_los50k_nlos50k_fit.pt" \
      nf256 2560 "$DATA_DIR/d2los_100k_upa8x8_nf256_los50k_nlos50k_fit.pt" \
    --train-pair \
      nf192 1920 "$DATA_DIR/d2los_100k_upa8x8_nf192_los50k_nlos50k_fit.pt" \
      nf256 2560 "$DATA_DIR/d2los_100k_upa8x8_nf256_los50k_nlos50k_fit.pt" \
    --test-pair \
      nf96 960 "$DATA_DIR/d2los_100k_upa8x8_nf96_los50k_nlos50k_val.pt" \
      nf128 1280 "$DATA_DIR/d2los_100k_upa8x8_nf128_los50k_nlos50k_val.pt" \
    --held-out-name nf128 \
    --max-delay-ns 1920 \
    --max-delay-spread-ns 400 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --single-weight 1.0 \
    --residue-weight 0.25 \
    --consistency-weight 0.1 \
    --direct-weight 0.25 \
    --uncertainty-weight 0.05 \
    --uncertainty-ranking-weight 0 \
    --gate-weight 0.1 \
    --period-prior-strength 0 \
    --fallback-confidence-threshold 0.5 \
    --regret-weight 0.25 \
    --residual-weight 0.01 \
    --residual-scale-ns 50 \
    --view-dropout 0.1 \
    --seed "$seed" \
    --output-dir "$train_output" \
    > "$train_output/train_validation_output.txt"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python scripts/evaluate_frozen_multinumerology_holdout.py \
    --checkpoint "$train_output/multinumerology_generalization.pt" \
    --test-pair \
      nf96 960 "$FINAL_DATA_DIR/d2los_6k_upa8x8_nf96_final_test.pt" \
      nf128 1280 "$FINAL_DATA_DIR/d2los_6k_upa8x8_nf128_final_test.pt" \
    --held-out-name nf128 \
    --batch-size "$BATCH_SIZE" \
    --report-frozen-baselines \
    --output-dir "$final_output" \
    > "$final_output/evaluate_output.txt"
done

python scripts/summarize_frozen_multinumerology_baselines.py \
  --root "$TRAIN_OUTPUT_ROOT" \
  --metrics-name multinumerology_generalization_test_metrics.csv \
  --output-prefix "$TRAIN_OUTPUT_ROOT/validation_3seed_summary"

python scripts/summarize_frozen_multinumerology_baselines.py \
  --root "$FINAL_OUTPUT_ROOT" \
  --metrics-name final_test_metrics.csv \
  --output-prefix "$FINAL_OUTPUT_ROOT/final_test_3seed_summary"

echo "training_numerologies=nf64,nf192,nf256"
echo "both_unseen_test_numerologies=nf96,nf128"
echo "final_output_method=paired_gate_ab_hard"
echo "saved_validation_results=$TRAIN_OUTPUT_ROOT"
echo "saved_final_test_results=$FINAL_OUTPUT_ROOT"
