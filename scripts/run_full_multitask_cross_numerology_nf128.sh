#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_ID=${GPU_ID:-1}
EPOCHS=${EPOCHS:-30}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
DATA_DIR=${DATA_DIR:-artifacts/d2los_80k_multiconfig_fit70k_val10k}
FINAL_DATA_DIR=${FINAL_DATA_DIR:-artifacts/d2los_6k_multiconfig_final_holdout_seed45678}
OUTPUT_ROOT=${OUTPUT_ROOT:-artifacts/full_multitask_cross_numerology_nf128_3seed}

NF64_TRAIN="$DATA_DIR/d2los_100k_upa8x8_nf64_los50k_nlos50k_fit.pt"
NF96_TRAIN="$DATA_DIR/d2los_100k_upa8x8_nf96_los50k_nlos50k_fit.pt"
NF192_TRAIN="$DATA_DIR/d2los_100k_upa8x8_nf192_los50k_nlos50k_fit.pt"
NF256_TRAIN="$DATA_DIR/d2los_100k_upa8x8_nf256_los50k_nlos50k_fit.pt"
NF64_TEST="$FINAL_DATA_DIR/d2los_6k_upa8x8_nf64_final_test.pt"
NF96_TEST="$FINAL_DATA_DIR/d2los_6k_upa8x8_nf96_final_test.pt"
NF128_TEST="$FINAL_DATA_DIR/d2los_6k_upa8x8_nf128_final_test.pt"
NF192_TEST="$FINAL_DATA_DIR/d2los_6k_upa8x8_nf192_final_test.pt"
NF256_TEST="$FINAL_DATA_DIR/d2los_6k_upa8x8_nf256_final_test.pt"

for path in \
  "$NF64_TRAIN" \
  "$NF96_TRAIN" \
  "$NF192_TRAIN" \
  "$NF256_TRAIN" \
  "$NF64_TEST" \
  "$NF96_TEST" \
  "$NF128_TEST" \
  "$NF192_TEST" \
  "$NF256_TEST"; do
  if [[ ! -f "$path" ]]; then
    echo "required_data_not_found=$path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
python scripts/validate_cross_numerology_splits.py \
  --train "nf64=$NF64_TRAIN" \
  --train "nf96=$NF96_TRAIN" \
  --train "nf192=$NF192_TRAIN" \
  --train "nf256=$NF256_TRAIN" \
  --test "nf64=$NF64_TEST" \
  --test "nf96=$NF96_TEST" \
  --test "nf128=$NF128_TEST" \
  --test "nf192=$NF192_TEST" \
  --test "nf256=$NF256_TEST" \
  --held-out-nf 128 \
  --require-aligned-training \
  --require-aligned-tests \
  --require-reflection-path-labels \
  --output-json "$OUTPUT_ROOT/split_validation.json" \
  > "$OUTPUT_ROOT/split_validation.txt"

for seed in 0 1 2; do
  seed_dir="$OUTPUT_ROOT/seed_${seed}"
  evaluation_root="$seed_dir/evaluation"
  mkdir -p "$evaluation_root"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/pretrain.py \
    --config configs/train.yaml \
    --data-path "$NF64_TRAIN" \
    --additional-data-path "$NF96_TRAIN" \
    --additional-data-path "$NF192_TRAIN" \
    --additional-data-path "$NF256_TRAIN" \
    --output-dir "$seed_dir" \
    --epochs "$EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --save-every "$EPOCHS" \
    --seed "$seed" \
    --use-continuous-config-encoding \
    --use-array-invariant-delay-encoder \
    --reflection-path-count-regression-weight 0.01 \
    > "$seed_dir/train_output.txt"

  checkpoint="$seed_dir/checkpoint_epoch_${EPOCHS}.pt"
  configurations=(seen_nf64 seen_nf96 unseen_nf128 seen_nf192 seen_nf256)
  test_paths=("$NF64_TEST" "$NF96_TEST" "$NF128_TEST" "$NF192_TEST" "$NF256_TEST")
  for index in "${!configurations[@]}"; do
    configuration="${configurations[$index]}"
    test_path="${test_paths[$index]}"
    evaluation_dir="$evaluation_root/$configuration"
    text_dir="$evaluation_dir/text_metrics"
    mkdir -p "$evaluation_dir" "$text_dir"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate.py \
      --data-path "$test_path" \
      --checkpoint "$checkpoint" \
      --batch-size "$EVAL_BATCH_SIZE" \
      --verbose-diagnostics \
      --signal-description-correction relational \
      --save-signal-descriptions "$evaluation_dir/signal_descriptions.pt" \
      > "$evaluation_dir/evaluate_output.txt"

    python scripts/evaluate_signal_descriptions.py \
      --payload "$evaluation_dir/signal_descriptions.pt" \
      --output-dir "$text_dir" \
      > "$text_dir/evaluate_text_output.txt"
  done
done

python scripts/summarize_full_multitask_numerology.py \
  --root "$OUTPUT_ROOT" \
  --seeds 0 1 2 \
  --configuration seen_nf64=seen_numerology \
  --configuration seen_nf96=seen_numerology \
  --configuration unseen_nf128=unseen_numerology \
  --configuration seen_nf192=seen_numerology \
  --configuration seen_nf256=seen_numerology \
  --output-prefix "$OUTPUT_ROOT/full_multitask_nf128_3seed_summary"

python scripts/summarize_signal_description_text_metrics.py \
  --input "seed_0=$OUTPUT_ROOT/seed_0/evaluation/unseen_nf128/text_metrics/signal_description_text_metrics.csv" \
  --input "seed_1=$OUTPUT_ROOT/seed_1/evaluation/unseen_nf128/text_metrics/signal_description_text_metrics.csv" \
  --input "seed_2=$OUTPUT_ROOT/seed_2/evaluation/unseen_nf128/text_metrics/signal_description_text_metrics.csv" \
  --output-csv "$OUTPUT_ROOT/unseen_nf128_text_metrics_3seed_summary.csv" \
  --output-json "$OUTPUT_ROOT/unseen_nf128_text_metrics_3seed_summary.json"

echo "training_numerologies=nf64,nf96,nf192,nf256"
echo "held_out_numerology=nf128"
echo "seen_final_references=nf64,nf96,nf192,nf256"
echo "model=full_multitask_physics_and_text"
echo "saved_results=$OUTPUT_ROOT"
