#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_PATH=${DATA_PATH:-"$ROOT/artifacts/d2los_50k_upa8x8_nf128_pathcount_seed23421/d2los_50k_upa8x8_nf128_pathcount_train.pt"}
BASE_CONFIG=${BASE_CONFIG:-"$ROOT/configs/train.yaml"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT/artifacts/full_multitask_d2los_50k_pathcount_noise_augmented_k_supervised_3seed"}
SEEDS=${SEEDS:-"0"}
BATCH_SIZE=${BATCH_SIZE:-64}

for seed in $SEEDS; do
  output_dir="$OUTPUT_ROOT/seed_$seed"
  mkdir -p "$output_dir"
  python "$ROOT/scripts/pretrain.py" \
    --config "$BASE_CONFIG" \
    --data-path "$DATA_PATH" \
    --seed "$seed" \
    --batch-size "$BATCH_SIZE" \
    --noise-augmentation \
    --noise-augmentation-probability 0.8 \
    --noise-snr-min-db 10 \
    --noise-snr-max-db 30 \
    --noise-augmented-main-weight 0.5 \
    --noise-delay-consistency-weight 0.1 \
    --noise-k-consistency-weight 0.1 \
    --noise-clean-k-supervised-weight 0.2 \
    --noise-noisy-k-supervised-weight 0.1 \
    --output-dir "$output_dir" \
    2>&1 | tee "$output_dir/train_runner.log"
done
