#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-2}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"

UPA8_TRAIN="artifacts/d2los_100k_upa8x8_nf128_los50k_nlos50k_train.pt"
UPA4_TRAIN="artifacts/d2los_100k_upa4x4_100mhz_los_nlos_balanced_train.pt"
ULA64_TRAIN="artifacts/d2los_100k_ula64_100mhz_los_nlos_balanced_train.pt"

UPA8_TEST="artifacts/d2los_100k_upa8x8_nf128_los50k_nlos50k_test.pt"
UPA4_TEST="artifacts/d2los_100k_upa4x4_100mhz_los_nlos_balanced_test.pt"
ULA64_TEST="artifacts/d2los_100k_ula64_100mhz_los_nlos_balanced_test.pt"

OUTPUT_ROOT="artifacts/array_generalization_joint"
TRAIN_OUTPUT="${OUTPUT_ROOT}/seed_${SEED}"
PAIRED_ROOT="${OUTPUT_ROOT}/paired_tests"
EVAL_ROOT="${TRAIN_OUTPUT}/evaluation"

mkdir -p "${TRAIN_OUTPUT}" "${PAIRED_ROOT}" "${EVAL_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/pretrain.py \
  --config configs/train.yaml \
  --data-path "${UPA8_TRAIN}" \
  --additional-data-path "${UPA4_TRAIN}" \
  --additional-data-path "${ULA64_TRAIN}" \
  --output-dir "${TRAIN_OUTPUT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --save-every 20 \
  --seed "${SEED}" \
  --use-continuous-config-encoding \
  --use-array-invariant-delay-encoder \
  --array-token-dropout 0.75 \
  --array-token-min-tokens 4 \
  2>&1 | tee "${TRAIN_OUTPUT}/train_output.txt"

CHECKPOINT="${TRAIN_OUTPUT}/checkpoint_epoch_${EPOCHS}.pt"

python scripts/make_paired_configuration_subset.py \
  --input "upa8x8=${UPA8_TEST}" \
  --input "upa4x4=${UPA4_TEST}" \
  --output-dir "${PAIRED_ROOT}/upa8x8_vs_upa4x4"

python scripts/make_paired_configuration_subset.py \
  --input "upa8x8=${UPA8_TEST}" \
  --input "ula64=${ULA64_TEST}" \
  --output-dir "${PAIRED_ROOT}/upa8x8_vs_ula64"

evaluate_config() {
  local name="$1"
  local data_path="$2"
  local output_dir="${EVAL_ROOT}/${name}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/evaluate.py \
    --data-path "${data_path}" \
    --checkpoint "${CHECKPOINT}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --verbose-diagnostics \
    --signal-description-correction relational \
    --save-signal-descriptions "${output_dir}/signal_descriptions.pt" \
    > "${output_dir}/evaluate_output.txt"
  python scripts/evaluate_signal_descriptions.py \
    --payload "${output_dir}/signal_descriptions.pt" \
    --output-dir "${output_dir}"
}

evaluate_config "full_upa8x8" "${UPA8_TEST}"
evaluate_config "full_upa4x4" "${UPA4_TEST}"
evaluate_config "full_ula64" "${ULA64_TEST}"
evaluate_config \
  "paired_upa8x8_for_upa4x4" \
  "${PAIRED_ROOT}/upa8x8_vs_upa4x4/upa8x8_paired.pt"
evaluate_config \
  "paired_upa4x4" \
  "${PAIRED_ROOT}/upa8x8_vs_upa4x4/upa4x4_paired.pt"
evaluate_config \
  "paired_upa8x8_for_ula64" \
  "${PAIRED_ROOT}/upa8x8_vs_ula64/upa8x8_paired.pt"
evaluate_config \
  "paired_ula64" \
  "${PAIRED_ROOT}/upa8x8_vs_ula64/ula64_paired.pt"

grep -H -E \
  '^(first_path_delay_context_MAE|los_delay_context_MAE|first_path_angle_los_MAE|first_path_angle_nlos_MAE|k_factor_db_MAE|reflection_count_head_accuracy)=' \
  "${EVAL_ROOT}"/*/evaluate_output.txt \
  | tee "${TRAIN_OUTPUT}/generalization_key_metrics.txt"
