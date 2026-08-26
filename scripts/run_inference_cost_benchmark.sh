#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
SEEDS="${SEEDS:-0 1 2}"
LIMIT="${LIMIT:-1000}"
WARMUP="${WARMUP:-50}"
REPEATS="${REPEATS:-3}"
QUALITY_SAMPLE_COUNT="${QUALITY_SAMPLE_COUNT:-$LIMIT}"
REUSE_QUALITY_METRICS="${REUSE_QUALITY_METRICS:-0}"
QUALITY_ROOT="${QUALITY_ROOT:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
OUT="${OUT:-artifacts/inference_cost_benchmark}"
TEST="${TEST:-artifacts/d2los_100k_multiconfig_aligned/d2los_100k_upa8x8_nf128_los50k_nlos50k_test.pt}"
BASELINE_ROOT="${BASELINE_ROOT:-artifacts/baselines_d2los_100k_upa8x8_nf128_los50k_nlos50k_fair_100ep}"
FULL_ROOT="${FULL_ROOT:-artifacts/pretrain_d2los_100k_upa8x8_nf128_los50k_nlos50k_reflection_path_3seed}"
QWEN_MODEL="${QWEN_MODEL:-artifacts/hf/Qwen3-1.7B}"
QWEN_RUN="${QWEN_RUN:-artifacts/qwen3_1p7b_explicit_csi_prefix/seed_0_gpu0}"
QWEN35_MODEL="${QWEN35_MODEL:-models/Qwen3.5-2B}"
QWEN35_RUN="${QWEN35_RUN:-artifacts/qwen3_5_2b_explicit_csi_prefix/seed_0}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-models/DeepSeek-R1-0528-Qwen3-8B}"
DEEPSEEK_RUN="${DEEPSEEK_RUN:-artifacts/deepseek_r1_qwen3_8b_csi_prefix/seed_0_bf16_json_prefill_500}"
QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-512}"
QWEN35_MAX_NEW_TOKENS="${QWEN35_MAX_NEW_TOKENS:-512}"
DEEPSEEK_MAX_NEW_TOKENS="${DEEPSEEK_MAX_NEW_TOKENS:-512}"
MAX_INITIAL_GPU_MEMORY_MIB="${MAX_INITIAL_GPU_MEMORY_MIB:-1024}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"

if [[ "$REUSE_QUALITY_METRICS" == 1 ]]; then
  test -n "$QUALITY_ROOT" || {
    echo "QUALITY_ROOT is required when REUSE_QUALITY_METRICS=1" >&2
    exit 1
  }
  QWEN_QUALITY_METRICS="${QWEN_QUALITY_METRICS:-$QUALITY_ROOT/qwen3_1p7b/seed_0/text_metrics/signal_description_text_metrics.csv}"
  QWEN35_QUALITY_METRICS="${QWEN35_QUALITY_METRICS:-$QUALITY_ROOT/qwen3_5_2b/seed_0/text_metrics/signal_description_text_metrics.csv}"
  DEEPSEEK_QUALITY_METRICS="${DEEPSEEK_QUALITY_METRICS:-$QUALITY_ROOT/deepseek_8b/seed_0/text_metrics/signal_description_text_metrics.csv}"
fi

full_quality_metrics_path() {
  local seed="$1"
  local variable="FULL_QUALITY_METRICS_SEED_${seed}"
  local override="${!variable:-}"
  if [[ -n "$override" ]]; then
    printf '%s\n' "$override"
  else
    printf '%s\n' \
      "$QUALITY_ROOT/full_multitask/seed_${seed}/text_metrics/signal_description_text_metrics.csv"
  fi
}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUT/logs"
rm -f "$OUT/BENCHMARK_COMPLETE"
test -f "$TEST" || { echo "missing test dataset: $TEST" >&2; exit 1; }
initial_gpu_memory_mib="$(
  nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits \
    | head -n 1 | tr -d ' '
)"
if [[ "$ALLOW_BUSY_GPU" != 1 ]] && (( initial_gpu_memory_mib > MAX_INITIAL_GPU_MEMORY_MIB )); then
  echo "GPU $GPU is already using ${initial_gpu_memory_mib} MiB; " \
       "choose an idle GPU or explicitly set ALLOW_BUSY_GPU=1." >&2
  exit 1
fi
echo "initial_gpu_memory_mib=$initial_gpu_memory_mib"
TEST_SHA256="${TEST_SHA256:-$(sha256sum "$TEST" | awk '{print $1}')}"
echo "test_data_sha256=$TEST_SHA256"
echo "cost_sample_count=$LIMIT"
echo "cost_repeat_count=$REPEATS"
echo "quality_sample_count=$QUALITY_SAMPLE_COUNT"
echo "reuse_quality_metrics=$REUSE_QUALITY_METRICS"

for seed in $SEEDS; do
  find_full_checkpoint_preflight=0
  for candidate in \
    "$FULL_ROOT/seed_${seed}/checkpoint_epoch_100.pt" \
    "$FULL_ROOT/seed_${seed}/checkpoint_last.pt"; do
    [[ -f "$candidate" ]] && find_full_checkpoint_preflight=1 && break
  done
  [[ "$find_full_checkpoint_preflight" == 1 ]] || {
    echo "missing full checkpoint for seed $seed under $FULL_ROOT" >&2
    exit 1
  }
  for model in \
    csi_encoder_single_task pdp_ifft_mlp flattened_mlp \
    transformer_no_branches cnn_baseline; do
    for target in \
      first_path_delay first_path_angle first_path_power k_factor reflection_count; do
      file="$BASELINE_ROOT/seed_${seed}/${model}_${target}.pt"
      test -f "$file" || { echo "missing baseline checkpoint: $file" >&2; exit 1; }
    done
  done
done
if [[ "$REUSE_QUALITY_METRICS" == 1 ]]; then
  for seed in $SEEDS; do
    full_quality_metrics="$(full_quality_metrics_path "$seed")"
    test -f "$full_quality_metrics" || {
      echo "missing reused Full-model quality metrics for seed $seed: $full_quality_metrics" >&2
      exit 1
    }
    for model in \
      csi_encoder_single_task pdp_ifft_mlp flattened_mlp \
      transformer_no_branches cnn_baseline; do
      test -f "$QUALITY_ROOT/$model/seed_${seed}/physics_metrics.csv" || {
        echo "missing reused baseline quality metrics: $QUALITY_ROOT/$model/seed_${seed}/physics_metrics.csv" >&2
        exit 1
      }
    done
  done
  for path in \
    "$QWEN_QUALITY_METRICS" \
    "$QWEN35_QUALITY_METRICS" \
    "$DEEPSEEK_QUALITY_METRICS"; do
    test -f "$path" || { echo "missing reused language-model quality metrics: $path" >&2; exit 1; }
  done
fi
for path in \
  "$QWEN_MODEL/config.json" \
  "$QWEN_RUN/mapping_final.pt" \
  "$QWEN_RUN/qwen_adapter/adapter_config.json" \
  "$QWEN35_MODEL/config.json" \
  "$QWEN35_RUN/mapping_final.pt" \
  "$QWEN35_RUN/qwen_adapter/adapter_config.json" \
  "$DEEPSEEK_MODEL/config.json" \
  "$DEEPSEEK_RUN/mapping_final.pt" \
  "$DEEPSEEK_RUN/qwen_adapter/adapter_config.json"; do
  test -f "$path" || { echo "missing benchmark input: $path" >&2; exit 1; }
done
echo "benchmark_preflight=passed"

run_logged() {
  local log="$1"
  shift
  echo "=== $* ===" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
}

reuse_quality_metrics() {
  local source="$1"
  local destination="$2"
  local metric_format="$3"
  local provenance_output="$4"
  python scripts/reuse_inference_quality_metrics.py \
    --source "$source" \
    --destination "$destination" \
    --metric-format "$metric_format" \
    --expected-samples "$QUALITY_SAMPLE_COUNT" \
    --test-data-sha256 "$TEST_SHA256" \
    --provenance-output "$provenance_output"
}

find_full_checkpoint() {
  local seed="$1"
  local candidate
  for candidate in \
    "$FULL_ROOT/seed_${seed}/checkpoint_epoch_100.pt" \
    "$FULL_ROOT/seed_${seed}/checkpoint_last.pt"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "missing full checkpoint for seed $seed under $FULL_ROOT" >&2
  return 1
}

for seed in $SEEDS; do
  full_out="$OUT/full_multitask/seed_${seed}"
  if [[ "$SKIP_EXISTING" != 1 || ! -f "$full_out/BENCHMARK_COMPLETE" ]]; then
    rm -f "$full_out/BENCHMARK_COMPLETE"
    mkdir -p "$full_out/text_metrics"
    checkpoint="$(find_full_checkpoint "$seed")"
    run_logged "$OUT/logs/full_seed_${seed}.log" \
      python scripts/benchmark_full_model.py \
        --data-path "$TEST" \
        --data-sha256 "$TEST_SHA256" \
        --checkpoint "$checkpoint" \
        --output-dir "$full_out" \
        --limit "$LIMIT" \
        --warmup-samples "$WARMUP" \
        --repeats "$REPEATS" \
        --seed "$seed" \
        --signal-description-correction relational
    if [[ "$REUSE_QUALITY_METRICS" == 1 ]]; then
      full_quality_metrics="$(full_quality_metrics_path "$seed")"
      reuse_quality_metrics \
        "$full_quality_metrics" \
        "$full_out/text_metrics/signal_description_text_metrics.csv" \
        text \
        "$full_out/quality_provenance.json"
    else
      run_logged "$OUT/logs/full_seed_${seed}_text.log" \
        python scripts/evaluate_signal_descriptions.py \
          --payload "$full_out/signal_descriptions.pt" \
          --output-dir "$full_out/text_metrics"
    fi
    touch "$full_out/BENCHMARK_COMPLETE"
  fi

  for model in \
    csi_encoder_single_task \
    pdp_ifft_mlp \
    flattened_mlp \
    transformer_no_branches \
    cnn_baseline; do
    baseline_out="$OUT/$model/seed_${seed}"
    checkpoint_dir="$BASELINE_ROOT/seed_${seed}"
    if [[ "$SKIP_EXISTING" != 1 || ! -f "$baseline_out/BENCHMARK_COMPLETE" ]]; then
      rm -f "$baseline_out/BENCHMARK_COMPLETE"
      run_logged "$OUT/logs/${model}_seed_${seed}.log" \
        python scripts/benchmark_physics_baselines.py \
          --data-path "$TEST" \
          --data-sha256 "$TEST_SHA256" \
          --checkpoint-dir "$checkpoint_dir" \
          --model-name "$model" \
          --output-dir "$baseline_out" \
          --limit "$LIMIT" \
          --warmup-samples "$WARMUP" \
          --repeats "$REPEATS" \
          --seed "$seed"
      if [[ "$REUSE_QUALITY_METRICS" == 1 ]]; then
        reuse_quality_metrics \
          "$QUALITY_ROOT/$model/seed_${seed}/physics_metrics.csv" \
          "$baseline_out/physics_metrics.csv" \
          baseline \
          "$baseline_out/quality_provenance.json"
      fi
      touch "$baseline_out/BENCHMARK_COMPLETE"
    fi
  done
done

run_lm_benchmark() {
  local directory="$1"
  local display_name="$2"
  local model_path="$3"
  local run_path="$4"
  local max_new_tokens="$5"
  local quality_metrics="$6"
  local lm_out="$OUT/$directory/seed_0"
  if [[ "$SKIP_EXISTING" == 1 && -f "$lm_out/BENCHMARK_COMPLETE" ]]; then
    return
  fi
  rm -f "$lm_out/BENCHMARK_COMPLETE"
  test -f "$run_path/mapping_final.pt" || {
    echo "missing mapping checkpoint: $run_path/mapping_final.pt" >&2
    exit 1
  }
  test -f "$run_path/qwen_adapter/adapter_config.json" || {
    echo "missing adapter: $run_path/qwen_adapter/adapter_config.json" >&2
    exit 1
  }
  mkdir -p "$lm_out/text_metrics"
  run_logged "$OUT/logs/${directory}.log" \
    python scripts/benchmark_csi_prefix_lm.py \
      --model-name "$display_name" \
      --model-path "$model_path" \
      --mapping-checkpoint "$run_path/mapping_final.pt" \
      --adapter-path "$run_path/qwen_adapter" \
      --data-path "$TEST" \
      --data-sha256 "$TEST_SHA256" \
      --output-dir "$lm_out" \
      --load-in-4bit \
      --compute-dtype bfloat16 \
      --limit "$LIMIT" \
      --warmup-samples "$WARMUP" \
      --repeats "$REPEATS" \
      --max-new-tokens "$max_new_tokens" \
      --seed 0
  if [[ "$REUSE_QUALITY_METRICS" == 1 ]]; then
    reuse_quality_metrics \
      "$quality_metrics" \
      "$lm_out/text_metrics/signal_description_text_metrics.csv" \
      text \
      "$lm_out/quality_provenance.json"
  else
    run_logged "$OUT/logs/${directory}_payload.log" \
      python scripts/build_qwen_signal_description_payload.py \
        --data-jsonl "$lm_out/targets.jsonl" \
        --predictions-jsonl "$lm_out/predictions.jsonl" \
        --output "$lm_out/signal_descriptions.pt"
    run_logged "$OUT/logs/${directory}_text.log" \
      python scripts/evaluate_signal_descriptions.py \
        --payload "$lm_out/signal_descriptions.pt" \
        --output-dir "$lm_out/text_metrics"
  fi
  touch "$lm_out/BENCHMARK_COMPLETE"
}

run_lm_benchmark \
  qwen3_1p7b Qwen3-1.7B "$QWEN_MODEL" "$QWEN_RUN" \
  "$QWEN_MAX_NEW_TOKENS" "${QWEN_QUALITY_METRICS:-}"
run_lm_benchmark \
  qwen3_5_2b \
  "Qwen3.5-2B updated direct decoder" \
  "$QWEN35_MODEL" \
  "$QWEN35_RUN" \
  "$QWEN35_MAX_NEW_TOKENS" \
  "${QWEN35_QUALITY_METRICS:-}"
run_lm_benchmark \
  deepseek_8b DeepSeek-R1-Qwen3-8B "$DEEPSEEK_MODEL" "$DEEPSEEK_RUN" \
  "$DEEPSEEK_MAX_NEW_TOKENS" "${DEEPSEEK_QUALITY_METRICS:-}"

python scripts/build_inference_benchmark_manifest.py \
  --benchmark-root "$OUT" \
  --output "$OUT/benchmark_manifest.json" \
  --quality-samples "$QUALITY_SAMPLE_COUNT" \
  --seeds $SEEDS
python scripts/summarize_inference_cost_benchmark.py \
  --manifest "$OUT/benchmark_manifest.json" \
  --output-dir "$OUT"
python scripts/plot_inference_cost_benchmark.py \
  --input-dir "$OUT" \
  --output "$OUT/inference_cost_panels.png"

validation_args=(
  --benchmark-root "$OUT"
  --expected-samples "$LIMIT"
  --expected-repeats "$REPEATS"
  --expected-quality-samples "$QUALITY_SAMPLE_COUNT"
)
if (( LIMIT <= 10 )); then
  validation_args+=(--require-perfect-parse)
fi
python scripts/validate_inference_cost_benchmark.py "${validation_args[@]}"

cp "$OUT/full_multitask/seed_0/environment.json" "$OUT/environment.json"
touch "$OUT/BENCHMARK_COMPLETE"
echo "benchmark_complete=$OUT"
