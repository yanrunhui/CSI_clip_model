#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
LIMIT="${LIMIT:-1000}"
WARMUP="${WARMUP:-50}"
REPEATS="${REPEATS:-3}"
QUALITY_SAMPLE_COUNT="${QUALITY_SAMPLE_COUNT:-$LIMIT}"
REUSE_QUALITY_METRICS="${REUSE_QUALITY_METRICS:-0}"
INTERNLM_QUALITY_METRICS="${INTERNLM_QUALITY_METRICS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAX_INITIAL_GPU_MEMORY_MIB="${MAX_INITIAL_GPU_MEMORY_MIB:-1024}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-artifacts/inference_cost_benchmark}"
TEST="${TEST:-artifacts/d2los_100k_multiconfig_aligned/d2los_100k_upa8x8_nf128_los50k_nlos50k_test.pt}"
MODEL="${MODEL:-models/InternLM3-8B-Instruct}"
RUN="${RUN:-artifacts/internlm3_8b_csi_prefix/seed_0}"
OUTPUT="$BENCHMARK_ROOT/internlm3_8b/seed_0"

if [[ "$REUSE_QUALITY_METRICS" == 1 ]]; then
  test -f "$INTERNLM_QUALITY_METRICS" || {
    echo "missing reused InternLM3 quality metrics: $INTERNLM_QUALITY_METRICS" >&2
    exit 1
  }
fi

for path in \
  "$TEST" \
  "$MODEL/config.json" \
  "$RUN/mapping_final.pt" \
  "$RUN/internlm_adapter/adapter_config.json"; do
  test -f "$path" || { echo "missing InternLM3 benchmark input: $path" >&2; exit 1; }
done
if [[ "$SKIP_EXISTING" == 1 && -f "$OUTPUT/BENCHMARK_COMPLETE" ]]; then
  echo "internlm3_benchmark_already_complete=$OUTPUT"
  exit 0
fi

initial_gpu_memory_mib="$(
  nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits \
    | head -n 1 | tr -d ' '
)"
if [[ "$ALLOW_BUSY_GPU" != 1 ]] && (( initial_gpu_memory_mib > MAX_INITIAL_GPU_MEMORY_MIB )); then
  echo "GPU $GPU is using ${initial_gpu_memory_mib} MiB; choose an idle GPU." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$BENCHMARK_ROOT/logs" "$OUTPUT/text_metrics"
rm -f "$OUTPUT/BENCHMARK_COMPLETE"
TEST_SHA256="${TEST_SHA256:-$(sha256sum "$TEST" | awk '{print $1}')}"
echo "internlm3_benchmark_test_sha256=$TEST_SHA256"
echo "cost_sample_count=$LIMIT"
echo "quality_sample_count=$QUALITY_SAMPLE_COUNT"
echo "reuse_quality_metrics=$REUSE_QUALITY_METRICS"

python scripts/benchmark_internlm3_csi_prefix.py \
  --model-name "InternLM3-8B-Instruct" \
  --model-path "$MODEL" \
  --mapping-checkpoint "$RUN/mapping_final.pt" \
  --adapter-path "$RUN/internlm_adapter" \
  --data-path "$TEST" \
  --data-sha256 "$TEST_SHA256" \
  --output-dir "$OUTPUT" \
  --load-in-4bit \
  --compute-dtype bfloat16 \
  --limit "$LIMIT" \
  --warmup-samples "$WARMUP" \
  --repeats "$REPEATS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed 0 \
  2>&1 | tee "$BENCHMARK_ROOT/logs/internlm3_8b.log"

if [[ "$REUSE_QUALITY_METRICS" == 1 ]]; then
  python scripts/reuse_inference_quality_metrics.py \
    --source "$INTERNLM_QUALITY_METRICS" \
    --destination "$OUTPUT/text_metrics/signal_description_text_metrics.csv" \
    --metric-format text \
    --expected-samples "$QUALITY_SAMPLE_COUNT" \
    --test-data-sha256 "$TEST_SHA256" \
    --provenance-output "$OUTPUT/quality_provenance.json"
else
  python scripts/build_qwen_signal_description_payload.py \
    --data-jsonl "$OUTPUT/targets.jsonl" \
    --predictions-jsonl "$OUTPUT/predictions.jsonl" \
    --output "$OUTPUT/signal_descriptions.pt" \
    2>&1 | tee "$BENCHMARK_ROOT/logs/internlm3_8b_payload.log"

  python scripts/evaluate_signal_descriptions.py \
    --payload "$OUTPUT/signal_descriptions.pt" \
    --output-dir "$OUTPUT/text_metrics" \
    2>&1 | tee "$BENCHMARK_ROOT/logs/internlm3_8b_text.log"
fi

touch "$OUTPUT/BENCHMARK_COMPLETE"
echo "internlm3_inference_cost_benchmark_complete=$OUTPUT"
