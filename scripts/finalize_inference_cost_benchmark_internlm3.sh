#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/inference_cost_benchmark}"
LIMIT="${LIMIT:-1000}"
REPEATS="${REPEATS:-3}"
QUALITY_SAMPLE_COUNT="${QUALITY_SAMPLE_COUNT:-$LIMIT}"
BASE_MANIFEST="${BASE_MANIFEST:-$ROOT/benchmark_manifest.json}"
EXTENDED_MANIFEST="$ROOT/benchmark_manifest_with_internlm3.json"

python scripts/extend_inference_benchmark_manifest_internlm3.py \
  --base-manifest "$BASE_MANIFEST" \
  --benchmark-root "$ROOT" \
  --output "$EXTENDED_MANIFEST"

python scripts/summarize_inference_cost_benchmark.py \
  --manifest "$EXTENDED_MANIFEST" \
  --output-dir "$ROOT"

python scripts/plot_inference_cost_benchmark.py \
  --input-dir "$ROOT" \
  --output "$ROOT/inference_cost_panels_with_internlm3.png"

cp "$EXTENDED_MANIFEST" "$ROOT/benchmark_manifest.json"
python scripts/validate_inference_cost_benchmark.py \
  --benchmark-root "$ROOT" \
  --expected-samples "$LIMIT" \
  --expected-repeats "$REPEATS" \
  --expected-quality-samples "$QUALITY_SAMPLE_COUNT"

touch "$ROOT/BENCHMARK_COMPLETE_WITH_INTERNLM3"
echo "benchmark_with_internlm3_complete=$ROOT"
