#!/usr/bin/env bash
# DiCache fixed thresholds 0.25–0.40; one prompt range per GPU (see run_dicache_prompt_split_one_gpu.sh).
# Usage: run_dicache_prompt_split_0.25_0.40.sh <CUDA_VISIBLE_DEVICES> <start-idx> <end-idx> [log-file]
set -euo pipefail

GPU_ID="${1:?need GPU id (0-3)}"
START="${2:?need start-idx (inclusive)}"
END="${3:?need end-idx (exclusive)}"
LOG="${4:-}"

cd /nfs/oagrawal/HunyuanVideo
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

RUNS=(
  "0.25:dicache_fixed_0.25"
  "0.30:dicache_fixed_0.30"
  "0.35:dicache_fixed_0.35"
  "0.40:dicache_fixed_0.40"
)

if [[ -n "${LOG}" ]]; then
  exec > >(tee -a "${LOG}") 2>&1
fi

echo "======================================================================"
echo "DiCache prompt-split (0.25–0.40) | GPU=${GPU_ID} | prompts [${START}, ${END})"
echo "======================================================================"

for run in "${RUNS[@]}"; do
  delta="${run%%:*}"
  mode="${run#*:}"
  echo ""
  echo "---- ${mode} (delta=${delta}) ----"
  python3 dicache_results/batch_generate_fixed.py \
    --prompts-file vbench_eval/prompts_subset.json \
    --output-dir dicache_results/videos \
    --results-dir dicache_results/results \
    --mode-name "${mode}" \
    --delta "${delta}" \
    --ret-ratio 0.0 \
    --probe-depth 1 \
    --generation-seed 0 \
    --start-idx "${START}" \
    --end-idx "${END}" \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --flow-reverse
done

echo ""
echo "======================================================================"
echo "Done GPU=${GPU_ID} prompts [${START}, ${END})"
echo "======================================================================"
