#!/usr/bin/env bash
# Run DiCache baseline + 3 adaptive modes for one prompt slice on one GPU.
# Usage: run_dicache_adaptive_one_gpu.sh <GPU_ID> <start-idx> <end-idx> [log-file]
set -euo pipefail

GPU_ID="${1:?need GPU id}"
START="${2:?need start-idx}"
END="${3:?need end-idx}"
LOG="${4:-}"

cd /nfs/oagrawal/HunyuanVideo
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ -n "${LOG}" ]]; then
  exec > >(tee -a "${LOG}") 2>&1
fi

COMMON="--prompts-file vbench_eval/prompts_subset.json
  --output-dir dicache_results/videos
  --results-dir dicache_results/results
  --generation-seed 0
  --start-idx ${START}
  --end-idx ${END}
  --video-size 544 960
  --video-length 129
  --infer-steps 50
  --flow-reverse"

echo "======================================================================"
echo "DiCache adaptive gen | GPU=${GPU_ID} | prompts [${START}, ${END})"
echo "======================================================================"

echo ""
echo "---- dicache_baseline (delta=0, no skipping) ----"
python3 dicache_results/batch_generate_fixed.py ${COMMON} \
  --mode-name dicache_baseline --delta 0

echo ""
echo "---- dicache_adaptive_0.05_0.20 (LOW=0.05, HIGH=0.20) ----"
python3 dicache_results/batch_generate_fixed.py ${COMMON} \
  --mode-name dicache_adaptive_0.05_0.20 \
  --delta 0.05 --delta-low 0.05 --delta-high 0.20 \
  --stable-start 8 --stable-end 40

echo ""
echo "---- dicache_adaptive_0.10_0.30 (LOW=0.10, HIGH=0.30) ----"
python3 dicache_results/batch_generate_fixed.py ${COMMON} \
  --mode-name dicache_adaptive_0.10_0.30 \
  --delta 0.10 --delta-low 0.10 --delta-high 0.30 \
  --stable-start 8 --stable-end 40

echo ""
echo "---- dicache_adaptive_0.15_0.40 (LOW=0.15, HIGH=0.40) ----"
python3 dicache_results/batch_generate_fixed.py ${COMMON} \
  --mode-name dicache_adaptive_0.15_0.40 \
  --delta 0.15 --delta-low 0.15 --delta-high 0.40 \
  --stable-start 8 --stable-end 40

echo ""
echo "======================================================================"
echo "Done GPU=${GPU_ID} prompts [${START}, ${END})"
echo "======================================================================"
