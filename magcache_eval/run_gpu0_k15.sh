#!/bin/bash
# K=15, retention_ratio=0.1 runs — GPU 0, prompts 0-9
# 3 fixed thresholds: low (0.06), medium (0.24), max (0.80)
cd /nfs/oagrawal/HunyuanVideo
CUDA_VISIBLE_DEVICES=0 python3 magcache_eval/batch_generate.py \
  --video-size 544 960 --video-length 129 --infer-steps 50 \
  --model-base ckpts --flow-reverse --use-cpu-offload --flow-shift 7.0 --cfg-scale 1.0 --embedded-cfg-scale 6.0 \
  --prompts-file vbench_eval/prompts_subset.json \
  --output-dir magcache_eval/videos \
  --generation-seed 0 \
  --modes magcache_k15_fixed_0.06,magcache_k15_fixed_0.24,magcache_k15_fixed_0.80 \
  --start-idx 0 --end-idx 9 \
  2>&1 | tee magcache_eval/logs/gpu0_k15.log
