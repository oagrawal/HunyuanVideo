#!/bin/bash

# Configuration
CONTAINER="hunyuanvideo"
SCRIPT_PATH="/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache/batch_generate.py"
MODES="easycache_fixed_0.0375,easycache_fixed_0.075,easycache_adaptive_0.025_0.075,easycache_adaptive_0.0375_0.050"

# GPU 0: 0-9
docker exec -e CUDA_VISIBLE_DEVICES=0 $CONTAINER python3 $SCRIPT_PATH \
    --start-idx 0 --end-idx 9 --modes $MODES &

# GPU 1: 9-18
docker exec -e CUDA_VISIBLE_DEVICES=1 $CONTAINER python3 $SCRIPT_PATH \
    --start-idx 9 --end-idx 18 --modes $MODES &

# GPU 2: 18-27
docker exec -e CUDA_VISIBLE_DEVICES=2 $CONTAINER python3 $SCRIPT_PATH \
    --start-idx 18 --end-idx 27 --modes $MODES &

# GPU 3: 27-33
docker exec -e CUDA_VISIBLE_DEVICES=3 $CONTAINER python3 $SCRIPT_PATH \
    --start-idx 27 --end-idx 33 --modes $MODES &

wait
echo "All 4 GPUs have finished generating videos for the requested modes."
