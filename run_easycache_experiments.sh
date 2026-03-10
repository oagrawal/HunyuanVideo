#!/bin/bash
#
# EasyCache Experiments Runner
# Runs EasyCache (baseline + caching) experiments inside the HunyuanVideo Docker container
#
# ============================================================================
# STEP-BY-STEP DOCKER INSTRUCTIONS
# ============================================================================
#
# 1. One-time setup: create the container (if you haven't already)
#
#    From the host, in the HunyuanVideo folder:
#
#    cd /nfs/oagrawal/HunyuanVideo
#
#    docker run -it --gpus all --init --net=host --uts=host --ipc=host \
#      --name hunyuanvideo \
#      --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 \
#      --privileged \
#      -v $(pwd):/workspace \
#      hunyuanvideo/hunyuanvideo:cuda_11 bash
#
#    This drops you into a shell inside the container. You only need this once.
#
# 2. For later runs: start and attach to the existing container
#
#    From the host:
#
#    cd /nfs/oagrawal/HunyuanVideo
#    docker start hunyuanvideo
#    docker exec -it hunyuanvideo bash
#
# 3. Run EasyCache commands inside the container
#
#    cd /workspace
#
#    Baseline (no caching, profiling only):
#    python3 easycache_sample_video.py \
#      --video-size 544 960 --video-length 129 --infer-steps 50 \
#      --flow-reverse --use-cpu-offload --save-path ./easycache_results \
#      --prompt "Two cats boxing in bright gloves on a spotlighted stage." \
#      --seed 12345 \
#      --easycache-mode baseline
#
#    EasyCache enabled (with accumulated metric plot):
#    python3 easycache_sample_video.py \
#      --video-size 544 960 --video-length 129 --infer-steps 50 \
#      --flow-reverse --use-cpu-offload --save-path ./easycache_results \
#      --prompt "Two cats boxing in bright gloves on a spotlighted stage." \
#      --seed 12345 \
#      --easycache-mode easycache \
#      --easycache-thresh 0.025 \
#      --easycache-ret-steps 5
#
# ============================================================================
#
# Usage: ./run_easycache_experiments.sh [OPTIONS]
#
# Options:
#   --prompt "..."       Custom prompt (default: boxing cats)
#   --seed NUM           Random seed (default: 12345)
#   --mode baseline|both|easycache  baseline only, both, or easycache only (default: both)
#   --thresh NUM         EasyCache threshold (default: 0.025, used when mode=both or easycache)
#   --dry-run            Print commands without executing
#

set -e

PROMPT="Two cats boxing in bright gloves on a spotlighted stage."
SEED=12345
MODE="both"
THRESH=0.025
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --thresh)
            THRESH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            head -70 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

COMMON_ARGS="--video-size 544 960 --video-length 129 --infer-steps 50 --flow-reverse --use-cpu-offload --save-path ./easycache_results"

echo "=============================================="
echo "EasyCache Experiments Configuration"
echo "=============================================="
echo "Prompt: $PROMPT"
echo "Seed: $SEED"
echo "Mode: $MODE"
echo "Threshold: $THRESH"
echo "=============================================="

COMMANDS=""

if [ "$MODE" = "baseline" ] || [ "$MODE" = "both" ]; then
    COMMANDS+="echo '========================================' && "
    COMMANDS+="echo '[1] Running baseline (no caching, profiling)...' && "
    COMMANDS+="echo '========================================' && "
    COMMANDS+="python3 easycache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --seed $SEED --easycache-mode baseline && "
fi

if [ "$MODE" = "easycache" ] || [ "$MODE" = "both" ]; then
    COMMANDS+="echo '========================================' && "
    COMMANDS+="echo '[2] Running EasyCache (threshold=$THRESH)...' && "
    COMMANDS+="echo '========================================' && "
    COMMANDS+="python3 easycache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --seed $SEED --easycache-mode easycache --easycache-thresh $THRESH --easycache-ret-steps 5 && "
fi

COMMANDS+="echo '========================================' && "
COMMANDS+="echo 'EasyCache experiments completed!' && "
COMMANDS+="echo '========================================'"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute in Docker container:"
    echo ""
    echo "$COMMANDS" | sed 's/ && /\n/g' | grep -E "echo|python3"
    exit 0
fi

if ! docker ps -a --format '{{.Names}}' | grep -q '^hunyuanvideo$'; then
    echo "Error: Docker container 'hunyuanvideo' not found."
    echo ""
    echo "Create it first with:"
    echo "  cd $(pwd)"
    echo "  docker run -it --gpus all --init --net=host --uts=host --ipc=host \\"
    echo "    --name hunyuanvideo \\"
    echo "    --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 \\"
    echo "    --privileged \\"
    echo "    -v \$(pwd):/workspace \\"
    echo "    hunyuanvideo/hunyuanvideo:cuda_11 bash"
    exit 1
fi

docker start hunyuanvideo
docker exec -it hunyuanvideo bash -c "cd /workspace && $COMMANDS"

echo ""
echo "=============================================="
echo "Results saved to: ./easycache_results/"
echo "=============================================="
