#!/bin/bash
#
# Automated TeaCache Experiments Runner
# Runs all 9 video generation experiments inside the HunyuanVideo Docker container
#
# Usage: ./run_teacache_experiments.sh [OPTIONS]
#
# Options:
#   --prompt "..."     Custom prompt (default: boxing cats)
#   --seed NUM         Random seed (default: 12345)
#   --dry-run          Print commands without executing
#

set -e  # Exit on error

# Default values
PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
SEED=12345
DRY_RUN=false

# Parse command line arguments
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prompt \"...\"     Custom prompt (default: boxing cats)"
            echo "  --seed NUM         Random seed (default: 12345)"
            echo "  --dry-run          Print commands without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Common arguments for all runs
COMMON_ARGS="--video-size 544 960 --video-length 129 --infer-steps 50 --flow-reverse --use-cpu-offload --save-path ./teacache_results"

echo "=============================================="
echo "TeaCache Experiments Configuration"
echo "=============================================="
echo "Prompt: $PROMPT"
echo "Seed: $SEED"
echo "Total experiments: 9"
echo "=============================================="
echo ""
echo "Experiments to run:"
echo "  1. Baseline (no caching)"
echo "  2. Fixed threshold 0.10"
echo "  3. Fixed threshold 0.20"
echo "  4. Fixed threshold 0.30"
echo "  5. Adaptive low=0.20 high=0.30"
echo "  6. Adaptive low=0.15 high=0.30"
echo "  7. Adaptive low=0.10 high=0.30"
echo "  8. Adaptive low=0.05 high=0.30"
echo "  9. Adaptive low=0.00 high=0.30"
echo "=============================================="
echo ""

# Build the commands to run inside the container
COMMANDS=""

# 1. Baseline (no caching)
COMMANDS+="echo '========================================' && "
COMMANDS+="echo '[1/9] Running baseline (no caching)...' && "
COMMANDS+="echo '========================================' && "
COMMANDS+="python3 teacache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --teacache-mode none --seed $SEED && "

# 2. Fixed threshold 0.10
COMMANDS+="echo '========================================' && "
COMMANDS+="echo '[2/9] Running fixed threshold 0.10...' && "
COMMANDS+="echo '========================================' && "
COMMANDS+="python3 teacache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --teacache-mode fixed --fixed-thresh 0.10 --seed $SEED && "

# 3. Fixed threshold 0.20
COMMANDS+="echo '========================================' && "
COMMANDS+="echo '[3/9] Running fixed threshold 0.20...' && "
COMMANDS+="echo '========================================' && "
COMMANDS+="python3 teacache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --teacache-mode fixed --fixed-thresh 0.20 --seed $SEED && "

# 4. Fixed threshold 0.30
COMMANDS+="echo '========================================' && "
COMMANDS+="echo '[4/9] Running fixed threshold 0.30...' && "
COMMANDS+="echo '========================================' && "
COMMANDS+="python3 teacache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --teacache-mode fixed --fixed-thresh 0.30 --seed $SEED && "

# 5. Adaptive low=0.20 high=0.30
COMMANDS+="echo '========================================' && "
COMMANDS+="echo '[5/9] Running adaptive low=0.20 high=0.30...' && "
COMMANDS+="echo '========================================' && "
COMMANDS+="python3 teacache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --teacache-mode adaptive --thresh-low 0.20 --thresh-high 0.30 --seed $SEED && "

# 6. Adaptive low=0.15 high=0.30
COMMANDS+="echo '========================================' && "
COMMANDS+="echo '[6/9] Running adaptive low=0.15 high=0.30...' && "
COMMANDS+="echo '========================================' && "
COMMANDS+="python3 teacache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --teacache-mode adaptive --thresh-low 0.15 --thresh-high 0.30 --seed $SEED && "

# 7. Adaptive low=0.10 high=0.30
COMMANDS+="echo '========================================' && "
COMMANDS+="echo '[7/9] Running adaptive low=0.10 high=0.30...' && "
COMMANDS+="echo '========================================' && "
COMMANDS+="python3 teacache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --teacache-mode adaptive --thresh-low 0.10 --thresh-high 0.30 --seed $SEED && "

# 8. Adaptive low=0.05 high=0.30
COMMANDS+="echo '========================================' && "
COMMANDS+="echo '[8/9] Running adaptive low=0.05 high=0.30...' && "
COMMANDS+="echo '========================================' && "
COMMANDS+="python3 teacache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --teacache-mode adaptive --thresh-low 0.05 --thresh-high 0.30 --seed $SEED && "

# 9. Adaptive low=0.00 high=0.30
COMMANDS+="echo '========================================' && "
COMMANDS+="echo '[9/9] Running adaptive low=0.00 high=0.30...' && "
COMMANDS+="echo '========================================' && "
COMMANDS+="python3 teacache_sample_video.py $COMMON_ARGS --prompt \"$PROMPT\" --teacache-mode adaptive --thresh-low 0 --thresh-high 0.30 --seed $SEED && "

# Completion message
COMMANDS+="echo '========================================' && "
COMMANDS+="echo 'All 9 experiments completed!' && "
COMMANDS+="echo '========================================'"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would execute in Docker container:"
    echo ""
    echo "Prompt: $PROMPT"
    echo "Seed: $SEED"
    echo ""
    echo "Commands:"
    echo "$COMMANDS" | sed 's/ && /\n/g' | grep "python3"
    exit 0
fi

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q '^hunyuanvideo$'; then
    echo "Error: Docker container 'hunyuanvideo' not found."
    echo "Please create it first with:"
    echo "  docker run -it --gpus all --init --net=host --uts=host --ipc=host --name hunyuanvideo --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged -v \$(pwd):/workspace hunyuanvideo/hunyuanvideo:cuda_11 bash"
    exit 1
fi

# Run commands inside the container
docker start hunyuanvideo
docker exec -it hunyuanvideo bash -c "cd /workspace && $COMMANDS"

echo ""
echo "=============================================="
echo "All experiments finished!"
echo "Results saved to: ./teacache_results/"
echo "=============================================="
