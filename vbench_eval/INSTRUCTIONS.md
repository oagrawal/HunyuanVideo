# VBench Evaluation Pipeline — Full Instructions

This document covers the end-to-end pipeline for evaluating TeaCache on HunyuanVideo
using VBench (16 dimensions) and fidelity metrics (PSNR/SSIM/LPIPS).

---

## Overview

### What we're evaluating

4 generation modes, compared across 33 VBench prompts (3 per dimension group, covering all 16 dimensions):

| Mode | Description |
|---|---|
| `hunyuan_baseline` | No caching (ground truth) |
| `hunyuan_fixed_0.1` | Fixed threshold 0.1 for all steps |
| `hunyuan_fixed_0.2` | Fixed threshold 0.2 for all steps |
| `hunyuan_adaptive` | Threshold 0.1 for first 5 + last 10 steps, 0.2 for middle steps |

### Folder structure

```
vbench_eval/
├── INSTRUCTIONS.md              # This file
├── prompts_subset.json          # 33 selected VBench prompts (all 16 dims)
├── select_prompts.py            # Script that generated the subset
├── batch_generate.py            # Batch video generation script
├── run_vbench_eval.py           # VBench 16-dimension evaluation
├── run_fidelity_metrics.py      # PSNR / SSIM / LPIPS vs baseline
├── compare_results.py           # Aggregate results into comparison table
├── videos/                      # Generated videos
│   ├── hunyuan_baseline/        #   {prompt}-{seed}.mp4
│   ├── hunyuan_fixed_0.1/
│   ├── hunyuan_fixed_0.2/
│   └── hunyuan_adaptive/
├── vbench_scores/               # VBench evaluation results
│   ├── hunyuan_baseline/        #   {dim}_eval_results.json
│   ├── hunyuan_fixed_0.1/
│   ├── hunyuan_fixed_0.2/
│   └── hunyuan_adaptive/
├── fidelity_metrics/            # PSNR/SSIM/LPIPS results
│   ├── hunyuan_fixed_0.1_vs_hunyuan_baseline.json
│   ├── hunyuan_fixed_0.2_vs_hunyuan_baseline.json
│   ├── hunyuan_adaptive_vs_hunyuan_baseline.json
│   └── all_fidelity_results.json
└── all_comparison_results.json  # Combined comparison output
```

### Estimated time

- **Video generation**: ~33 prompts × 4 modes = 132 videos, ~15.5 hours per GPU on 2 GPUs
- **VBench evaluation**: ~2-4 hours per mode (16 dimensions)
- **Fidelity metrics**: ~30 min total

---

## Step 1: Generate Videos

All commands run **inside the Docker container** (`hv` to enter).

### Prerequisites

```bash
# In both terminals, make sure correct transformers version is installed
pip install transformers==4.46.3
```

### Setting up 2 terminals (with tmux for SSH safety)

Generation takes ~15.5 hours per GPU. Use `tmux` so your processes survive SSH
disconnections.

```bash
# Terminal 1: create a tmux session, enter Docker
tmux new -s gpu0
hv
# (run GPU 0 command here)

# Terminal 2 (new SSH connection): create another tmux session, enter Docker
tmux new -s gpu1
docker exec -it hunyuanvideo bash
# (run GPU 1 command here)
```

If SSH disconnects, reconnect and reattach:
```bash
tmux attach -t gpu0   # reattach to GPU 0 session
tmux attach -t gpu1   # reattach to GPU 1 session
```

> **Note**: `hv` = `docker start -ai hunyuanvideo`. You can only run it once.
> For additional shells, use `docker exec -it hunyuanvideo bash`.

### Dry run (test without generating)

Works outside Docker too — no heavy imports needed.

```bash
python3 vbench_eval/batch_generate.py --dry-run \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload
```

### Generate all videos on a single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate.py \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload
```

### Split across 2 GPUs (recommended)

There are 33 prompts. Split them into two halves. Each GPU runs all 4 modes
for its half of the prompts (~66 videos per GPU ≈ 15.5 hours each).

**GPU 0 — Terminal 1** (prompts 0-16):
```bash
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate.py \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload \
    --start-idx 0 --end-idx 17
```

**GPU 1 — Terminal 2** (prompts 17-32):
```bash
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/batch_generate.py \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload \
    --start-idx 17 --end-idx 33
```

### Generate only specific modes

```bash
# Only baseline
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate.py \
    --modes hunyuan_baseline \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload

# Only the cached modes (skip baseline)
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate.py \
    --modes hunyuan_fixed_0.1,hunyuan_fixed_0.2,hunyuan_adaptive \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload
```

### Resume after interruption

Just re-run the exact same command. The script automatically:
- Checks which video files already exist on disk
- Prints `[X/Y] SKIP (exists)` for each one it skips
- Only generates the remaining videos
- Shows a summary at the end: completed / skipped / failed

Progress is saved to per-process log files (e.g. `generation_log_0-17.json`,
`generation_log_17-33.json`) with per-video timing data. The `compare_results.py`
script automatically merges all log files when aggregating results.

### Notes

- Video filenames follow VBench convention: `{prompt_text}-{seed}.mp4`
- Generation timing is logged per-process (e.g. `generation_log_0-17.json`)
- The script auto-injects `--save-path` for hyvideo's parser; you don't need to set it
- Both GPUs write to the same output directories (no conflicts since each writes
  different prompt files)

---

## Step 2: Run VBench Evaluation

**IMPORTANT**: Switch transformers version before running VBench.

```bash
# Inside Docker container (both terminals)
pip install transformers==4.33.2
```

### Setting up terminals with tmux

Evaluation takes ~2-3 hours per GPU. Use tmux so processes survive SSH drops.

```bash
# Terminal 1
tmux new -s eval0
hv                  # or: docker exec -it hunyuanvideo bash (if already running)
pip install transformers==4.33.2

# Terminal 2 (new SSH connection)
tmux new -s eval1
docker exec -it hunyuanvideo bash
pip install transformers==4.33.2
```

If SSH disconnects, reconnect and reattach:
```bash
tmux attach -t eval0
tmux attach -t eval1
```

### Quick test (run first to verify everything works)

Test both GPUs in parallel, each evaluating 1 dimension on 1 mode. Takes a few minutes.
These results won't be overwritten — the full run skips dimensions that already have results.

**GPU 0 — Terminal 1:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_vbench_eval.py \
    --modes hunyuan_baseline \
    --dimensions subject_consistency
```

**GPU 1 — Terminal 2:**
```bash
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/run_vbench_eval.py \
    --modes hunyuan_fixed_0.2 \
    --dimensions temporal_flickering
```

If both succeed, proceed to the full run below. The 2 test dimensions will be
automatically skipped (they write to the same directories the full run uses).

### Full run — split across 2 GPUs (recommended)

Each GPU evaluates 2 modes (all 16 dimensions). ~2-3 hours per GPU.
No race conditions — each mode writes to its own directory.

**GPU 0 — Terminal 1** (baseline + fixed 0.1):
```bash
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_vbench_eval.py \
    --modes hunyuan_baseline,hunyuan_fixed_0.1
```

**GPU 1 — Terminal 2** (fixed 0.2 + adaptive):
```bash
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/run_vbench_eval.py \
    --modes hunyuan_fixed_0.2,hunyuan_adaptive
```

### Resume after interruption

The script automatically skips dimensions that already have results
(`{dimension}_eval_results.json`). Just re-run the same command and it picks up
where it left off.

### Evaluate a single mode or specific dimensions

```bash
# Single mode
python3 vbench_eval/run_vbench_eval.py --modes hunyuan_baseline

# Specific dimensions
python3 vbench_eval/run_vbench_eval.py \
    --dimensions subject_consistency,motion_smoothness,temporal_flickering
```

### Output

Results saved to `vbench_eval/vbench_scores/{mode}/`:
- `{dimension}_eval_results.json` — raw score for each dimension
- `{dimension}_full_info.json` — metadata about which videos were evaluated

### Viewing VBench results

Check how many dimensions are complete per mode:
```bash
for mode in hunyuan_baseline hunyuan_fixed_0.1 hunyuan_fixed_0.2 hunyuan_adaptive; do
    count=$(ls vbench_eval/vbench_scores/$mode/*_eval_results.json 2>/dev/null | wc -l)
    echo "$mode: $count/16 dimensions"
done
```

View all scores in a comparison table (no GPU needed, runs outside Docker):
```bash
python3 vbench_eval/compare_results.py
```

### Dependencies

VBench dimensions `multiple_objects`, `spatial_relationship`, `object_class`, and `color`
require `detectron2`. Install inside Docker if not already present:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

---

## Step 3: Run Fidelity Metrics

Compares each cached mode's videos frame-by-frame against the baseline (no-cache) videos
for the same prompt and seed. Takes ~30 minutes on a single GPU.

**Requires**: baseline videos to exist in `vbench_eval/videos/hunyuan_baseline/`.

### Setting up terminal with tmux

```bash
tmux new -s fidelity
hv   # or: docker exec -it hunyuanvideo bash (if container already running)
```

If SSH disconnects, reconnect and reattach:
```bash
tmux attach -t fidelity
```

### Run fidelity metrics

```bash
# Install lpips if not already installed
pip install lpips

# Run (single GPU is sufficient, ~30 min)
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_fidelity_metrics.py
```

### Compare specific modes only

```bash
python3 vbench_eval/run_fidelity_metrics.py --modes hunyuan_fixed_0.1
```

### Output

Results saved to `vbench_eval/fidelity_metrics/`:
- `{mode}_vs_hunyuan_baseline.json` — per-mode results
- `all_fidelity_results.json` — combined results

---

## Step 4: Compare All Results

Aggregates VBench scores, fidelity metrics, and generation timing into one view.
**No heavy dependencies** — runs outside Docker.

```bash
python3 vbench_eval/compare_results.py
```

### Output files

| File | Description |
|---|---|
| `vbench_eval/vbench_scores_table.csv` | VBench scores — all 16 dimensions per mode (quality dims, then semantic dims) + latency |
| `vbench_eval/fidelity_table.csv` | Fidelity metrics — PSNR/SSIM/LPIPS per mode + latency |
| `vbench_eval/summary_table.csv` | Compact summary — speedup, latency, VBench total, PSNR, SSIM, LPIPS |
| `vbench_eval/all_comparison_results.json` | Combined JSON with all raw data |

### What the output shows

The script prints **3 tables** to stdout (and saves each as a CSV):

**Table 1 — VBench Scores:**
Rows = 4 modes, Columns = 7 quality dimensions | 9 semantic dimensions | latency.
Quality and semantic dimensions are grouped together. Aggregate scores
(quality score, semantic score, total score) are shown below the table.

**Table 2 — Fidelity Metrics:**
Rows = 4 modes, Columns = PSNR | SSIM | LPIPS | latency.
Baseline shows "—" (it's the reference). Higher PSNR/SSIM = better,
lower LPIPS = better.

**Table 3 — Compact Summary:**
One-line-per-mode overview: speedup, latency, VBench total, PSNR, SSIM, LPIPS.

---

## Quick Reference: Transformers Version

| Task | Transformers Version |
|---|---|
| Video generation (`batch_generate.py`) | `4.46.3` |
| VBench evaluation (`run_vbench_eval.py`) | `4.33.2` |
| Fidelity metrics (`run_fidelity_metrics.py`) | Either works |
| Compare results (`compare_results.py`) | Either (no torch needed) |

Switch with:
```bash
pip install transformers==4.46.3   # for generation
pip install transformers==4.33.2   # for VBench eval
```

---

## Scaling Up: Multiple Seeds

VBench standard mode expects up to 5 seeds per prompt (seed indices 0-4). To run with
multiple seeds:

```bash
# Seed 0 (default)
python3 vbench_eval/batch_generate.py --generation-seed 0 ...

# Seed 1
python3 vbench_eval/batch_generate.py --generation-seed 1 ...

# Seed 2
python3 vbench_eval/batch_generate.py --generation-seed 2 ...
```

Each seed's video is saved as `{prompt}-{seed_index}.mp4` in the same directory.
VBench automatically finds and averages across all available seeds per prompt.

---

## Troubleshooting

### EACCES permission error
```bash
chmod -R 777 ./
```

### Docker container not found
```bash
# Recreate container
docker rm -f hunyuanvideo
docker run -it --gpus all --init --net=host --uts=host --ipc=host \
    --name hunyuanvideo --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged \
    -v $(pwd):/workspace hunyuanvideo/hunyuanvideo:cuda_11 bash

# Install requirements
pip install -r requirements.txt
pip install transformers==4.46.3
```

### VBench "video not found" warnings
VBench prints warnings for missing seed indices (1-4) when running with 1 seed.
This is normal and does not affect evaluation — it just uses whatever seeds exist.

### Out of GPU memory
Use `--use-cpu-offload` flag in the generation command (already included in examples).
