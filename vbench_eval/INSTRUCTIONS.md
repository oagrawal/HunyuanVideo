# VBench Evaluation Pipeline — Full Instructions

This document covers the end-to-end pipeline for evaluating TeaCache on HunyuanVideo
using VBench (16 dimensions) and fidelity metrics (PSNR/SSIM/LPIPS).

---

## Overview

### What we're evaluating

4 generation modes, compared across 55 VBench prompts (covering all 16 dimensions):

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
├── prompts_subset.json          # 55 selected VBench prompts (all 16 dims)
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

- **Video generation**: ~55 prompts × 4 modes × ~7 min/video = ~25 hours on 2 GPUs
- **VBench evaluation**: ~2-4 hours per mode (16 dimensions)
- **Fidelity metrics**: ~30 min total

---

## Step 1: Generate Videos

All commands run **inside the Docker container** (`hv` to enter).

### Prerequisites

```bash
# Enter Docker container
hv

# Make sure correct transformers version is installed
pip install transformers==4.46.3
```

### Dry run (test without generating)

Works outside Docker too — no heavy imports needed.

```bash
python3 vbench_eval/batch_generate.py --dry-run \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload --save-path /tmp/unused
```

### Generate all videos on a single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate.py \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload --save-path /tmp/unused \
    --generation-seed 0
```

### Split across 2 GPUs

There are 55 prompts. Split them into two halves:

**GPU 0 — Terminal 1** (prompts 0-27):
```bash
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate.py \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload --save-path /tmp/unused \
    --generation-seed 0 --start-idx 0 --end-idx 28
```

**GPU 1 — Terminal 2** (prompts 28-54):
```bash
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/batch_generate.py \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload --save-path /tmp/unused \
    --generation-seed 0 --start-idx 28 --end-idx 55
```

### Generate only specific modes

```bash
# Only baseline
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate.py \
    --modes hunyuan_baseline \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload --save-path /tmp/unused

# Only the cached modes (skip baseline)
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate.py \
    --modes hunyuan_fixed_0.1,hunyuan_fixed_0.2,hunyuan_adaptive \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --flow-reverse --use-cpu-offload --save-path /tmp/unused
```

### Resume after interruption

Just re-run the same command. The script automatically skips videos that already exist
on disk and logs their completion in `generation_log.json`.

### Notes

- `--save-path /tmp/unused` is required by hyvideo's parser but not used by the batch
  script (it writes to `--output-dir` instead, defaulting to `vbench_eval/videos/`).
- Video filenames follow VBench convention: `{prompt_text}-{seed}.mp4`
- Generation timing is logged to `vbench_eval/videos/generation_log.json`

---

## Step 2: Run VBench Evaluation

**IMPORTANT**: Switch transformers version before running VBench.

```bash
# Inside Docker container
pip install transformers==4.33.2
```

### Evaluate all modes (all 16 dimensions)

```bash
python3 vbench_eval/run_vbench_eval.py
```

### Evaluate a single mode

```bash
python3 vbench_eval/run_vbench_eval.py --modes hunyuan_baseline
```

### Evaluate specific dimensions only

```bash
python3 vbench_eval/run_vbench_eval.py \
    --dimensions subject_consistency,motion_smoothness,temporal_flickering
```

### Output

Results saved to `vbench_eval/vbench_scores/{mode}/`:
- `{dimension}_eval_results.json` — raw score for each dimension
- `{dimension}_full_info.json` — metadata about which videos were evaluated

---

## Step 3: Run Fidelity Metrics

Compares each cached mode's videos frame-by-frame against the baseline (no-cache) videos
for the same prompt and seed.

**Requires**: baseline videos to exist in `vbench_eval/videos/hunyuan_baseline/`.

```bash
# Inside Docker (needs torch, lpips, imageio, opencv)
pip install lpips  # if not already installed

python3 vbench_eval/run_fidelity_metrics.py
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

### Also save as CSV (for spreadsheets)

```bash
python3 vbench_eval/compare_results.py --output-csv vbench_eval/results_table.csv
```

### Output

- Prints a full comparison table to stdout
- Saves `vbench_eval/all_comparison_results.json` (combined JSON)
- Optionally saves CSV file

### What the output shows

1. **VBench raw scores** — all 16 dimensions per mode (higher = better)
2. **VBench aggregate scores** — quality score, semantic score, total score
   (normalized using TeaCache's exact normalization weights)
3. **Fidelity metrics** — PSNR (higher = better), SSIM (higher = better),
   LPIPS (lower = better) vs baseline
4. **Generation timing** — avg time per video, speedup vs baseline

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
