# EasyCache — Run Instructions

All commands run in the HunyuanVideo project root. Uses Docker container **`hunyuanvideo`** (image: `hunyuanvideo/hunyuanvideo:cuda_11`).

---

## Part A: Quick Single-Video Run (4 modes)

Use this to test the 4 EasyCache modes on one prompt.

### 1. One-time: create the container

```bash
cd /nfs/oagrawal/HunyuanVideo

docker run -it --gpus all --init --net=host --uts=host --ipc=host \
  --name hunyuanvideo \
  --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 \
  --privileged \
  -v $(pwd):/workspace \
  hunyuanvideo/hunyuanvideo:cuda_11 bash
```

This enters the container. You only need this once.

### 2. Later runs: start and enter container

```bash
cd /nfs/oagrawal/HunyuanVideo
docker start hunyuanvideo
docker exec -it hunyuanvideo bash
```

### 3. Run inside container

```bash
cd /workspace
pip install transformers==4.46.3
```

**Mode 1 — Baseline (no caching):**
```bash
python3 easycache_sample_video.py \
  --video-size 544 960 --video-length 129 --infer-steps 50 \
  --flow-reverse --use-cpu-offload --save-path ./easycache_results \
  --prompt "Two cats boxing in bright gloves on a spotlighted stage." \
  --seed 12345 --easycache-mode baseline
```

**Mode 2 — Fixed low (0.025):**
```bash
python3 easycache_sample_video.py \
  --video-size 544 960 --video-length 129 --infer-steps 50 \
  --flow-reverse --use-cpu-offload --save-path ./easycache_results \
  --prompt "Two cats boxing in bright gloves on a spotlighted stage." \
  --seed 12345 --easycache-mode easycache --easycache-thresh 0.025
```

**Mode 3 — Fixed high (0.050):**
```bash
python3 easycache_sample_video.py \
  --video-size 544 960 --video-length 129 --infer-steps 50 \
  --flow-reverse --use-cpu-offload --save-path ./easycache_results \
  --prompt "Two cats boxing in bright gloves on a spotlighted stage." \
  --seed 12345 --easycache-mode easycache --easycache-thresh 0.050
```

**Mode 4 — Adaptive (low 0.025 at start/end, high 0.050 in middle):**
```bash
python3 easycache_sample_video.py \
  --video-size 544 960 --video-length 129 --infer-steps 50 \
  --flow-reverse --use-cpu-offload --save-path ./easycache_results \
  --prompt "Two cats boxing in bright gloves on a spotlighted stage." \
  --seed 12345 --easycache-mode adaptive \
  --easycache-thresh-low 0.025 --easycache-thresh-high 0.050 \
  --easycache-first-steps 8 --easycache-last-steps 6
```

Results go to `./easycache_results/` with plots (k_t, pred_change, acc_pred_change).

---

## Part B: Full VBench Evaluation (4 modes × 33 prompts)

Generates 132 videos, runs VBench (16 dims) and fidelity metrics. Use **tmux** so jobs survive SSH disconnects.

### Container

- **Name:** `hunyuanvideo`
- **Image:** `hunyuanvideo/hunyuanvideo:cuda_11`
- **Working dir:** `/workspace` (project root mounted)

### 4 EasyCache modes

| Mode | Description |
|------|-------------|
| `easycache_baseline` | No caching (ground truth) |
| `easycache_fixed_0.025` | Fixed threshold 0.025 |
| `easycache_fixed_0.050` | Fixed threshold 0.050 |
| `easycache_adaptive` | Low 0.025 (steps 5–12, 43–48), high 0.050 (middle) |

### Folder layout

```
vbench_eval_easycache/
├── videos/                    # Generated videos
│   ├── easycache_baseline/
│   ├── easycache_fixed_0.025/
│   ├── easycache_fixed_0.050/
│   └── easycache_adaptive/
├── vbench_scores/             # VBench per-dimension results
├── fidelity_metrics/          # PSNR / SSIM / LPIPS vs baseline
└── all_comparison_results.json
```

Uses `vbench_eval/prompts_subset.json` (33 prompts).

---

### Step 1: Generate videos (split across 4 GPUs)

Each GPU gets 1/4 of the prompts. No race conditions — each GPU writes different files.

**On host, create 4 tmux sessions, then attach to the container in each:**

```bash
# Terminal 1
tmux new -s ec_gpu0
docker start hunyuanvideo
docker exec -it hunyuanvideo bash
cd /workspace
pip install transformers==4.46.3

# Terminal 2 (new SSH or new tab)
tmux new -s ec_gpu1
docker exec -it hunyuanvideo bash
cd /workspace
pip install transformers==4.46.3

# Terminal 3
tmux new -s ec_gpu2
docker exec -it hunyuanvideo bash
cd /workspace
pip install transformers==4.46.3

# Terminal 4
tmux new -s ec_gpu3
docker exec -it hunyuanvideo bash
cd /workspace
pip install transformers==4.46.3
```

**Run generation in each terminal (different `--start-idx` / `--end-idx`):**

**Terminal 1 (GPU 0):**
```bash
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval_easycache/batch_generate.py \
  --video-size 544 960 --video-length 129 --infer-steps 50 \
  --flow-reverse --use-cpu-offload \
  --start-idx 0 --end-idx 9
```

**Terminal 2 (GPU 1):**
```bash
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval_easycache/batch_generate.py \
  --video-size 544 960 --video-length 129 --infer-steps 50 \
  --flow-reverse --use-cpu-offload \
  --start-idx 9 --end-idx 18
```

**Terminal 3 (GPU 2):**
```bash
CUDA_VISIBLE_DEVICES=2 python3 vbench_eval_easycache/batch_generate.py \
  --video-size 544 960 --video-length 129 --infer-steps 50 \
  --flow-reverse --use-cpu-offload \
  --start-idx 18 --end-idx 27
```

**Terminal 4 (GPU 3):**
```bash
CUDA_VISIBLE_DEVICES=3 python3 vbench_eval_easycache/batch_generate.py \
  --video-size 544 960 --video-length 129 --infer-steps 50 \
  --flow-reverse --use-cpu-offload \
  --start-idx 27 --end-idx 33
```

**Resume:** Re-run the same commands; existing videos are skipped.

**Reattach if SSH drops:**
```bash
tmux attach -t ec_gpu0   # etc.
```

---

### Step 2: Run VBench evaluation

Requires `transformers==4.33.2` in the container.

```bash
pip install transformers==4.33.2
```

**Split across 4 GPUs (one mode per GPU):**

**Terminal 1:**
```bash
tmux new -s ec_vb0
docker exec -it hunyuanvideo bash
cd /workspace
pip install transformers==4.33.2

CUDA_VISIBLE_DEVICES=0 python3 vbench_eval_easycache/run_vbench_eval.py \
  --modes easycache_baseline
```

**Terminal 2:**
```bash
tmux new -s ec_vb1
docker exec -it hunyuanvideo bash
cd /workspace
pip install transformers==4.33.2

CUDA_VISIBLE_DEVICES=1 python3 vbench_eval_easycache/run_vbench_eval.py \
  --modes easycache_fixed_0.025
```

**Terminal 3:**
```bash
tmux new -s ec_vb2
docker exec -it hunyuanvideo bash
cd /workspace
pip install transformers==4.33.2

CUDA_VISIBLE_DEVICES=2 python3 vbench_eval_easycache/run_vbench_eval.py \
  --modes easycache_fixed_0.050
```

**Terminal 4:**
```bash
tmux new -s ec_vb3
docker exec -it hunyuanvideo bash
cd /workspace
pip install transformers==4.33.2

CUDA_VISIBLE_DEVICES=3 python3 vbench_eval_easycache/run_vbench_eval.py \
  --modes easycache_adaptive
```

No race conditions — each mode writes to its own directory.

**2 GPU alternative:** Run 2 modes per GPU in parallel. Use two tmux sessions:

**Terminal 1 (GPU 0):**
```bash
tmux new -s ec_vb0
docker exec -it hunyuanvideo bash
cd /workspace
pip install transformers==4.33.2

CUDA_VISIBLE_DEVICES=0 python3 vbench_eval_easycache/run_vbench_eval.py \
  --modes easycache_baseline,easycache_fixed_0.025
```

**Terminal 2 (GPU 1):**
```bash
tmux new -s ec_vb1
docker exec -it hunyuanvideo bash
cd /workspace
pip install transformers==4.33.2

CUDA_VISIBLE_DEVICES=1 python3 vbench_eval_easycache/run_vbench_eval.py \
  --modes easycache_fixed_0.050,easycache_adaptive
```

---

### Step 3: Run fidelity metrics (PSNR / SSIM / LPIPS)

**Single GPU:**
```bash
tmux new -s ec_fid
docker exec -it hunyuanvideo bash
cd /workspace
pip install lpips

CUDA_VISIBLE_DEVICES=0 python3 vbench_eval_easycache/run_fidelity_metrics.py
```

**2 GPU alternative:** Each GPU writes only its own mode files (no shared writes, no race conditions). Use two tmux sessions:

**Terminal 1 (GPU 0):**
```bash
tmux new -s ec_fid0
docker exec -it hunyuanvideo bash
cd /workspace
pip install lpips

CUDA_VISIBLE_DEVICES=0 python3 vbench_eval_easycache/run_fidelity_metrics.py \
  --modes easycache_fixed_0.025,easycache_fixed_0.050
```

**Terminal 2 (GPU 1):**
```bash
tmux new -s ec_fid1
docker exec -it hunyuanvideo bash
cd /workspace
pip install lpips

CUDA_VISIBLE_DEVICES=1 python3 vbench_eval_easycache/run_fidelity_metrics.py \
  --modes easycache_adaptive
```

After both finish, `compare_results.py` will merge the per-mode JSON files automatically.

---

### Step 4: Compare all results

**No GPU needed** — run on host or in container:

```bash
python3 vbench_eval_easycache/compare_results.py
```

Output: `vbench_eval_easycache/all_comparison_results.json` and printed summary.

---

## Transformers versions

| Task | Version |
|------|---------|
| Video generation | `4.46.3` |
| VBench evaluation | `4.33.2` |
| Fidelity / compare | Either |

---

## Troubleshooting

**Container not found:**
```bash
docker run -it --gpus all --init --net=host --uts=host --ipc=host \
  --name hunyuanvideo \
  --security-opt=seccomp=unconfined --ulimit=stack=67108864 --ulimit=memlock=-1 \
  --privileged \
  -v $(pwd):/workspace \
  hunyuanvideo/hunyuanvideo:cuda_11 bash
```

**Permission errors:**
```bash
chmod -R 777 ./
```

**detectron2 for VBench (some dimensions):**
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
