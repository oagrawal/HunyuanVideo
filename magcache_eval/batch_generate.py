#!/usr/bin/env python3
"""
Batch video generation for MagCache VBench evaluation — HunyuanVideo 544p.

Loads HunyuanVideo once, loops over prompts and MagCache modes.
Saves videos in VBench format: {videos_root}/{mode_name}/{prompt}-{seed}.mp4

Each mode_cfg entry may override K and retention_ratio; global defaults are K=15, rr=0.1.
The k15 modes use K=15, retention_ratio=0.1 to break the speedup ceiling seen with K=6.
Original K=6 modes are kept for reference (their videos already exist on disk and are skipped).

Note on paper naming: "slow/fast mode" in magcache_sample_video.py refers to video *content*
speed (slow-motion vs. fast-motion), NOT generation speed. Higher δ → more skips → faster
generation. Lower δ → fewer skips → slower generation but higher quality.

K_FIXED = 15  (raised from 6; K=6 was a hard ceiling — δ=0.48/0.64/0.80 all capped at 2.60x)
retention_ratio = 0.1  (lowered from 0.2; paper does not prescribe this value.
                         Steps 0-4 early spike remain retained; steps 5-9 now cacheable.)

Resume: skips videos that already exist on disk.
GPU split: --start-idx / --end-idx over the prompt list.
Mode filter: --modes comma-separated list or 'all'.

Usage (inside hv_eval_wan container, from /nfs/oagrawal/HunyuanVideo):
  python3 magcache_eval/batch_generate.py \\
      --video-size 544 960 --video-length 129 --infer-steps 50 \\
      --model-base ckpts --flow-reverse --use-cpu-offload \\
      --flow-shift 7.0 --cfg-scale 1.0 --embedded-cfg-scale 6.0 \\
      --prompts-file vbench_eval/prompts_subset.json \\
      --output-dir magcache_eval/videos \\
      --generation-seed 0 \\
      --modes magcache_k15_fixed_0.06,magcache_k15_fixed_0.12,magcache_k15_fixed_0.24 \\
      --start-idx 0 --end-idx 9
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

# ── MagCache defaults ────────────────────────────────────────────────────────
# K=15: raised from 6 — K=6 was a hard ceiling (δ=0.48/0.64/0.80 all capped at 2.60x speedup).
# retention_ratio=0.1: lowered from 0.2 — paper does not prescribe this value.
#   Steps 0-4 (early spike, Δγ up to 0.22) stay retained; steps 5-9 (Δγ=0.03-0.06) now cacheable.
# Each mode_cfg may override these per-mode via "K" and "retention_ratio" keys.
K_FIXED          = 15
RETENTION_RATIO  = 0.1

# Pre-calibrated mag_ratios for HunyuanVideo 544p (from magcache_sample_video.py)
MAG_RATIOS_544P = np.array([
    1.0,
    1.06971, 1.29073, 1.11245, 1.09596, 1.05233, 1.01415, 1.05672, 1.00848,
    1.03632, 1.02974, 1.00984, 1.03028, 1.00681, 1.06614, 1.05022, 1.02592,
    1.01776, 1.02985, 1.00726, 1.03727, 1.01502, 1.00992, 1.03371, 0.9976,
    1.02742, 1.0093,  1.01869, 1.00815, 1.01461, 1.01152, 1.03082, 1.0061,
    1.02162, 1.01999, 0.99063, 1.01186, 1.0217,  0.99947, 1.01711, 0.9904,
    1.00258, 1.00878, 0.97039, 0.97686, 0.94315, 0.97728, 0.91154, 0.86139,
    0.76592
])

# ── Experiment matrix ─────────────────────────────────────────────────────────
# Each entry may carry optional "K" and "retention_ratio" to override globals.
# configure_magcache() falls back to K_FIXED and RETENTION_RATIO if not present.
#
# Original K=6 / rr=0.2 runs (COMPLETE — videos already on disk, will be skipped):
MODES_K6 = [
    {"name": "magcache_baseline",   "mode": "baseline", "K": 6, "retention_ratio": 0.2},
    {"name": "magcache_fixed_0.06", "mode": "magcache", "delta": 0.06, "K": 6, "retention_ratio": 0.2},
    {"name": "magcache_fixed_0.12", "mode": "magcache", "delta": 0.12, "K": 6, "retention_ratio": 0.2},
    {"name": "magcache_fixed_0.24", "mode": "magcache", "delta": 0.24, "K": 6, "retention_ratio": 0.2},
    {"name": "magcache_fixed_0.36", "mode": "magcache", "delta": 0.36, "K": 6, "retention_ratio": 0.2},
    {"name": "magcache_fixed_0.48", "mode": "magcache", "delta": 0.48, "K": 6, "retention_ratio": 0.2},
    {"name": "magcache_fixed_0.64", "mode": "magcache", "delta": 0.64, "K": 6, "retention_ratio": 0.2},
    {"name": "magcache_fixed_0.80", "mode": "magcache", "delta": 0.80, "K": 6, "retention_ratio": 0.2},
]

# New K=15 / rr=0.1 runs — breaks the 2.60x speedup ceiling from K=6.
# Baseline is reused from K=6 run (no caching → unaffected by K / rr).
MODES_K15 = [
    {"name": "magcache_k15_fixed_0.06", "mode": "magcache", "delta": 0.06},
    {"name": "magcache_k15_fixed_0.12", "mode": "magcache", "delta": 0.12},
    {"name": "magcache_k15_fixed_0.24", "mode": "magcache", "delta": 0.24},
    {"name": "magcache_k15_fixed_0.36", "mode": "magcache", "delta": 0.36},
    {"name": "magcache_k15_fixed_0.48", "mode": "magcache", "delta": 0.48},
    {"name": "magcache_k15_fixed_0.64", "mode": "magcache", "delta": 0.64},
    {"name": "magcache_k15_fixed_0.80", "mode": "magcache", "delta": 0.80},
]

# Default mode list used when --modes all is passed
MODES = MODES_K6 + MODES_K15


def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])
    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


def magcache_baseline_forward(
        self,
        x,
        t,
        text_states=None,
        text_mask=None,
        text_states_2=None,
        freqs_cos=None,
        freqs_sin=None,
        guidance=None,
        return_dict=True,
):
    """Baseline (no-caching) forward — runs the full model every step."""
    import torch
    from hyvideo.modules.attenion import get_cu_seqlens

    out = {}
    img = x
    txt = text_states
    _, _, ot, oh, ow = x.shape
    tt = ot // self.patch_size[0]
    th = oh // self.patch_size[1]
    tw = ow // self.patch_size[2]

    vec = self.time_in(t)
    vec = vec + self.vector_in(text_states_2)
    if self.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + self.guidance_in(guidance)

    img = self.img_in(img)
    if self.text_projection == "linear":
        txt = self.txt_in(txt)
    elif self.text_projection == "single_refiner":
        txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
    else:
        raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

    txt_seq_len = txt.shape[1]
    img_seq_len = img.shape[1]
    cu_seqlens_q  = get_cu_seqlens(text_mask, img_seq_len)
    cu_seqlens_kv = cu_seqlens_q
    max_seqlen_q  = img_seq_len + txt_seq_len
    max_seqlen_kv = max_seqlen_q
    freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

    for _, block in enumerate(self.double_blocks):
        img, txt = block(img, txt, vec, cu_seqlens_q, cu_seqlens_kv,
                         max_seqlen_q, max_seqlen_kv, freqs_cis)

    x = torch.cat((img, txt), 1)
    if len(self.single_blocks) > 0:
        for _, block in enumerate(self.single_blocks):
            x = block(x, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv,
                      max_seqlen_q, max_seqlen_kv, (freqs_cos, freqs_sin))

    img = x[:, :img_seq_len, ...]
    img = self.final_layer(img, vec)
    img = self.unpatchify(img, tt, th, tw)

    if return_dict:
        out["x"] = img
        return out
    return img


def configure_magcache(sampler, mode_cfg, infer_steps):
    """Attach MagCache state to the transformer for one generation run.

    State is set on the INSTANCE (not the class) so it resets cleanly between
    runs. The forward method must be set on the CLASS so PyTorch's nn.Module
    picks it up via MRO.
    """
    from magcache_sample_video import magcache_forward

    transformer     = sampler.pipeline.transformer
    transformer_cls = transformer.__class__

    # Resize mag_ratios if infer_steps differs from the 50-step lookup table
    mag_ratios = MAG_RATIOS_544P
    if len(mag_ratios) != infer_steps:
        mag_ratios = nearest_interp(mag_ratios, infer_steps)

    # ── Per-run state on the INSTANCE ────────────────────────────────────────
    transformer.cnt               = 0
    transformer.num_steps         = infer_steps
    transformer.mag_ratios        = mag_ratios
    transformer.K                 = mode_cfg.get("K", K_FIXED)
    transformer.retention_ratio   = mode_cfg.get("retention_ratio", RETENTION_RATIO)
    transformer.accumulated_ratio = 1.0
    transformer.accumulated_err   = 0.0
    transformer.accumulated_steps = 0
    transformer.residual_cache    = None

    if mode_cfg["mode"] == "baseline":
        transformer_cls.forward     = magcache_baseline_forward
        transformer.magcache_thresh = None
    else:
        transformer.magcache_thresh = mode_cfg["delta"]
        transformer_cls.forward     = magcache_forward


def load_generation_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_generation_log(log_path, log_data):
    tmp = log_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log_data, f, indent=2)
    os.replace(tmp, log_path)


def main():
    batch_parser = argparse.ArgumentParser(add_help=False)
    batch_parser.add_argument("--prompts-file",    type=str, default="vbench_eval/prompts_subset.json")
    batch_parser.add_argument("--output-dir",      type=str, default="magcache_eval/videos")
    batch_parser.add_argument("--generation-seed", type=int, default=0)
    batch_parser.add_argument("--start-idx",       type=int, default=0)
    batch_parser.add_argument("--end-idx",         type=int, default=-1)
    batch_parser.add_argument("--modes",           type=str, default="all",
                               help="Comma-separated mode names or 'all'")
    batch_parser.add_argument("--dry-run",         action="store_true")
    batch_args, remaining_argv = batch_parser.parse_known_args()

    with open(batch_args.prompts_file, "r") as f:
        all_prompts = json.load(f)
    end_idx = len(all_prompts) if batch_args.end_idx == -1 else batch_args.end_idx
    prompts = all_prompts[batch_args.start_idx:end_idx]

    if batch_args.modes == "all":
        modes = MODES
    else:
        names = {m.strip() for m in batch_args.modes.split(",")}
        modes = [m for m in MODES if m["name"] in names]
        if not modes:
            print(f"ERROR: No valid modes. Available: {[m['name'] for m in MODES]}")
            sys.exit(1)

    seed       = batch_args.generation_seed
    output_dir = batch_args.output_dir
    start_idx  = batch_args.start_idx

    print("=" * 70)
    print("MagCache VBench Batch Generation — HunyuanVideo 544p")
    print("=" * 70)
    print(f"Prompts [{start_idx}, {end_idx}):  {len(prompts)}")
    print(f"Modes:   {[m['name'] for m in modes]}")
    print(f"Seed:    {seed}  |  K_default={K_FIXED}  |  retention_ratio_default={RETENTION_RATIO}")
    print(f"Output:  {output_dir}")
    print("=" * 70)

    if batch_args.dry_run:
        for entry in prompts:
            prompt = entry["prompt_en"]
            for m in modes:
                p = os.path.join(output_dir, m["name"], f"{prompt}-{seed}.mp4")
                print(f"  {'EXISTS' if os.path.exists(p) else 'NEW':6s}  {m['name']}/{prompt[:60]}-{seed}.mp4")
        return

    # ── Heavy imports only when actually generating ───────────────────────────
    if "--save-path" not in remaining_argv:
        remaining_argv += ["--save-path", output_dir]
    sys.argv = [sys.argv[0]] + remaining_argv

    from hyvideo.config import parse_args
    from hyvideo.inference import HunyuanVideoSampler
    from hyvideo.utils.file_utils import save_videos_grid

    args = parse_args()

    log_filename = f"generation_log_{start_idx}-{end_idx}.json"
    log_path     = os.path.join(output_dir, log_filename)
    gen_log      = load_generation_log(log_path)

    print("\nLoading HunyuanVideo model...")
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"Model path not found: {models_root_path}")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args
    print("Model loaded.\n")

    completed, skipped, failed = 0, 0, 0
    total_gen_time = 0.0
    total_videos   = len(prompts) * len(modes)

    timing_logs = {}  # mode_name -> list of {prompt_index, time_seconds}

    for prompt_idx, entry in enumerate(prompts):
        prompt     = entry["prompt_en"]
        global_idx = start_idx + prompt_idx

        for mode_idx, mode in enumerate(modes):
            mode_name  = mode["name"]
            video_dir  = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, f"{prompt}-{seed}.mp4")
            run_num    = prompt_idx * len(modes) + mode_idx + 1

            if os.path.exists(video_path):
                logger.info(f"[{run_num}/{total_videos}] SKIP {mode_name} | {prompt[:55]}...")
                skipped += 1
                continue

            configure_magcache(hunyuan_video_sampler, mode, args.infer_steps)
            logger.info(f"[{run_num}/{total_videos}] GEN  {mode_name} | {prompt[:55]}...")

            try:
                t0 = time.time()
                outputs = hunyuan_video_sampler.predict(
                    prompt=prompt,
                    height=args.video_size[0],
                    width=args.video_size[1],
                    video_length=args.video_length,
                    seed=seed,
                    negative_prompt=args.neg_prompt,
                    infer_steps=args.infer_steps,
                    guidance_scale=args.cfg_scale,
                    num_videos_per_prompt=1,
                    flow_shift=args.flow_shift,
                    batch_size=args.batch_size,
                    embedded_guidance_scale=args.embedded_cfg_scale,
                )
                gen_time = time.time() - t0

                os.makedirs(video_dir, exist_ok=True)
                save_videos_grid(outputs["samples"][0].unsqueeze(0), video_path, fps=24)

                run_key = f"{mode_name}|{prompt}|{seed}"
                gen_log["runs"].append({
                    "prompt": prompt, "seed": seed, "mode": mode_name,
                    "time_seconds": round(gen_time, 2), "video_path": video_path,
                    "prompt_index": global_idx, "timestamp": datetime.now().isoformat(),
                })
                gen_log["completed_keys"].append(run_key)
                save_generation_log(log_path, gen_log)

                timing_logs.setdefault(mode_name, []).append({
                    "prompt_index": global_idx, "time_seconds": round(gen_time, 2),
                })

                completed      += 1
                total_gen_time += gen_time
                logger.info(f"  → saved {video_path}  ({gen_time:.1f}s)")

            except Exception as e:
                logger.error(f"  FAILED [{mode_name}] {prompt[:50]}: {e}")
                import traceback; traceback.print_exc()
                failed += 1

    # ── Save per-mode timing JSONs ────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(output_dir), "results")
    os.makedirs(results_dir, exist_ok=True)
    for mode_name, entries in timing_logs.items():
        mode_cfg_lookup = {m["name"]: m for m in modes}
        mc = mode_cfg_lookup.get(mode_name, {})
        timing_path = os.path.join(results_dir, f"timing_{mode_name}_{start_idx}-{end_idx}.json")
        with open(timing_path, "w") as f:
            json.dump({"mode": mode_name, "seed": seed,
                       "K": mc.get("K", K_FIXED),
                       "retention_ratio": mc.get("retention_ratio", RETENTION_RATIO),
                       "runs": entries}, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Done.  Completed={completed}  Skipped={skipped}  Failed={failed}")
    print(f"Total generation time: {total_gen_time:.1f}s ({total_gen_time/3600:.2f}h)")
    print("=" * 70)


if __name__ == "__main__":
    main()
