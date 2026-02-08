#!/usr/bin/env python3
"""
Batch video generation for VBench evaluation.

Loads the HunyuanVideo model ONCE, then loops over all prompts and modes.
Saves videos in VBench-compatible naming format: {prompt}-{seed}.mp4

Supports:
  - Resume: skips videos that already exist
  - GPU splitting: --start-idx / --end-idx to split prompts across GPUs
  - All 4 modes: baseline, fixed 0.1, fixed 0.2, adaptive

Usage (inside Docker container):
    # GPU 0: first half of prompts
    python3 vbench_eval/batch_generate.py \\
        --video-size 544 960 --video-length 129 --infer-steps 50 \\
        --flow-reverse --use-cpu-offload \\
        --start-idx 0 --end-idx 28

    # GPU 1: second half of prompts
    python3 vbench_eval/batch_generate.py \\
        --video-size 544 960 --video-length 129 --infer-steps 50 \\
        --flow-reverse --use-cpu-offload \\
        --start-idx 28 --end-idx 55
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    from loguru import logger
except ImportError:
    # Fallback logger for dry-run outside Docker
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

from typing import Optional, Union, Dict


def _import_heavy():
    """Import heavy dependencies. Called only when actually generating."""
    global np, torch, modulate, get_cu_seqlens
    import numpy as np
    import torch
    from hyvideo.modules.modulate_layers import modulate
    from hyvideo.modules.attenion import get_cu_seqlens


# --- TeaCache forward (reused from teacache_sample_video.py) ---

def teacache_forward(
        self,
        x,       # torch.Tensor
        t,       # torch.Tensor
        text_states=None,
        text_mask=None,
        text_states_2=None,
        freqs_cos=None,
        freqs_sin=None,
        guidance=None,
        return_dict: bool = True,
    ):
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        vec = self.time_in(t)
        vec = vec + self.vector_in(text_states_2)

        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(guidance)

        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        if self.enable_teacache:
            inp = img.clone()
            vec_ = vec.clone()
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
            normed_inp = self.double_blocks[0].img_norm1(inp)
            modulated_inp = modulate(
                normed_inp, shift=img_mod1_shift, scale=img_mod1_scale
            )

            if self.cnt == 0 or self.cnt == self.num_steps - 1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:
                coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                rescale_func = np.poly1d(coefficients)
                delta = rescale_func(((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                self.accumulated_rel_l1_distance += delta

                if self.cnt <= self.first_steps or self.cnt >= self.num_steps - self.last_steps:
                    current_thresh = self.rel_l1_thresh_low
                else:
                    current_thresh = self.rel_l1_thresh_high

                if self.accumulated_rel_l1_distance < current_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0

            self.previous_modulated_input = modulated_inp
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0

        if self.enable_teacache:
            if not should_calc:
                img += self.previous_residual
            else:
                ori_img = img.clone()
                for _, block in enumerate(self.double_blocks):
                    double_block_args = [
                        img, txt, vec,
                        cu_seqlens_q, cu_seqlens_kv,
                        max_seqlen_q, max_seqlen_kv,
                        freqs_cis,
                    ]
                    img, txt = block(*double_block_args)

                x = torch.cat((img, txt), 1)
                if len(self.single_blocks) > 0:
                    for _, block in enumerate(self.single_blocks):
                        single_block_args = [
                            x, vec, txt_seq_len,
                            cu_seqlens_q, cu_seqlens_kv,
                            max_seqlen_q, max_seqlen_kv,
                            (freqs_cos, freqs_sin),
                        ]
                        x = block(*single_block_args)

                img = x[:, :img_seq_len, ...]
                self.previous_residual = img - ori_img
        else:
            for _, block in enumerate(self.double_blocks):
                double_block_args = [
                    img, txt, vec,
                    cu_seqlens_q, cu_seqlens_kv,
                    max_seqlen_q, max_seqlen_kv,
                    freqs_cis,
                ]
                img, txt = block(*double_block_args)

            x = torch.cat((img, txt), 1)
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x, vec, txt_seq_len,
                        cu_seqlens_q, cu_seqlens_kv,
                        max_seqlen_q, max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                    ]
                    x = block(*single_block_args)

            img = x[:, :img_seq_len, ...]

        img = self.final_layer(img, vec)
        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img


# --- Mode configurations ---

MODES = [
    {
        "name": "hunyuan_baseline",
        "teacache": False,
    },
    {
        "name": "hunyuan_fixed_0.1",
        "teacache": True,
        "thresh_low": 0.1,
        "thresh_high": 0.1,
        "first_steps": 5,
        "last_steps": 10,
    },
    {
        "name": "hunyuan_fixed_0.2",
        "teacache": True,
        "thresh_low": 0.2,
        "thresh_high": 0.2,
        "first_steps": 5,
        "last_steps": 10,
    },
    {
        "name": "hunyuan_adaptive",
        "teacache": True,
        "thresh_low": 0.1,
        "thresh_high": 0.2,
        "first_steps": 5,
        "last_steps": 10,
    },
]


def configure_teacache(sampler, mode_cfg, infer_steps):
    """Configure TeaCache mode on the transformer. Call before each generation."""
    transformer_cls = sampler.pipeline.transformer.__class__

    if not mode_cfg["teacache"]:
        # Baseline: disable TeaCache
        transformer_cls.enable_teacache = False
    else:
        # Enable TeaCache with specified thresholds
        transformer_cls.enable_teacache = True
        transformer_cls.forward = teacache_forward
        transformer_cls.num_steps = infer_steps
        transformer_cls.rel_l1_thresh_low = mode_cfg["thresh_low"]
        transformer_cls.rel_l1_thresh_high = mode_cfg["thresh_high"]
        transformer_cls.first_steps = mode_cfg["first_steps"]
        transformer_cls.last_steps = mode_cfg["last_steps"]


def reset_teacache_state(sampler):
    """Reset TeaCache per-run state. MUST call before each video generation."""
    transformer_cls = sampler.pipeline.transformer.__class__
    transformer_cls.cnt = 0
    transformer_cls.accumulated_rel_l1_distance = 0
    transformer_cls.previous_modulated_input = None
    transformer_cls.previous_residual = None


def load_generation_log(log_path):
    """Load existing generation log for resume support."""
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_generation_log(log_path, log_data):
    """Save generation log (atomic write via temp file)."""
    tmp_path = log_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(log_data, f, indent=2)
    os.replace(tmp_path, log_path)


def main():
    # ---- Parse batch-specific args first, pass the rest to hyvideo ----
    batch_parser = argparse.ArgumentParser(add_help=False)
    batch_parser.add_argument("--prompts-file", type=str,
                              default="vbench_eval/prompts_subset.json",
                              help="Path to VBench prompts JSON file")
    batch_parser.add_argument("--output-dir", type=str,
                              default="vbench_eval/videos",
                              help="Base output directory for videos")
    batch_parser.add_argument("--generation-seed", type=int, default=0,
                              help="Seed for video generation (VBench expects 0-4)")
    batch_parser.add_argument("--start-idx", type=int, default=0,
                              help="Start prompt index (inclusive, for GPU splitting)")
    batch_parser.add_argument("--end-idx", type=int, default=-1,
                              help="End prompt index (exclusive, -1 = all)")
    batch_parser.add_argument("--modes", type=str, default="all",
                              help="Comma-separated modes to run, or 'all'. "
                                   "Options: hunyuan_baseline, hunyuan_fixed_0.1, "
                                   "hunyuan_fixed_0.2, hunyuan_adaptive")
    batch_parser.add_argument("--dry-run", action="store_true",
                              help="Print what would be generated without running")
    batch_args, remaining_argv = batch_parser.parse_known_args()

    # ---- Load prompts ----
    with open(batch_args.prompts_file, "r") as f:
        all_prompts = json.load(f)

    # Apply prompt range
    end_idx = len(all_prompts) if batch_args.end_idx == -1 else batch_args.end_idx
    start_idx = batch_args.start_idx
    prompts = all_prompts[start_idx:end_idx]

    # Filter modes
    if batch_args.modes == "all":
        modes = MODES
    else:
        mode_names = [m.strip() for m in batch_args.modes.split(",")]
        modes = [m for m in MODES if m["name"] in mode_names]
        if not modes:
            print(f"ERROR: No valid modes found in '{batch_args.modes}'")
            print(f"Available: {[m['name'] for m in MODES]}")
            sys.exit(1)

    seed = batch_args.generation_seed
    output_dir = batch_args.output_dir

    # ---- Summary ----
    total_videos = len(prompts) * len(modes)
    print("=" * 70)
    print("VBench Batch Video Generation")
    print("=" * 70)
    print(f"Prompts file:  {batch_args.prompts_file}")
    print(f"Prompt range:  [{start_idx}, {end_idx}) = {len(prompts)} prompts")
    print(f"Seed:          {seed}")
    print(f"Modes:         {[m['name'] for m in modes]}")
    print(f"Total videos:  {total_videos}")
    print(f"Output dir:    {output_dir}")
    print("=" * 70)

    # ---- Dry run (no heavy imports needed) ----
    if batch_args.dry_run:
        print("\n[DRY RUN] Would generate these videos:\n")
        for i, entry in enumerate(prompts):
            prompt = entry["prompt_en"]
            for mode in modes:
                filename = f"{prompt}-{seed}.mp4"
                filepath = os.path.join(output_dir, mode["name"], filename)
                exists = os.path.exists(filepath)
                status = "EXISTS" if exists else "NEW"
                print(f"  [{status}] {mode['name']}/{filename}")
        print(f"\nTotal: {total_videos} videos")
        existing = sum(1 for entry in prompts for mode in modes
                       if os.path.exists(os.path.join(output_dir, mode["name"],
                                                      f"{entry['prompt_en']}-{seed}.mp4")))
        print(f"Already exist: {existing}")
        print(f"To generate: {total_videos - existing}")
        return

    # ---- Heavy imports (only for actual generation) ----
    _import_heavy()

    # Pass remaining args to hyvideo's parser
    sys.argv = [sys.argv[0]] + remaining_argv

    from hyvideo.config import parse_args
    from hyvideo.inference import HunyuanVideoSampler
    from hyvideo.utils.file_utils import save_videos_grid

    args = parse_args()

    print(f"Video config:  {args.video_size[0]}x{args.video_size[1]}, "
          f"{args.video_length} frames, {args.infer_steps} steps")

    # ---- Generation log ----
    log_path = os.path.join(output_dir, "generation_log.json")
    gen_log = load_generation_log(log_path)

    # ---- Load model ONCE ----
    print("\nLoading HunyuanVideo model (this takes a few minutes)...")
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args
    print("Model loaded successfully.\n")

    # ---- Main generation loop ----
    completed = 0
    skipped = 0
    failed = 0
    total_gen_time = 0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        global_idx = start_idx + prompt_idx
        dims = ", ".join(entry["dimension"])

        for mode_idx, mode in enumerate(modes):
            mode_name = mode["name"]
            video_filename = f"{prompt}-{seed}.mp4"
            video_dir = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, video_filename)

            run_key = f"{mode_name}|{prompt}|{seed}"
            run_num = prompt_idx * len(modes) + mode_idx + 1

            # ---- Resume: skip if already exists ----
            if os.path.exists(video_path):
                logger.info(f"[{run_num}/{total_videos}] SKIP (exists): {mode_name} | {prompt[:50]}...")
                skipped += 1
                continue

            # ---- Configure TeaCache for this mode ----
            configure_teacache(hunyuan_video_sampler, mode, args.infer_steps)
            reset_teacache_state(hunyuan_video_sampler)

            # ---- Generate ----
            logger.info(f"[{run_num}/{total_videos}] Generating: {mode_name} | {prompt[:50]}...")
            logger.info(f"  Prompt #{global_idx} | Dims: {dims}")

            try:
                start_time = time.time()
                outputs = hunyuan_video_sampler.predict(
                    prompt=prompt,
                    height=args.video_size[0],
                    width=args.video_size[1],
                    video_length=args.video_length,
                    seed=seed,
                    negative_prompt=args.neg_prompt,
                    infer_steps=args.infer_steps,
                    guidance_scale=args.cfg_scale,
                    num_videos_per_prompt=args.num_videos,
                    flow_shift=args.flow_shift,
                    batch_size=args.batch_size,
                    embedded_guidance_scale=args.embedded_cfg_scale,
                )
                gen_time = time.time() - start_time

                # ---- Save video ----
                samples = outputs["samples"]
                os.makedirs(video_dir, exist_ok=True)
                sample = samples[0].unsqueeze(0)
                save_videos_grid(sample, video_path, fps=24)

                logger.info(f"  Saved: {video_path}")
                logger.info(f"  Time: {gen_time:.1f}s ({gen_time/60:.1f}min)")

                # ---- Log result ----
                run_entry = {
                    "prompt": prompt,
                    "seed": seed,
                    "mode": mode_name,
                    "dimensions": entry["dimension"],
                    "time_seconds": round(gen_time, 2),
                    "video_path": video_path,
                    "timestamp": datetime.now().isoformat(),
                    "prompt_index": global_idx,
                }
                gen_log["runs"].append(run_entry)
                gen_log["completed_keys"].append(run_key)
                save_generation_log(log_path, gen_log)

                completed += 1
                total_gen_time += gen_time

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

                # Log failure
                run_entry = {
                    "prompt": prompt,
                    "seed": seed,
                    "mode": mode_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "prompt_index": global_idx,
                }
                gen_log["runs"].append(run_entry)
                save_generation_log(log_path, gen_log)

    # ---- Final summary ----
    print("\n" + "=" * 70)
    print("BATCH GENERATION COMPLETE")
    print("=" * 70)
    print(f"Completed:  {completed}")
    print(f"Skipped:    {skipped} (already existed)")
    print(f"Failed:     {failed}")
    print(f"Total generation time: {total_gen_time:.1f}s ({total_gen_time/3600:.1f}h)")
    if completed > 0:
        print(f"Avg time per video: {total_gen_time/completed:.1f}s ({total_gen_time/completed/60:.1f}min)")
    print(f"Generation log: {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
