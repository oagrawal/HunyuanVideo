#!/usr/bin/env python3
"""
Batch video generation for EasyCache VBench evaluation.

Loads HunyuanVideo once, loops over prompts and 4 EasyCache modes.
Saves videos in VBench format: {prompt}-{seed}.mp4

4 modes:
  - easycache_baseline: no caching (ground truth)
  - easycache_fixed_0.025: fixed threshold 0.025
  - easycache_fixed_0.050: fixed threshold 0.050
  - easycache_adaptive: low=0.025 (steps 5-12, 43-48), high=0.050 (middle)

Resume: skips videos that exist. GPU split via --start-idx / --end-idx.
"""

import os
import sys
import time
import json
import argparse
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

MODES = [
    {"name": "easycache_baseline", "mode": "baseline"},
    {"name": "easycache_fixed_0.025", "mode": "easycache", "thresh": 0.025},
    {"name": "easycache_fixed_0.050", "mode": "easycache", "thresh": 0.050},
    {"name": "easycache_adaptive", "mode": "adaptive", "thresh_low": 0.025, "thresh_high": 0.050, "first_steps": 8, "last_steps": 6},
    {"name": "easycache_fixed_0.0375", "mode": "easycache", "thresh": 0.0375},
    {"name": "easycache_fixed_0.075", "mode": "easycache", "thresh": 0.075},
    {"name": "easycache_adaptive_0.025_0.075", "mode": "adaptive", "thresh_low": 0.025, "thresh_high": 0.075, "first_steps": 8, "last_steps": 6},
    {"name": "easycache_adaptive_0.0375_0.050", "mode": "adaptive", "thresh_low": 0.0375, "thresh_high": 0.050, "first_steps": 8, "last_steps": 6},
]


def configure_easycache(sampler, mode_cfg, infer_steps):
    """Configure EasyCache on the transformer. Call before each generation.

    IMPORTANT: State must be set on the *instance* (not the class) so that
    attributes like cnt, k, previous_raw_input, etc. are properly reset between
    runs. Setting them on the class does NOT reset instance-level attributes that
    were created by previous forward calls via `self.X = ...` / `self.X += 1`.
    The forward method itself must still be set on the class so that PyTorch's
    nn.Module.__call__ picks it up via the normal MRO.
    """
    from easycache_sample_video import easycache_forward, easycache_baseline_forward

    transformer = sampler.pipeline.transformer
    transformer_cls = transformer.__class__

    # Reset all per-run state on the INSTANCE so it shadows any stale class attrs.
    transformer.cnt = 0
    transformer.num_steps = infer_steps
    transformer.total_time = 0.0
    transformer.k = None
    transformer.previous_raw_input = None
    transformer.previous_output = None
    transformer.prev_prev_raw_input = None
    transformer.k_history = []
    transformer.pred_change_history = []
    transformer.accumulated_error_history = []

    # forward must be set on the class (nn.Module.__call__ resolves it via MRO).
    if mode_cfg["mode"] == "baseline":
        transformer_cls.forward = easycache_baseline_forward
    else:
        transformer_cls.forward = easycache_forward
        transformer.cache = None
        transformer.accumulated_error = 0.0
        transformer.ret_steps = 5
        if mode_cfg["mode"] == "adaptive":
            transformer.easycache_adaptive = True
            transformer.thresh_low = mode_cfg["thresh_low"]
            transformer.thresh_high = mode_cfg["thresh_high"]
            transformer.first_steps = mode_cfg["first_steps"]
            transformer.last_steps = mode_cfg["last_steps"]
            transformer.thresh = mode_cfg["thresh_low"]  # fallback
        else:
            transformer.easycache_adaptive = False
            transformer.thresh = mode_cfg["thresh"]
            transformer.thresh_low = mode_cfg["thresh"]
            transformer.thresh_high = mode_cfg["thresh"]


def load_generation_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_generation_log(log_path, log_data):
    tmp_path = log_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(log_data, f, indent=2)
    os.replace(tmp_path, log_path)


def main():
    batch_parser = argparse.ArgumentParser(add_help=False)
    batch_parser.add_argument("--prompts-file", type=str, default="vbench_eval/prompts_subset.json")
    batch_parser.add_argument("--output-dir", type=str, default="vbench_eval_easycache/videos")
    batch_parser.add_argument("--generation-seed", type=int, default=0)
    batch_parser.add_argument("--start-idx", type=int, default=0)
    batch_parser.add_argument("--end-idx", type=int, default=-1)
    batch_parser.add_argument("--modes", type=str, default="all",
                              help="Comma-separated modes or 'all'")
    batch_parser.add_argument("--dry-run", action="store_true")
    batch_args, remaining_argv = batch_parser.parse_known_args()

    with open(batch_args.prompts_file, "r") as f:
        all_prompts = json.load(f)
    end_idx = len(all_prompts) if batch_args.end_idx == -1 else batch_args.end_idx
    prompts = all_prompts[batch_args.start_idx:end_idx]

    if batch_args.modes == "all":
        modes = MODES
    else:
        names = [m.strip() for m in batch_args.modes.split(",")]
        modes = [m for m in MODES if m["name"] in names]
        if not modes:
            print(f"ERROR: No valid modes. Available: {[m['name'] for m in MODES]}")
            sys.exit(1)

    seed = batch_args.generation_seed
    output_dir = batch_args.output_dir
    start_idx = batch_args.start_idx
    total_videos = len(prompts) * len(modes)

    print("=" * 70)
    print("EasyCache VBench Batch Generation")
    print("=" * 70)
    print(f"Prompts: [{start_idx}, {end_idx}) = {len(prompts)}")
    print(f"Modes: {[m['name'] for m in modes]}")
    print(f"Total videos: {total_videos}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    if batch_args.dry_run:
        for entry in prompts:
            prompt = entry["prompt_en"]
            for m in modes:
                p = os.path.join(output_dir, m["name"], f"{prompt}-{seed}.mp4")
                print(f"  {'EXISTS' if os.path.exists(p) else 'NEW'} {m['name']}/{prompt}-{seed}.mp4")
        return

    # Heavy imports
    if "--save-path" not in remaining_argv:
        remaining_argv += ["--save-path", output_dir]
    sys.argv = [sys.argv[0]] + remaining_argv

    from hyvideo.config import parse_args
    from hyvideo.inference import HunyuanVideoSampler
    from hyvideo.utils.file_utils import save_videos_grid

    args = parse_args()

    log_filename = f"generation_log_{start_idx}-{end_idx}.json"
    log_path = os.path.join(output_dir, log_filename)
    gen_log = load_generation_log(log_path)

    print("\nLoading HunyuanVideo...")
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"Model path not found: {models_root_path}")
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args
    print("Model loaded.\n")

    completed, skipped, failed = 0, 0, 0
    total_gen_time = 0.0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        global_idx = start_idx + prompt_idx

        for mode_idx, mode in enumerate(modes):
            mode_name = mode["name"]
            video_dir = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, f"{prompt}-{seed}.mp4")
            run_num = prompt_idx * len(modes) + mode_idx + 1

            if os.path.exists(video_path):
                logger.info(f"[{run_num}/{total_videos}] SKIP: {mode_name} | {prompt[:50]}...")
                skipped += 1
                continue

            configure_easycache(hunyuan_video_sampler, mode, args.infer_steps)
            logger.info(f"[{run_num}/{total_videos}] Generating: {mode_name} | {prompt[:50]}...")

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
                    num_videos_per_prompt=1,
                    flow_shift=args.flow_shift,
                    batch_size=args.batch_size,
                    embedded_guidance_scale=args.embedded_cfg_scale,
                )
                gen_time = time.time() - start_time

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

                completed += 1
                total_gen_time += gen_time
                logger.info(f"  Saved {video_path} ({gen_time:.1f}s)")

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print("\n" + "=" * 70)
    print("EasyCache batch generation complete")
    print(f"Completed: {completed}  Skipped: {skipped}  Failed: {failed}")
    print(f"Total time: {total_gen_time:.1f}s ({total_gen_time/3600:.1f}h)")
    print("=" * 70)


if __name__ == "__main__":
    main()
