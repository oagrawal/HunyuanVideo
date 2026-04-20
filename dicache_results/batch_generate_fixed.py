#!/usr/bin/env python3
"""
Batch generation for DiCache fixed-threshold runs on HunyuanVideo.

Outputs:
  videos:  <output-dir>/<mode-name>/<prompt>-<seed>.mp4
  timing:  <results-dir>/timing_<mode-name>_<start>-<end>.json
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

from run_hunyuanvideo_dicache import dicache_forward


def load_generation_log(log_path: str):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_generation_log(log_path: str, log_data):
    tmp = log_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log_data, f, indent=2)
    os.replace(tmp, log_path)


def configure_dicache(transformer, infer_steps: int, delta, ret_ratio: float, probe_depth: int):
    cls = transformer.__class__
    transformer.cnt = 0
    transformer.probe_depth = probe_depth
    transformer.num_steps = infer_steps
    transformer.rel_l1_thresh = delta
    transformer.ret_ratio = ret_ratio
    transformer.accumulated_rel_l1_distance = 0
    transformer.residual_cache = None
    transformer.probe_residual_cache = None
    transformer.residual_window = []
    transformer.probe_residual_window = []
    transformer.resume_flag = False
    transformer.previous_input = None
    transformer.previous_probe_img = None
    transformer.trace_enabled = False
    transformer.dicache_trace_only = False
    cls.forward = dicache_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", type=str, default="vbench_eval/prompts_subset.json")
    parser.add_argument("--output-dir", type=str, default="dicache_results/videos")
    parser.add_argument("--results-dir", type=str, default="dicache_results/results")
    parser.add_argument("--mode-name", type=str, default="dicache_fixed_0.60")
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.60)
    parser.add_argument("--ret-ratio", type=float, default=0.0)
    parser.add_argument("--probe-depth", type=int, default=1)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    # Adaptive schedule args (alternative to --delta for a single threshold)
    parser.add_argument("--delta-low", type=float, default=None,
                        help="LOW threshold for volatile steps (0–stable_start-1 and stable_end–end). Overrides --delta when set.")
    parser.add_argument("--delta-high", type=float, default=None,
                        help="HIGH threshold for stable middle steps (stable_start–stable_end-1).")
    parser.add_argument("--stable-start", type=int, default=8,
                        help="First step of stable (high-threshold) region (default: 8).")
    parser.add_argument("--stable-end", type=int, default=40,
                        help="First step of final volatile (low-threshold) region (default: 40).")
    args, remaining_argv = parser.parse_known_args()

    with open(args.prompts_file, "r") as f:
        all_prompts = json.load(f)
    end_idx = len(all_prompts) if args.end_idx == -1 else args.end_idx
    prompts = all_prompts[args.start_idx:end_idx]

    # Build threshold schedule (list overrides scalar --delta when adaptive args provided)
    if args.delta_low is not None and args.delta_high is not None:
        n = 50  # infer_steps; will be overridden by hv_args after model load, but 50 is standard
        delta_eff = [args.delta_low if (t < args.stable_start or t >= args.stable_end)
                     else args.delta_high for t in range(n)]
        delta_label = f"adaptive low={args.delta_low}/high={args.delta_high} stable=[{args.stable_start},{args.stable_end})"
    else:
        delta_eff = args.delta
        delta_label = str(args.delta)

    print("=" * 70)
    print("DiCache fixed-threshold batch generation")
    print("=" * 70)
    print(f"Prompts [{args.start_idx}, {end_idx}): {len(prompts)}")
    print(f"Mode: {args.mode_name} | delta={delta_label} | ret_ratio={args.ret_ratio} | probe_depth={args.probe_depth}")
    print(f"Seed: {args.generation_seed}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 70)

    if args.dry_run:
        for item in prompts:
            p = item["prompt_en"]
            out = os.path.join(args.output_dir, args.mode_name, f"{p}-{args.generation_seed}.mp4")
            print(("EXISTS" if os.path.exists(out) else "NEW").ljust(8), out)
        return

    if "--save-path" not in remaining_argv:
        remaining_argv += ["--save-path", args.output_dir]
    sys.argv = [sys.argv[0]] + remaining_argv

    from hyvideo.config import parse_args
    from hyvideo.inference import HunyuanVideoSampler
    from hyvideo.utils.file_utils import save_videos_grid

    hv_args = parse_args()
    models_root_path = Path(hv_args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"Model path not found: {models_root_path}")

    os.makedirs(os.path.join(args.output_dir, args.mode_name), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"generation_log_{args.mode_name}_{args.start_idx}-{end_idx}.json")
    gen_log = load_generation_log(log_path)

    print("Loading HunyuanVideo model once...")
    sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=hv_args)
    hv_args = sampler.args
    print("Model loaded.")

    timing_runs = []
    total = len(prompts)
    completed = skipped = failed = 0

    for i, item in enumerate(prompts):
        prompt = item["prompt_en"]
        global_idx = args.start_idx + i
        video_path = os.path.join(args.output_dir, args.mode_name, f"{prompt}-{args.generation_seed}.mp4")
        if os.path.exists(video_path):
            print(f"[{i+1}/{total}] SKIP {global_idx:02d} {prompt[:60]}")
            skipped += 1
            continue

        # Rebuild schedule with actual infer_steps now that model is loaded
        if args.delta_low is not None and args.delta_high is not None:
            delta_eff = [args.delta_low if (t < args.stable_start or t >= args.stable_end)
                         else args.delta_high for t in range(hv_args.infer_steps)]
        configure_dicache(
            sampler.pipeline.transformer,
            infer_steps=hv_args.infer_steps,
            delta=delta_eff,
            ret_ratio=args.ret_ratio,
            probe_depth=args.probe_depth,
        )

        print(f"[{i+1}/{total}] GEN  {global_idx:02d} {prompt[:60]}")
        try:
            t0 = time.time()
            outputs = sampler.predict(
                prompt=prompt,
                height=hv_args.video_size[0],
                width=hv_args.video_size[1],
                video_length=hv_args.video_length,
                seed=args.generation_seed,
                negative_prompt=hv_args.neg_prompt,
                infer_steps=hv_args.infer_steps,
                guidance_scale=hv_args.cfg_scale,
                num_videos_per_prompt=1,
                flow_shift=hv_args.flow_shift,
                batch_size=hv_args.batch_size,
                embedded_guidance_scale=hv_args.embedded_cfg_scale,
            )
            elapsed = time.time() - t0
            save_videos_grid(outputs["samples"][0].unsqueeze(0), video_path, fps=24)

            run_key = f"{args.mode_name}|{prompt}|{args.generation_seed}"
            gen_log["runs"].append(
                {
                    "prompt": prompt,
                    "seed": args.generation_seed,
                    "mode": args.mode_name,
                    "time_seconds": round(elapsed, 2),
                    "video_path": video_path,
                    "prompt_index": global_idx,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            gen_log["completed_keys"].append(run_key)
            save_generation_log(log_path, gen_log)

            timing_runs.append({"prompt_index": global_idx, "time_seconds": round(elapsed, 2)})
            completed += 1
            print(f"  -> saved {video_path} ({elapsed:.1f}s)")
        except Exception as e:
            failed += 1
            print(f"  FAILED [{global_idx}] {prompt[:50]}: {e}")

    timing_path = os.path.join(args.results_dir, f"timing_{args.mode_name}_{args.start_idx}-{end_idx}.json")
    with open(timing_path, "w") as f:
        json.dump(
            {
                "mode": args.mode_name,
                "seed": args.generation_seed,
                "delta": delta_label,
                "ret_ratio": args.ret_ratio,
                "probe_depth": args.probe_depth,
                "runs": timing_runs,
            },
            f,
            indent=2,
        )

    print("=" * 70)
    print(f"Done. Completed={completed} Skipped={skipped} Failed={failed}")
    print(f"Timing JSON: {timing_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

