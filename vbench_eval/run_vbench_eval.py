#!/usr/bin/env python3
"""
Run VBench evaluation on generated videos.

Evaluates all 16 VBench dimensions for each mode's video directory.
Uses vbench_standard mode with the prompts_subset.json file.

IMPORTANT: Requires transformers==4.33.2 (switch before running)
    pip install transformers==4.33.2

Usage (inside Docker container):
    # Evaluate all modes
    python3 vbench_eval/run_vbench_eval.py

    # Evaluate a specific mode
    python3 vbench_eval/run_vbench_eval.py --modes hunyuan_baseline

    # Use a custom VBench info file
    python3 vbench_eval/run_vbench_eval.py --full-info vbench_eval/prompts_subset.json
"""

import argparse
import os
import sys

import torch
from vbench import VBench

DIMENSIONS = [
    "subject_consistency",
    "imaging_quality",
    "background_consistency",
    "motion_smoothness",
    "overall_consistency",
    "human_action",
    "multiple_objects",
    "spatial_relationship",
    "object_class",
    "color",
    "aesthetic_quality",
    "appearance_style",
    "temporal_flickering",
    "scene",
    "temporal_style",
    "dynamic_degree",
]

ALL_MODES = [
    "hunyuan_baseline",
    "hunyuan_fixed_0.1",
    "hunyuan_fixed_0.2",
    "hunyuan_adaptive",
]


def main():
    parser = argparse.ArgumentParser(description="Run VBench evaluation on generated videos")
    parser.add_argument("--video-dir", type=str, default="vbench_eval/videos",
                        help="Base directory containing mode subdirectories with videos")
    parser.add_argument("--save-dir", type=str, default="vbench_eval/vbench_scores",
                        help="Base directory to save VBench scores")
    parser.add_argument("--full-info", type=str, default="vbench_eval/prompts_subset.json",
                        help="Path to VBench full info JSON (use subset for fewer warnings)")
    parser.add_argument("--modes", type=str, default="all",
                        help="Comma-separated modes to evaluate, or 'all'")
    parser.add_argument("--dimensions", type=str, default="all",
                        help="Comma-separated dimensions, or 'all'")
    args = parser.parse_args()

    # Resolve modes
    if args.modes == "all":
        modes = ALL_MODES
    else:
        modes = [m.strip() for m in args.modes.split(",")]

    # Resolve dimensions
    if args.dimensions == "all":
        dimensions = DIMENSIONS
    else:
        dimensions = [d.strip() for d in args.dimensions.split(",")]

    print("=" * 70)
    print("VBench Evaluation")
    print("=" * 70)
    print(f"Video dir:    {args.video_dir}")
    print(f"Save dir:     {args.save_dir}")
    print(f"Full info:    {args.full_info}")
    print(f"Modes:        {modes}")
    print(f"Dimensions:   {len(dimensions)} dimensions")
    print("=" * 70)

    # Validate paths
    if not os.path.exists(args.full_info):
        print(f"ERROR: Full info file not found: {args.full_info}")
        sys.exit(1)

    for mode in modes:
        video_path = os.path.join(args.video_dir, mode)
        if not os.path.exists(video_path):
            print(f"WARNING: Video directory not found: {video_path}")
            continue

        # Count videos
        videos = [f for f in os.listdir(video_path) if f.endswith(".mp4")]
        if not videos:
            print(f"WARNING: No videos found in {video_path}")
            continue

        save_path = os.path.join(args.save_dir, mode)
        os.makedirs(save_path, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"Evaluating: {mode} ({len(videos)} videos)")
        print(f"{'='*70}")

        kwargs = {}
        kwargs["imaging_quality_preprocessing_mode"] = "longer"

        # Check which dimensions are already evaluated (for resume)
        remaining_dims = []
        skipped_dims = []
        for dimension in dimensions:
            result_file = os.path.join(save_path, f"{dimension}_eval_results.json")
            if os.path.exists(result_file):
                skipped_dims.append(dimension)
            else:
                remaining_dims.append(dimension)

        if skipped_dims:
            print(f"  Skipping {len(skipped_dims)} already-evaluated dimensions: {skipped_dims}")
        if not remaining_dims:
            print(f"  All dimensions already evaluated for {mode}. Skipping.")
            continue

        print(f"  Evaluating {len(remaining_dims)}/{len(dimensions)} dimensions")

        for dim_idx, dimension in enumerate(remaining_dims, 1):
            print(f"\n  [{dim_idx}/{len(remaining_dims)}] {dimension}...")

            try:
                my_VBench = VBench(torch.device("cuda"), args.full_info, save_path)
                my_VBench.evaluate(
                    videos_path=video_path,
                    name=dimension,
                    local=False,
                    read_frame=False,
                    dimension_list=[dimension],
                    mode="vbench_standard",
                    **kwargs,
                )
                print(f"    Done.")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n  Scores saved to: {save_path}")

    print(f"\n{'='*70}")
    print("VBench evaluation complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
