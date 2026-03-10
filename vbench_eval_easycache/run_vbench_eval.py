#!/usr/bin/env python3
"""Run VBench evaluation on EasyCache generated videos."""

import argparse
import os
import sys
import torch
from vbench import VBench

DIMENSIONS = [
    "subject_consistency", "imaging_quality", "background_consistency",
    "motion_smoothness", "overall_consistency", "human_action",
    "multiple_objects", "spatial_relationship", "object_class", "color",
    "aesthetic_quality", "appearance_style", "temporal_flickering",
    "scene", "temporal_style", "dynamic_degree",
]

ALL_MODES = [
    "easycache_baseline",
    "easycache_fixed_0.025",
    "easycache_fixed_0.050",
    "easycache_adaptive",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", default="vbench_eval_easycache/videos")
    p.add_argument("--save-dir", default="vbench_eval_easycache/vbench_scores")
    p.add_argument("--full-info", default="vbench_eval/prompts_subset.json")
    p.add_argument("--modes", default="all")
    p.add_argument("--dimensions", default="all")
    args = p.parse_args()

    modes = ALL_MODES if args.modes == "all" else [m.strip() for m in args.modes.split(",")]
    dims = DIMENSIONS if args.dimensions == "all" else [d.strip() for d in args.dimensions.split(",")]

    print("=" * 70)
    print("EasyCache VBench Evaluation")
    print(f"Video dir: {args.video_dir}  Save dir: {args.save_dir}")
    print(f"Modes: {modes}  Dimensions: {len(dims)}")
    print("=" * 70)

    for mode in modes:
        video_path = os.path.join(args.video_dir, mode)
        if not os.path.exists(video_path):
            print(f"WARNING: {video_path} not found, skipping")
            continue
        save_path = os.path.join(args.save_dir, mode)
        os.makedirs(save_path, exist_ok=True)

        remaining = [d for d in dims if not os.path.exists(os.path.join(save_path, f"{d}_eval_results.json"))]
        if not remaining:
            print(f"  {mode}: all dimensions done")
            continue

        print(f"\nEvaluating {mode} ({len(remaining)} dims)...")
        for i, dimension in enumerate(remaining, 1):
            print(f"  [{i}/{len(remaining)}] {dimension}...")
            try:
                vbench = VBench(torch.device("cuda"), args.full_info, save_path)
                vbench.evaluate(
                    videos_path=video_path,
                    name=dimension,
                    local=False,
                    read_frame=False,
                    dimension_list=[dimension],
                    mode="vbench_standard",
                    imaging_quality_preprocessing_mode="longer",
                )
            except Exception as e:
                print(f"    ERROR: {e}")

    print("\nEasyCache VBench evaluation complete.")


if __name__ == "__main__":
    main()
