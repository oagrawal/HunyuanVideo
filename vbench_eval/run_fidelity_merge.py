#!/usr/bin/env python3
"""
Merge partial fidelity metrics from multi-GPU runs into a single result file.

Each partial file (from run_fidelity_metrics.py --partial-output) contains raw
per-video PSNR/SSIM/LPIPS scores. This script concatenates them, computes
mean/std, and writes the final {mode}_vs_{baseline}.json plus all_fidelity_results.json.

Usage (inside Docker or host):
    python3 vbench_eval/run_fidelity_merge.py \
      --save-dir /path/to/fidelity_metrics_teacache \
      --mode mochi_adaptive_f34s14l16 \
      --baseline mochi_diff_baseline

Expects partial files: {save_dir}/partial_{mode}_vs_{baseline}_0.json, _1.json, _2.json, _3.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Merge partial fidelity results from multi-GPU runs"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory containing partial JSON files and where final output is written",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Mode name (e.g. mochi_adaptive_f34s14l16)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline mode name (e.g. mochi_diff_baseline)",
    )
    parser.add_argument(
        "--partial-pattern",
        type=str,
        default=None,
        help="Glob pattern for partial files (default: {save_dir}/partial_{mode}_vs_{baseline}_*.json)",
    )
    args = parser.parse_args()

    pattern = args.partial_pattern
    if pattern is None:
        safe_mode = args.mode.replace("/", "_")
        safe_baseline = args.baseline.replace("/", "_")
        pattern = os.path.join(
            args.save_dir,
            f"partial_{safe_mode}_vs_{safe_baseline}_*.json",
        )

    partial_files = sorted(glob.glob(pattern))
    if not partial_files:
        print(f"ERROR: No partial files found for pattern: {pattern}")
        sys.exit(1)

    print(f"Merging {len(partial_files)} partial file(s):")
    for f in partial_files:
        print(f"  {f}")

    psnr_all = []
    ssim_all = []
    lpips_all = []
    total_videos = 0

    for path in partial_files:
        with open(path, "r") as f:
            data = json.load(f)
        psnr_all.extend(data["psnr_scores"])
        ssim_all.extend(data["ssim_scores"])
        lpips_all.extend(data["lpips_scores"])
        total_videos += data["num_videos"]

    result = {
        "mode": args.mode,
        "baseline": args.baseline,
        "num_videos": total_videos,
        "psnr": {
            "mean": float(np.mean(psnr_all)),
            "std": float(np.std(psnr_all)),
        },
        "ssim": {
            "mean": float(np.mean(ssim_all)),
            "std": float(np.std(ssim_all)),
        },
        "lpips": {
            "mean": float(np.mean(lpips_all)),
            "std": float(np.std(lpips_all)),
        },
    }

    os.makedirs(args.save_dir, exist_ok=True)
    final_path = os.path.join(args.save_dir, f"{args.mode}_vs_{args.baseline}.json")
    with open(final_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nMerged result saved to: {final_path}")
    print(f"  num_videos: {result['num_videos']}")
    print(f"  PSNR:  {result['psnr']['mean']:.4f} +/- {result['psnr']['std']:.4f}")
    print(f"  SSIM:  {result['ssim']['mean']:.4f} +/- {result['ssim']['std']:.4f}")
    print(f"  LPIPS: {result['lpips']['mean']:.4f} +/- {result['lpips']['std']:.4f}")

    # Update all_fidelity_results.json if it exists (add/update this mode)
    all_path = os.path.join(args.save_dir, "all_fidelity_results.json")
    if os.path.exists(all_path):
        with open(all_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results[args.mode] = result
    with open(all_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nUpdated {all_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
