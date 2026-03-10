#!/usr/bin/env python3
"""
Compute fidelity metrics (PSNR, SSIM, LPIPS) comparing cached videos to baseline.

Compares each cached mode against the no-cache baseline, video by video
(same prompt + same seed = paired comparison).

Based on TeaCache's common_metrics evaluation code.

Usage (inside Docker container):
    python3 vbench_eval/run_fidelity_metrics.py

    # Compare only specific modes
    python3 vbench_eval/run_fidelity_metrics.py --modes hunyuan_fixed_0.1,hunyuan_fixed_0.2
"""

import argparse
import json
import math
import os
import sys

import cv2
import imageio
import lpips
import numpy as np
import torch
import torchvision.transforms.functional as F
import tqdm


# ---- Metric functions (from TeaCache's common_metrics) ----

def img_psnr(img1, img2):
    """Compute PSNR between two images in [0,1] range."""
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100
    return 20 * math.log10(1 / math.sqrt(mse))


def ssim_single(img1, img2):
    """Compute SSIM for a single channel."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def img_ssim(img1, img2):
    """Compute SSIM for a multi-channel image. Input: (C, H, W) numpy arrays."""
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim_single(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            return np.mean([ssim_single(img1[i], img2[i]) for i in range(3)])
        elif img1.shape[0] == 1:
            return ssim_single(np.squeeze(img1), np.squeeze(img2))
    raise ValueError("Wrong input image dimensions.")


# ---- Video loading ----

def load_video(video_path):
    """Load video as tensor of shape (T, C, H, W) with uint8 values."""
    reader = imageio.get_reader(video_path, "ffmpeg")
    frames = []
    for frame in reader:
        frame_tensor = torch.tensor(frame).permute(2, 0, 1)  # (C, H, W)
        frames.append(frame_tensor)
    reader.close()
    return torch.stack(frames)  # (T, C, H, W)


# ---- Main ----

CACHED_MODES = [
    "hunyuan_fixed_0.1",
    "hunyuan_fixed_0.2",
    "hunyuan_adaptive",
]


def main():
    parser = argparse.ArgumentParser(description="Compute fidelity metrics vs baseline")
    parser.add_argument("--video-dir", type=str, default="vbench_eval/videos",
                        help="Base directory containing mode subdirectories")
    parser.add_argument("--baseline", type=str, default="hunyuan_baseline",
                        help="Baseline mode name (ground truth)")
    parser.add_argument("--modes", type=str, default="all",
                        help="Comma-separated cached modes to compare, or 'all'")
    parser.add_argument("--save-dir", type=str, default="vbench_eval/fidelity_metrics",
                        help="Directory to save results")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start video index (inclusive, for multi-GPU split)")
    parser.add_argument("--end-idx", type=int, default=-1,
                        help="End video index (exclusive, -1 = all)")
    parser.add_argument("--partial-output", type=str, default=None,
                        help="Path for partial output (raw scores for merge). When set, writes partial JSON instead of final aggregated result.")
    args = parser.parse_args()

    if args.modes == "all":
        modes = CACHED_MODES
    else:
        modes = [m.strip() for m in args.modes.split(",")]

    baseline_dir = os.path.join(args.video_dir, args.baseline)
    if not os.path.exists(baseline_dir):
        print(f"ERROR: Baseline directory not found: {baseline_dir}")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    # Get baseline video list
    baseline_videos = sorted([f for f in os.listdir(baseline_dir) if f.endswith(".mp4")])
    print(f"Baseline: {baseline_dir} ({len(baseline_videos)} videos)")

    # Initialize LPIPS model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(device)

    all_results = {}

    for mode in modes:
        mode_dir = os.path.join(args.video_dir, mode)
        if not os.path.exists(mode_dir):
            print(f"WARNING: Mode directory not found: {mode_dir}, skipping.")
            continue

        mode_videos = sorted([f for f in os.listdir(mode_dir) if f.endswith(".mp4")])

        # Find common videos (present in both baseline and this mode)
        common = sorted(set(baseline_videos) & set(mode_videos))
        if not common:
            print(f"WARNING: No matching videos between baseline and {mode}")
            continue

        # Slice for multi-GPU split
        end_idx = len(common) if args.end_idx == -1 else min(args.end_idx, len(common))
        start_idx = max(0, args.start_idx)
        common_slice = common[start_idx:end_idx]
        if not common_slice:
            print(f"WARNING: No videos in range [{start_idx}, {end_idx}) for {mode}")
            continue

        print(f"\n{'='*70}")
        print(f"Comparing: {args.baseline} vs {mode} (videos {start_idx}-{end_idx-1} of {len(common)} = {len(common_slice)} videos)")
        print(f"{'='*70}")

        lpips_scores = []
        psnr_scores = []
        ssim_scores = []

        for video_name in tqdm.tqdm(common_slice, desc=mode):
            gt_path = os.path.join(baseline_dir, video_name)
            gen_path = os.path.join(mode_dir, video_name)

            try:
                gt_video = load_video(gt_path)    # (T, C, H, W) uint8
                gen_video = load_video(gen_path)   # (T, C, H, W) uint8

                # Ensure same number of frames
                T = min(gt_video.shape[0], gen_video.shape[0])
                gt_video = gt_video[:T]
                gen_video = gen_video[:T]

                # Normalize to [0, 1]
                gt_float = gt_video.float() / 255.0
                gen_float = gen_video.float() / 255.0

                # Per-frame metrics
                frame_psnr = []
                frame_ssim = []
                frame_lpips = []

                for t in range(T):
                    gt_frame = gt_float[t]    # (C, H, W)
                    gen_frame = gen_float[t]   # (C, H, W)

                    # PSNR
                    frame_psnr.append(img_psnr(gt_frame.numpy(), gen_frame.numpy()))

                    # SSIM
                    frame_ssim.append(img_ssim(gt_frame.numpy(), gen_frame.numpy()))

                    # LPIPS (needs [-1, 1] range)
                    gt_lpips = (gt_frame * 2 - 1).unsqueeze(0).to(device)
                    gen_lpips = (gen_frame * 2 - 1).unsqueeze(0).to(device)
                    with torch.no_grad():
                        lp = lpips_fn(gt_lpips, gen_lpips).mean().item()
                    frame_lpips.append(lp)

                # Average across frames for this video
                psnr_scores.append(np.mean(frame_psnr))
                ssim_scores.append(np.mean(frame_ssim))
                lpips_scores.append(np.mean(frame_lpips))

            except Exception as e:
                print(f"\n  ERROR processing {video_name}: {e}")
                continue

        # Save results for this mode (partial or final)
        if psnr_scores:
            if args.partial_output:
                # Partial output: raw scores for later merge (only first mode when multiple)
                partial = {
                    "mode": mode,
                    "baseline": args.baseline,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "num_videos": len(psnr_scores),
                    "psnr_scores": psnr_scores,
                    "ssim_scores": ssim_scores,
                    "lpips_scores": lpips_scores,
                }
                os.makedirs(os.path.dirname(args.partial_output) or ".", exist_ok=True)
                with open(args.partial_output, "w") as f:
                    json.dump(partial, f, indent=2)
                print(f"\n  Partial results for {mode} (videos {start_idx}-{end_idx-1}):")
                print(f"    PSNR:  {np.mean(psnr_scores):.4f}  SSIM:  {np.mean(ssim_scores):.4f}  LPIPS: {np.mean(lpips_scores):.4f}")
                print(f"    Saved to: {args.partial_output}")
            else:
                result = {
                    "mode": mode,
                    "baseline": args.baseline,
                    "num_videos": len(psnr_scores),
                    "psnr": {"mean": float(np.mean(psnr_scores)), "std": float(np.std(psnr_scores))},
                    "ssim": {"mean": float(np.mean(ssim_scores)), "std": float(np.std(ssim_scores))},
                    "lpips": {"mean": float(np.mean(lpips_scores)), "std": float(np.std(lpips_scores))},
                }
                all_results[mode] = result

                print(f"\n  Results for {mode}:")
                print(f"    PSNR:  {result['psnr']['mean']:.4f} +/- {result['psnr']['std']:.4f}")
                print(f"    SSIM:  {result['ssim']['mean']:.4f} +/- {result['ssim']['std']:.4f}")
                print(f"    LPIPS: {result['lpips']['mean']:.4f} +/- {result['lpips']['std']:.4f}")

                mode_result_path = os.path.join(args.save_dir, f"{mode}_vs_{args.baseline}.json")
                with open(mode_result_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"    Saved to: {mode_result_path}")

    # Save combined results (skip when using partial output)
    if all_results and not args.partial_output:
        combined_path = os.path.join(args.save_dir, "all_fidelity_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to: {combined_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("FIDELITY METRICS SUMMARY (vs baseline)")
    print(f"{'='*70}")
    print(f"{'Mode':<25} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
    print("-" * 55)
    for mode, result in all_results.items():
        print(f"{mode:<25} {result['psnr']['mean']:>10.4f} {result['ssim']['mean']:>10.4f} {result['lpips']['mean']:>10.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
