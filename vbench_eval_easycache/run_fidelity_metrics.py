#!/usr/bin/env python3
"""Compute PSNR/SSIM/LPIPS for EasyCache modes vs baseline."""

import argparse
import sys
import json
import math
import os
import cv2
import imageio
import lpips
import numpy as np
import torch
import tqdm

def img_psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100
    return 20 * math.log10(1 / math.sqrt(mse))

def ssim_single(img1, img2):
    C1, C2 = 0.01**2, 0.03**2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.T)
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1**2
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2**2
    sigma12 = cv2.filter2D(img1*img2, -1, window)[5:-5, 5:-5] - mu1*mu2
    ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def img_ssim(img1, img2):
    if img1.ndim == 2:
        return ssim_single(img1, img2)
    return np.mean([ssim_single(img1[i], img2[i]) for i in range(3)])

def load_video(path):
    r = imageio.get_reader(path, "ffmpeg")
    frames = [torch.tensor(f).permute(2, 0, 1) for f in r]
    r.close()
    return torch.stack(frames)

CACHED_MODES = ["easycache_fixed_0.025", "easycache_fixed_0.050", "easycache_adaptive"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", default="vbench_eval_easycache/videos")
    p.add_argument("--baseline", default="easycache_baseline")
    p.add_argument("--modes", default="all", help="Comma-separated modes, or 'all'. When subset, skips all_fidelity_results.json (for 2-GPU parallel runs).")
    p.add_argument("--save-dir", default="vbench_eval_easycache/fidelity_metrics")
    args = p.parse_args()

    modes = CACHED_MODES if args.modes == "all" else [m.strip() for m in args.modes.split(",")]
    baseline_dir = os.path.join(args.video_dir, args.baseline)
    if not os.path.exists(baseline_dir):
        print(f"ERROR: {baseline_dir} not found")
        sys.exit(1)
    os.makedirs(args.save_dir, exist_ok=True)

    baseline_videos = sorted([f for f in os.listdir(baseline_dir) if f.endswith(".mp4")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(device)
    all_results = {}

    for mode in modes:
        mode_dir = os.path.join(args.video_dir, mode)
        if not os.path.exists(mode_dir):
            continue
        common = sorted(set(baseline_videos) & set(os.listdir(mode_dir)))
        if not common:
            continue

        psnr_scores, ssim_scores, lpips_scores = [], [], []
        for name in tqdm.tqdm(common, desc=mode):
            gt = load_video(os.path.join(baseline_dir, name))
            gen = load_video(os.path.join(mode_dir, name))
            T = min(gt.shape[0], gen.shape[0])
            gt = (gt[:T].float() / 255.0)
            gen = (gen[:T].float() / 255.0)

            fpsnr, fssim, flpips = [], [], []
            for t in range(T):
                fpsnr.append(img_psnr(gt[t].numpy(), gen[t].numpy()))
                fssim.append(img_ssim(gt[t].numpy(), gen[t].numpy()))
                with torch.no_grad():
                    flpips.append(lpips_fn((gt[t]*2-1).unsqueeze(0).to(device), (gen[t]*2-1).unsqueeze(0).to(device)).mean().item())
            psnr_scores.append(np.mean(fpsnr))
            ssim_scores.append(np.mean(fssim))
            lpips_scores.append(np.mean(flpips))

        r = {"mode": mode, "baseline": args.baseline, "num_videos": len(psnr_scores),
             "psnr": {"mean": float(np.mean(psnr_scores)), "std": float(np.std(psnr_scores))},
             "ssim": {"mean": float(np.mean(ssim_scores)), "std": float(np.std(ssim_scores))},
             "lpips": {"mean": float(np.mean(lpips_scores)), "std": float(np.std(lpips_scores))}}
        all_results[mode] = r
        with open(os.path.join(args.save_dir, f"{mode}_vs_{args.baseline}.json"), "w") as f:
            json.dump(r, f, indent=2)
        print(f"  {mode}: PSNR {r['psnr']['mean']:.4f}  SSIM {r['ssim']['mean']:.4f}  LPIPS {r['lpips']['mean']:.4f}")

    # Only write aggregate file when running all modes (avoids race when splitting across GPUs)
    if args.modes == "all":
        with open(os.path.join(args.save_dir, "all_fidelity_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
