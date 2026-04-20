#!/usr/bin/env python3
"""
Phase 1 calibration for HunyuanVideo MagCache.

The mag_ratios (gamma) values are pre-calibrated by the MagCache authors and
hardcoded in magcache_sample_video.py. This script:
  1. Saves them to gamma_curve_hunyuanvideo.json
  2. Plots gamma and delta_gamma
  3. Determines T_boundary and saves regions_hunyuanvideo.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Pre-calibrated gamma values from magcache_sample_video.py (544p resolution)
# Index 0 is the placeholder for step 0 (no previous step), steps 1-49 are real ratios
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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "hunyuanvideo_544p"
NUM_STEPS = len(MAG_RATIOS_544P)  # 50

# ── 1. Save gamma curve JSON ──────────────────────────────────────────────────
gamma_data = {
    "model": MODEL_NAME,
    "resolution": "544p",
    "source": "pre-calibrated by MagCache authors (magcache_sample_video.py line 317)",
    "num_steps": NUM_STEPS,
    "timestep": list(range(NUM_STEPS)),
    "gamma": [round(float(g), 5) for g in MAG_RATIOS_544P],
}
json_path = os.path.join(OUT_DIR, f"gamma_curve_{MODEL_NAME}.json")
with open(json_path, "w") as f:
    json.dump(gamma_data, f, indent=2)
print(f"Saved: {json_path}")

# ── 2. Compute delta_gamma ────────────────────────────────────────────────────
delta_gamma = np.abs(np.diff(MAG_RATIOS_544P))  # length 49, indices 1..49

# ── 3. Determine T_boundary ───────────────────────────────────────────────────
# The "unstable" region is where gamma drops sharply below ~0.97 and delta_gamma spikes.
# Retention ratio = 0.2 → first 10 steps always run (no skipping ever happens there).
# From the data: values drop significantly starting at step 43 (0.97039).
# Using the standard "last 20%" heuristic → T_boundary = 40.
# Cross-checking: gamma[40]=0.9904, gamma[43]=0.97039, gamma[47]=0.91154.
# The meaningful drop (below 0.97) starts at step 43, but we use 40 to be conservative.
RETENTION_RATIO = 0.2
T_BOUNDARY = int(0.8 * NUM_STEPS)  # 40: last 20% = steps 40-49

stable_steps   = list(range(0, T_BOUNDARY))     # 0-39
unstable_steps = list(range(T_BOUNDARY, NUM_STEPS))  # 40-49

print(f"\nT_boundary = {T_BOUNDARY}")
print(f"Stable steps:   {stable_steps[0]}..{stable_steps[-1]}  "
      f"(gamma range [{MAG_RATIOS_544P[stable_steps].min():.4f}, {MAG_RATIOS_544P[stable_steps].max():.4f}])")
print(f"Unstable steps: {unstable_steps[0]}..{unstable_steps[-1]}  "
      f"(gamma range [{MAG_RATIOS_544P[unstable_steps].min():.4f}, {MAG_RATIOS_544P[unstable_steps].max():.4f}])")

regions_data = {
    "model": MODEL_NAME,
    "T_boundary": T_BOUNDARY,
    "total_steps": NUM_STEPS,
    "retention_ratio": RETENTION_RATIO,
    "retention_steps": int(RETENTION_RATIO * NUM_STEPS),
    "stable_steps": stable_steps,
    "unstable_steps": unstable_steps,
    "notes": (
        "T_boundary=40 = last 20% of 50 steps. "
        "Gamma first drops below 0.97 at step 43 (0.97039), "
        "collapses to 0.766 by step 49. "
        "Stable region (0-39): gamma near 1.0. "
        "Unstable region (40-49): gamma falling, large delta_gamma."
    ),
}
regions_path = os.path.join(OUT_DIR, f"regions_{MODEL_NAME}.json")
with open(regions_path, "w") as f:
    json.dump(regions_data, f, indent=2)
print(f"Saved: {regions_path}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
timesteps = np.arange(NUM_STEPS)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle(
    f"MagCache Magnitude Ratio Curve — HunyuanVideo 544p\n"
    f"(pre-calibrated, 50 steps, retention_ratio=0.2)",
    fontsize=13, fontweight="bold"
)

# Top: gamma
ax = axes[0]
ax.axvspan(0, T_BOUNDARY - 0.5, alpha=0.08, color="green", label="stable region (0–39)")
ax.axvspan(T_BOUNDARY - 0.5, NUM_STEPS - 1, alpha=0.12, color="red", label="unstable region (40–49)")
ax.axvline(T_BOUNDARY, color="red", linestyle="--", linewidth=1.2, label=f"T_boundary={T_BOUNDARY}")
ax.axvline(int(RETENTION_RATIO * NUM_STEPS), color="gray", linestyle=":", linewidth=1.2,
           label=f"retention cutoff (step {int(RETENTION_RATIO * NUM_STEPS)})")
ax.plot(timesteps, MAG_RATIOS_544P, "b-o", markersize=3.5, linewidth=1.5, label="γ (mag ratio)")
ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_ylabel("γ (magnitude ratio)", fontsize=11)
ax.set_ylim(0.7, 1.35)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# Bottom: |Δgamma|
ax2 = axes[1]
ax2.axvspan(0, T_BOUNDARY - 0.5, alpha=0.08, color="green")
ax2.axvspan(T_BOUNDARY - 0.5, NUM_STEPS - 1, alpha=0.12, color="red")
ax2.axvline(T_BOUNDARY, color="red", linestyle="--", linewidth=1.2)
ax2.axvline(int(RETENTION_RATIO * NUM_STEPS), color="gray", linestyle=":", linewidth=1.2)
ax2.bar(timesteps[1:], delta_gamma, color="steelblue", alpha=0.7, width=0.7, label="|Δγ| = |γ[t] − γ[t−1]|")
ax2.set_xlabel("Diffusion step index (t)", fontsize=11)
ax2.set_ylabel("|Δγ|", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, f"gamma_curve_{MODEL_NAME}.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved: {plot_path}")
plt.close()

print("\n✓ Phase 1 calibration complete.")
print(f"  JSON:    {json_path}")
print(f"  Regions: {regions_path}")
print(f"  Plot:    {plot_path}")
