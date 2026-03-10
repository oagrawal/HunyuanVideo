"""
Plot pred_change for steps 5-48 only (excludes 0-4 and 49).
Mirrors the style of pred_change_plot.png from easycache_results.

Usage:
    python plot_pred_change_5_48.py

Data is read from the baseline_profile results directory.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path to pred_change.txt (baseline run)
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "easycache_results",
    "baseline_profile_2026-03-07-06:12:49_seed12345_Two_cats_boxing_in_bright_gloves_on_a_spotlighted_",
    "pred_change.txt",
)

# Load data
with open(DATA_PATH) as f:
    pred_change_history = [float(line.strip()) for line in f if line.strip()]

# Original: steps 1..49 (0-indexed: indices 0..48)
# User wants: steps 5-48 (inclusive) → 0-indexed indices 4..47
START_STEP = 5
END_STEP = 48
data_slice = pred_change_history[START_STEP - 1 : END_STEP]
x_pc = list(range(START_STEP, END_STEP + 1))

plt.figure(figsize=(10, 5))
plt.plot(x_pc, data_slice, "g-", linewidth=2, marker="s", markersize=5)
plt.xlabel("Diffusion Step")
plt.ylabel("pred_change")
plt.title(
    "EasyCache Per-Step Relative Change  "
    "pred_change_t = k_{t-1} · (||x_t − x_{t-1}|| / ||v_{t-1}||)\n"
    "(steps 5–48 only)"
)
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(SCRIPT_DIR, "pred_change_plot_5_48.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
