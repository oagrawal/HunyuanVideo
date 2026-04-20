import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# User provided latencies
latencies = {
    "dicache_baseline": 1265,
    "dicache_fixed_0.05": 658,
    "dicache_fixed_0.10": 419,
    "dicache_fixed_0.15": 346,
    "dicache_fixed_0.20": 286,
    "dicache_fixed_0.25": 250,
    "dicache_fixed_0.30": 226,
    "dicache_fixed_0.35": 202,
    "dicache_fixed_0.40": 186,
    "dicache_fixed_0.60": 164,
    "dicache_adaptive_0.05_0.20": 465,
    "dicache_adaptive_0.10_0.30": 326,
    "dicache_adaptive_0.15_0.40": 275,
}

# Scan fidelity metrics
fidelity_dir = "/nfs/oagrawal/HunyuanVideo/dicache_results/fidelity_metrics"
data = []

# Add baseline (perfect reference)
data.append({
    "mode": "dicache_baseline",
    "latency": 1265,
    "psnr": 45.0, # Approximate reasonable value for "perfect" comparison or just skip from plot if needed
    "ssim": 1.0,
    "lpips": 0.0,
    "is_adaptive": False
})

for filename in os.listdir(fidelity_dir):
    if filename.endswith(".json"):
        with open(os.path.join(fidelity_dir, filename), "r") as f:
            js = json.load(f)
            mode = js["mode"]
            if mode in latencies:
                data.append({
                    "mode": mode,
                    "latency": latencies[mode],
                    "psnr": js["psnr"]["mean"],
                    "ssim": js["ssim"]["mean"],
                    "lpips": js["lpips"]["mean"],
                    "is_adaptive": "adaptive" in mode
                })

df = pd.DataFrame(data)

# Colors
colors = {True: "#FF6F61", False: "#6B5B95"} # Adaptive vs Fixed/Baseline
labels = {True: "Adaptive Modes", False: "Baseline & Fixed Modes"}

metrics = ["psnr", "ssim", "lpips"]
metric_titles = {"psnr": "PSNR (dB) - Higher is Better", 
                 "ssim": "SSIM - Higher is Better", 
                 "lpips": "LPIPS - Lower is Better"}

for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    # Sort for plotting line
    fixed_group = df[~df["is_adaptive"]].sort_values("latency")
    
    # Plot connecting line for Fixed Modes
    plt.plot(fixed_group["latency"], fixed_group[metric], linestyle="-", color="#6B5B95", alpha=0.5, label="Fixed Threshold Curve", zorder=1)
    
    # Scatter points
    for is_adapt, group in df.groupby("is_adaptive"):
        plt.scatter(group["latency"], group[metric], 
                    c=colors[is_adapt], label=labels[is_adapt], s=120, edgecolors="black", alpha=0.8, zorder=3)
    
    plt.xlabel("Latency (seconds)", fontsize=12)
    plt.ylabel(metric_titles[metric], fontsize=12)
    plt.title(f"HunyuanVideo DiCache Pareto Frontier: {metric.upper()} vs Latency", fontsize=14, pad=20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left" if metric == "lpips" else "lower right")
    
    output_path = f"/nfs/oagrawal/HunyuanVideo/dicache_results/pareto_frontier_hv_{metric}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
