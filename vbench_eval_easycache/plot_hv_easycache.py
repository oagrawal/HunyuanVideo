import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths
base_dir = "/nfs/oagrawal/HunyuanVideo/vbench_eval_easycache"
fidelity_metrics_dir = os.path.join(base_dir, "fidelity_metrics")
generation_logs_dir = os.path.join(base_dir, "videos")

# 1. Load generation logs to calculate average latency per mode
mode_latencies = {} # mode -> [latencies]

log_files = glob.glob(os.path.join(generation_logs_dir, "generation_log_*.json"))
for log_file in log_files:
    with open(log_file, 'r') as f:
        data = json.load(f)
        for run in data.get('runs', []):
            mode = run['mode']
            latency = run['time_seconds']
            if mode not in mode_latencies:
                mode_latencies[mode] = []
            mode_latencies[mode].append(latency)

# Calculate mean latency
avg_latencies = {mode: np.mean(lats) for mode, lats in mode_latencies.items()}

# Baseline latency
baseline_mode = "easycache_baseline"
if baseline_mode not in avg_latencies:
    print(f"Error: {baseline_mode} not found in logs.")
    exit(1)

baseline_latency = avg_latencies[baseline_mode]
print(f"Baseline ({baseline_mode}) latency: {baseline_latency:.2f}s")

# 2. Load fidelity metrics and calculate speedups
data = []

metric_files = glob.glob(os.path.join(fidelity_metrics_dir, "*.json"))
for metric_file in metric_files:
    with open(metric_file, 'r') as f:
        js = json.load(f)
        mode = js['mode']
        
        if "baseline" in mode:
            continue
            
        if mode not in avg_latencies:
            print(f"Warning: Mode {mode} found in metrics but not in logs. Skipping.")
            continue
            
        speedup = baseline_latency / avg_latencies[mode]
        is_adaptive = "adaptive" in mode
        
        data.append({
            "mode": mode,
            "speedup": speedup,
            "psnr": js["psnr"]["mean"],
            "ssim": js["ssim"]["mean"],
            "lpips": js["lpips"]["mean"],
            "is_adaptive": is_adaptive
        })

if not data:
    print("Error: No data points found to plot.")
    exit(1)

df = pd.DataFrame(data)
print(f"Plotting {len(df)} points.")

# Set plot style
colors = {True: "#FF6F61", False: "#6B5B95"} # Adaptive vs Fixed
labels = {True: "Adaptive Modes", False: "Fixed Modes"}

metrics = ["psnr", "ssim", "lpips"]
metric_titles = {
    "psnr": "PSNR (dB) - Higher is Better", 
    "ssim": "SSIM - Higher is Better", 
    "lpips": "LPIPS - Lower is Better (Inverted Axis)"
}

for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    # Sort for plotting line
    fixed_group = df[~df["is_adaptive"]].sort_values("speedup")
    
    # Plot connecting line for Fixed Modes
    plt.plot(fixed_group["speedup"], fixed_group[metric], linestyle="-", color="#6B5B95", alpha=0.5, label="Fixed Threshold Curve", zorder=1)
    
    # Scatter points
    for is_adapt, group in df.groupby("is_adaptive"):
        plt.scatter(group["speedup"], group[metric], 
                    c=colors[is_adapt], label=labels[is_adapt], s=120, edgecolors="black", alpha=0.8, zorder=3)
    
    plt.xlabel("Speedup (x)", fontsize=12)
    plt.ylabel(metric_titles[metric], fontsize=12)
    plt.title(f"HunyuanVideo EasyCache Pareto: {metric.upper()} vs Speedup", fontsize=14, pad=20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    
    if metric == "lpips":
        plt.gca().invert_yaxis() # Lower is better
    
    output_path = os.path.join(base_dir, f"pareto_frontier_hv_ec_{metric}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
