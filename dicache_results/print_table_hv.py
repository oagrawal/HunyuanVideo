import os
import json

latencies = {
    "dicache_baseline": ("1265s", "1.00x"),
    "dicache_fixed_0.05": ("658s", "1.92x"),
    "dicache_fixed_0.10": ("419s", "3.02x"),
    "dicache_fixed_0.15": ("346s", "3.65x"),
    "dicache_fixed_0.20": ("286s", "4.42x"),
    "dicache_fixed_0.25": ("250s", "5.05x"),
    "dicache_fixed_0.30": ("226s", "5.59x"),
    "dicache_fixed_0.35": ("202s", "6.25x"),
    "dicache_fixed_0.40": ("186s", "6.79x"),
    "dicache_fixed_0.60": ("164s", "7.69x"),
    "dicache_adaptive_0.05_0.20": ("465s", "2.72x"),
    "dicache_adaptive_0.10_0.30": ("326s", "3.87x"),
    "dicache_adaptive_0.15_0.40": ("275s", "4.60x"),
}

fidelity_dir = "/nfs/oagrawal/HunyuanVideo/dicache_results/fidelity_metrics"
results = []

def sort_key(mode):
    if "baseline" in mode: return (0, 0)
    if "fixed" in mode: 
        try: return (1, float(mode.split("_")[-1]))
        except: return (1, 0)
    return (2, mode)

for mode in sorted(latencies.keys(), key=sort_key):
    lat, speedup = latencies[mode]
    if mode == "dicache_baseline":
        results.append((mode, lat, speedup, "Inf", "1.0000", "0.0000"))
        continue
    
    path = os.path.join(fidelity_dir, mode + "_vs_dicache_baseline.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            js = json.load(f)
            psnr = "%.4f" % js["psnr"]["mean"]
            ssim = "%.4f" % js["ssim"]["mean"]
            lpips = "%.4f" % js["lpips"]["mean"]
            results.append((mode, lat, speedup, psnr, ssim, lpips))
    else:
        results.append((mode, lat, speedup, "N/A", "N/A", "N/A"))

print("| Mode | Latency | Speedup | PSNR | SSIM | LPIPS |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")
for r in results:
    print("| " + " | ".join(r) + " |")
