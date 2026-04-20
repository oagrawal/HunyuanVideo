import pandas as pd
import matplotlib.pyplot as plt
import os

# Data provided by the user for HunyuanVideo + DiCache
data = [
    {"mode": "dicache_baseline", "latency": 1265, "vbench": 0.823473},
    {"mode": "dicache_fixed_0.05", "latency": 658, "vbench": 0.823744},
    {"mode": "dicache_fixed_0.10", "latency": 419, "vbench": 0.816793},
    {"mode": "dicache_fixed_0.15", "latency": 346, "vbench": 0.824883},
    {"mode": "dicache_fixed_0.20", "latency": 286, "vbench": 0.814471},
    {"mode": "dicache_fixed_0.25", "latency": 250, "vbench": 0.805960},
    {"mode": "dicache_fixed_0.30", "latency": 226, "vbench": 0.807700},
    {"mode": "dicache_fixed_0.35", "latency": 202, "vbench": 0.763163},
    {"mode": "dicache_fixed_0.40", "latency": 186, "vbench": 0.749989},
    {"mode": "dicache_fixed_0.60", "latency": 164, "vbench": 0.635090},
    {"mode": "dicache_adaptive_0.05_0.20", "latency": 465, "vbench": 0.822214},
    {"mode": "dicache_adaptive_0.10_0.30", "latency": 326, "vbench": 0.802284},
    {"mode": "dicache_adaptive_0.15_0.40", "latency": 275, "vbench": 0.785787},
]

df = pd.DataFrame(data)

# Separate baseline/fixed from adaptive
df['is_adaptive'] = df['mode'].str.contains('adaptive')

# Define colors
colors = {True: '#FF6F61', False: '#6B5B95'} # Adaptive: Coral/Red-ish, Fixed/Baseline: Purple/Blue-ish
labels = {True: 'Adaptive Modes', False: 'Baseline & Fixed Modes'}

plt.figure(figsize=(10, 6))

# Plot the connecting line for Baseline & Fixed Modes
fixed_group = df[~df['is_adaptive']].sort_values('latency')
plt.plot(fixed_group['latency'], fixed_group['vbench'], 
         linestyle='-', color='#6B5B95', alpha=0.5, label='Fixed Threshold Curve', zorder=1)

# Scatter points
for is_adapt, group in df.groupby('is_adaptive'):
    plt.scatter(group['latency'], group['vbench'], 
                c=colors[is_adapt], label=labels[is_adapt], s=120, edgecolors='black', alpha=0.8, zorder=3)

plt.xlabel('Latency (seconds)', fontsize=12)
plt.ylabel('VBench Score (Aggregated)', fontsize=12)
plt.title('HunyuanVideo DiCache Pareto Frontier: Quality vs. Latency', fontsize=14, pad=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left')

# Save the plot
output_path = "/nfs/oagrawal/HunyuanVideo/dicache_results/pareto_frontier_hv.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
