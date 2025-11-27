import matplotlib.pyplot as plt
import os

# Read delta values
delta_values = []
with open('delta_values.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            delta_values.append(float(line))

# Read intermediate L1 distances
intermediate_l1_distances = []
with open('intermediate_l1_distances.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            intermediate_l1_distances.append(float(line))

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot delta values
if len(delta_values) > 0:
    ax1.plot(range(1, len(delta_values) + 1), 
            delta_values, 
            'g-', linewidth=2, marker='s', markersize=6, label='TEMNI of Inputs')
    
    ax1.set_xlabel('Step Number')
    ax1.set_ylabel('L1 Distance Between Consecutive Steps')
    ax1.set_title('TEMNI of Inputs')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax1.legend()

# Plot intermediate L1 distances
if len(intermediate_l1_distances) > 0:
    ax2.plot(range(1, len(intermediate_l1_distances) + 1), 
            intermediate_l1_distances, 
            'b-', linewidth=2, marker='o', markersize=6, label='Outputs')
    
    ax2.set_xlabel('Step Number')
    ax2.set_ylabel('L1 Distance Between Consecutive Steps')
    ax2.set_title('Outputs')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax2.legend()

plt.tight_layout()

# Save the plot
save_path = 'combined_metrics_plot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Combined metrics plot saved to: {save_path}')

plt.close()

# Also create an overlaid plot on the same axes
plt.figure(figsize=(12, 6))

# Plot both lines on the same y-axis
ax1 = plt.gca()
lines = []
if len(delta_values) > 0:
    line1 = ax1.plot(range(1, len(delta_values) + 1), 
                     delta_values, 
                     'g-', linewidth=2, marker='s', markersize=6, label='TEMNI of Inputs')
    lines.extend(line1)

if len(intermediate_l1_distances) > 0:
    line2 = ax1.plot(range(1, len(intermediate_l1_distances) + 1), 
                     intermediate_l1_distances, 
                     'b-', linewidth=2, marker='o', markersize=6, label='Outputs')
    lines.extend(line2)

ax1.set_xlabel('Step Number')
ax1.set_ylabel('L1 Distance Between Consecutive Steps')
ax1.xaxis.set_major_locator(plt.MultipleLocator(10))

# Add title and grid
plt.title('L1 Distance Between Consecutive Steps: TEMNI of Inputs vs Outputs')
ax1.grid(True, alpha=0.3)

# Add legend
ax1.legend(loc='upper right')

# Save the overlaid plot
save_path_overlay = 'combined_metrics_overlay_plot.png'
plt.savefig(save_path_overlay, dpi=300, bbox_inches='tight')
print(f'Overlaid metrics plot saved to: {save_path_overlay}')

plt.close()

print(f'\nTotal delta values: {len(delta_values)}')
print(f'Total intermediate L1 distances: {len(intermediate_l1_distances)}')

