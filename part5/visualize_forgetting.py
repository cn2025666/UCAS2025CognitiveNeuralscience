import matplotlib.pyplot as plt
import numpy as np
import json
import os

# --- 1. 数据准备 ---
stages = ['Stage 1\n(Train T1)', 'Stage 2\n(Train T2)', 'Stage 3\n(Train T3)']
tasks = ['Task 1 (GoNogo)', 'Task 2 (DelayComp)', 'Task 3 (DMS)']

# Part 1 Data (Baseline Environment) - From part1/forget_visible.py
# Note: Part 1 data uses 0.0 for untrained tasks, which is a simplification.
p1_data = {
    'T1': [1.0000, 0.3209, 0.3087],
    'T2': [0.0, 1.0000, 0.5208],
    'T3': [0.0, 0.0, 1.0000]
}

# Part 5 Data (Realistic Environment)
json_path = os.path.join(os.path.dirname(__file__), 'sequential_results.json')
try:
    with open(json_path, 'r') as f:
        results = json.load(f)
        history = results['history']
        # history is [[s1_t1, s1_t2, s1_t3], [s2_t1, ...], [s3_t1, ...]]
        p5_data = {
            'T1': [history[0][0], history[1][0], history[2][0]],
            'T2': [history[0][1], history[1][1], history[2][1]],
            'T3': [history[0][2], history[1][2], history[2][2]]
        }
except FileNotFoundError:
    print("Warning: sequential_results.json not found. Using default values.")
    p5_data = {
        'T1': [0.9746, 0.5415, 0.5300],
        'T2': [0.3375, 0.9119, 0.5686],
        'T3': [0.4775, 0.5930, 1.0000]
    }

# --- 2. 绘图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=100, sharey=True)

# Plot Part 1 (Baseline)
ax1.plot(stages, p1_data['T1'], 'o-', label=tasks[0], color='#e74c3c', linewidth=2, markersize=8)
ax1.plot(stages, p1_data['T2'], 's-', label=tasks[1], color='#3498db', linewidth=2, markersize=8)
ax1.plot(stages, p1_data['T3'], '^-', label=tasks[2], color='#2ecc71', linewidth=2, markersize=8)

ax1.set_title('Baseline Environment (Part 1)\nCatastrophic Forgetting', fontsize=14, fontweight='bold')
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.set_ylim(-0.05, 1.1)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower left')

# Annotate drops for T1
drop_p1 = p1_data['T1'][0] - p1_data['T1'][-1]
ax1.annotate(f'Drop: -{drop_p1:.2f}', 
             xy=(2, p1_data['T1'][-1]), 
             xytext=(2, p1_data['T1'][-1] + 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             ha='center', color='#e74c3c', fontweight='bold')

# Plot Part 5 (Realistic)
ax2.plot(stages, p5_data['T1'], 'o-', label=tasks[0], color='#e74c3c', linewidth=2, markersize=8)
ax2.plot(stages, p5_data['T2'], 's-', label=tasks[1], color='#3498db', linewidth=2, markersize=8)
ax2.plot(stages, p5_data['T3'], '^-', label=tasks[2], color='#2ecc71', linewidth=2, markersize=8)

ax2.set_title('Realistic Environment (Part 5)\nCatastrophic Forgetting', fontsize=14, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower left')

# Annotate drops for T1
drop_p5 = p5_data['T1'][0] - p5_data['T1'][-1]
ax2.annotate(f'Drop: -{drop_p5:.2f}', 
             xy=(2, p5_data['T1'][-1]), 
             xytext=(2, p5_data['T1'][-1] + 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             ha='center', color='#e74c3c', fontweight='bold')

# Add comparison text
plt.figtext(0.5, 0.02, 
            f"Comparison: Forgetting is slightly mitigated in Realistic Env (Drop {drop_p1:.2f} vs {drop_p5:.2f}).\n"
            "Noise and variable timing might induce more robust feature learning.", 
            ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.15) # Make room for text
save_path = os.path.join(os.path.dirname(__file__), 'forgetting_comparison.png')
plt.savefig(save_path)
print(f"Saved forgetting comparison plot to {save_path}")
# plt.show()
