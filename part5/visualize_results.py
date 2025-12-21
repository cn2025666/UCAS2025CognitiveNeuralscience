import matplotlib.pyplot as plt
import numpy as np
import json
import os

# --- 1. 数据准备 ---

# 任务名称简化映射
task_map = {
    'AntiReach-v0': 'AntiReach',
    'ContextDecisionMaking-v0': 'ContextDM',
    'DelayComparison-v0': 'DelayComp',
    'DelayMatchCategory-v0': 'MatchCat',
    'DelayMatchSample-v0': 'MatchSample',
    'DelayMatchSampleDistractor1D-v0': 'Distractor',
    'DelayPairedAssociation-v0': 'PairedAssoc',
    'DualDelayMatchSample-v0': 'DualMatch',
    'EconomicDecisionMaking-v0': 'EconomicDM',
    'GoNogo-v0': 'GoNogo'
}

tasks_short = list(task_map.values())

# Part 1 Baseline 数据 (来自 part1/visible.py)
baseline_data = {
    "LSTM": [1.0000, 0.3258, 0.9659, 0.5775, 1.0000, 1.0000, 1.0000, 0.6720, 1.0000, 1.0000],
    "GRU":  [1.0000, 0.3258, 1.0000, 0.5699, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    "CTRNN":[0.9247, 0.3257, 0.8997, 0.5642, 1.0000, 1.0000, 1.0000, 0.8361, 1.0000, 1.0000]
}

# Part 5 Realistic 数据
# 尝试从文件读取，如果失败则使用默认值 (基于之前的运行结果)
realistic_data = {}
json_path = os.path.join(os.path.dirname(__file__), 'results_realistic.json')

try:
    with open(json_path, 'r') as f:
        results = json.load(f)
        print(f"Loaded realistic results from {json_path}")
        for model in ["LSTM", "GRU", "CTRNN"]:
            realistic_data[model] = results[model]['f1']
except FileNotFoundError:
    print("Warning: results_realistic.json not found. Using default values from previous run.")
    realistic_data = {
        "LSTM": [0.7686, 0.4275, 0.8133, 0.6520, 0.6477, 1.0000, 0.7226, 0.5855, 1.0000, 0.9890],
        "GRU":  [0.6519, 0.3834, 0.7712, 0.5695, 0.6731, 1.0000, 1.0000, 0.6676, 1.0000, 0.9876],
        "CTRNN":[0.2546, 0.4071, 0.7379, 0.5775, 0.7896, 1.0000, 0.9879, 0.5894, 1.0000, 0.9572]
    }

# --- 绘图 1: Realistic 环境下的模型对比 (对应 part1/visible.py) ---
def plot_realistic_comparison():
    x = np.arange(len(tasks_short))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
    
    rects1 = ax.bar(x - width, realistic_data['LSTM'], width, label='LSTM', color='#3498db', alpha=0.9, edgecolor='white')
    rects2 = ax.bar(x, realistic_data['GRU'], width, label='GRU', color='#2ecc71', alpha=0.9, edgecolor='white')
    rects3 = ax.bar(x + width, realistic_data['CTRNN'], width, label='CTRNN', color='#e67e22', alpha=0.9, edgecolor='white')

    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Model Performance in Realistic Environment (Noise + Variable Timing)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks_short, rotation=30, ha='right', fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.legend(loc='upper right', fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # 数值标注
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)

    autolabel(rects1); autolabel(rects2); autolabel(rects3)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'realistic_comparison.png')
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    # plt.show()

# --- 绘图 2: 环境影响分析 (Baseline vs Realistic) ---
def plot_impact_analysis():
    models = ['LSTM', 'GRU', 'CTRNN']
    colors = ['#3498db', '#2ecc71', '#e67e22']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), dpi=100, sharex=True)
    
    x = np.arange(len(tasks_short))
    width = 0.35
    
    for i, model in enumerate(models):
        ax = axes[i]
        base_scores = baseline_data[model]
        real_scores = realistic_data[model]
        
        rects1 = ax.bar(x - width/2, base_scores, width, label='Baseline (Part 1)', color='lightgray', edgecolor='grey')
        rects2 = ax.bar(x + width/2, real_scores, width, label='Realistic (Part 5)', color=colors[i], edgecolor='white')
        
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title(f'{model} Performance: Baseline vs Realistic', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.3)
        ax.legend(loc='upper right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        
        # 标注差异
        for j, (b, r) in enumerate(zip(base_scores, real_scores)):
            diff = r - b
            if abs(diff) > 0.05: # 只标注显著差异
                color = 'red' if diff < 0 else 'green'
                ax.text(j, max(b, r) + 0.05, f'{diff:+.2f}', ha='center', color=color, fontsize=9, fontweight='bold')

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(tasks_short, rotation=30, ha='right', fontsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'impact_analysis.png')
    plt.savefig(save_path)
    print(f"Saved impact analysis plot to {save_path}")
    # plt.show()

if __name__ == "__main__":
    plot_realistic_comparison()
    plot_impact_analysis()
