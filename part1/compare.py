import matplotlib.pyplot as plt
import numpy as np

# 1. 准备最新数据 (Seed 42 实测值)
task_labels = [
    'AntiReach', 'ContextDM', 'DelayComp', 'MatchCat',
    'MatchSample', 'Distractor', 'PairedAssoc', 'DualDelay',
    'EconomicDM', 'GoNogo'
]

# 填入你最新的实测数据
lstm_f1 = [1.0000, 0.3258, 0.7333, 0.5733, 1.0000, 1.0000, 1.0000, 0.6625, 1.0000, 1.0000]
gru_f1  = [0.9531, 0.3255, 0.7074, 0.6330, 1.0000, 1.0000, 1.0000, 0.7069, 1.0000, 1.0000]
ctrnn_f1 = [0.9121, 0.3262, 0.5699, 0.5539, 0.7619, 1.0000, 1.0000, 0.7139, 1.0000, 1.0000]

# 2. 设置绘图参数
x = np.arange(len(task_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(15, 8), dpi=100)

# 3. 绘制三组柱状图
rects1 = ax.bar(x - width, lstm_f1, width, label='LSTM', color='#3498db', edgecolor='white')
rects2 = ax.bar(x, gru_f1, width, label='GRU', color='#2ecc71', edgecolor='white')
rects3 = ax.bar(x + width, ctrnn_f1, width, label='CTRNN', color='#e67e22', edgecolor='white')

# 4. 图表细节美化
ax.set_ylabel('F1 Score (Parallel Training)', fontsize=12)
ax.set_title('Cognitive Task Performance Comparison (Seed 42)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(task_labels, rotation=30, ha='right', fontsize=10)
ax.set_ylim(0, 1.25)
ax.legend(loc='upper right', fontsize=12)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)

# 5. 数值标注
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=45)

autolabel(rects1); autolabel(rects2); autolabel(rects3)

plt.tight_layout()
plt.show()