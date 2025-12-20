import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据录入 (1000 Epochs 实测值) ---
tasks = ['AntiReach', 'ContextDM', 'DelayComp', 'MatchCat', 'MatchSample',
         'Distractor', 'PairedAssoc', 'DualMatch', 'EconomicDM', 'GoNogo']

# 最新 F1 分数
f1_lstm = [1.0000, 0.3258, 0.9659, 0.5775, 1.0000, 1.0000, 1.0000, 0.6720, 1.0000, 1.0000]
f1_gru  = [1.0000, 0.3258, 1.0000, 0.5699, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
f1_ctrnn = [0.9247, 0.3257, 0.8997, 0.5642, 1.0000, 1.0000, 1.0000, 0.8361, 1.0000, 1.0000]

# 最新训练 Loss (每 100 Epoch 采样)
epochs = np.arange(0, 1100, 100)
loss_lstm = [2.9265, 0.0834, 0.0588, 0.0497, 0.0480, 0.0462, 0.0402, 0.0294, 0.0299, 0.0253, 0.0238]
loss_gru  = [3.1232, 0.0923, 0.0558, 0.0534, 0.0441, 0.0395, 0.0333, 0.0325, 0.0262, 0.0218, 0.0225]
loss_ctrnn = [3.0982, 0.1499, 0.1175, 0.0874, 0.0677, 0.0546, 0.0482, 0.0433, 0.0385, 0.0377, 0.0304]

# --- 绘图 1：F1 分数对比柱状图 ---
x = np.arange(len(tasks))
width = 0.25

fig, ax1 = plt.subplots(figsize=(14, 7), dpi=100)
rects1 = ax1.bar(x - width, f1_lstm, width, label='LSTM', color='#3498db', alpha=0.9)
rects2 = ax1.bar(x, f1_gru, width, label='GRU', color='#2ecc71', alpha=0.9)
rects3 = ax1.bar(x + width, f1_ctrnn, width, label='CTRNN', color='#e67e22', alpha=0.9)

ax1.set_ylabel('F1 Score', fontsize=12)
ax1.set_title('Performance Comparison Across 10 Tasks (1000 Epochs, Seed 42)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(tasks, rotation=30, ha='right')
ax1.set_ylim(0, 1.2)
ax1.legend(loc='upper right')

# 在柱状图上添加数值标注
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

autolabel(rects1); autolabel(rects2); autolabel(rects3)
plt.tight_layout()
plt.savefig('compare.png') # 自动保存为你的 LaTeX 引用名
plt.show()

# --- 绘图 2：训练 Loss 收敛曲线 ---
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(epochs, loss_lstm, 'o-', markersize=4, label='LSTM', color='#3498db', linewidth=2)
plt.plot(epochs, loss_gru, 's-', markersize=4, label='GRU', color='#2ecc71', linewidth=2)
plt.plot(epochs, loss_ctrnn, '^-', markersize=4, label='CTRNN', color='#e67e22', linewidth=2)

plt.yscale('log') # 对数坐标可以更清楚地看到后期微小的下降
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('CE Loss (Log Scale)', fontsize=12)
plt.title('Multi-Task Learning Convergence (0-1000 Epochs)', fontsize=14, fontweight='bold')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig('loss.png') # 自动保存为你的 LaTeX 引用名
plt.show()