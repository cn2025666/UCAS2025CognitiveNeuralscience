import matplotlib.pyplot as plt

# 任务阶段
stages = ['Stage 1', 'Stage 2', 'Stage 3']

# 修正后的确定性数据
t1_f1 = [1.0000, 0.3209, 0.3087] # GoNogo
t2_f1 = [0.0, 1.0000, 0.5208]    # DelayComp
t3_f1 = [0.0, 0.0, 1.0000]       # DMS

plt.figure(figsize=(9, 5))
plt.plot(stages, t1_f1, 'o-', label='Task 1 (GoNogo)', color='red', linewidth=2)
plt.plot(stages, t2_f1, 's-', label='Task 2 (DelayComp)', color='blue', linewidth=2)
plt.plot(stages, t3_f1, '^-', label='Task 3 (DMS)', color='green', linewidth=2)

plt.title('True Catastrophic Forgetting (Seed 42)', fontsize=14)
plt.ylabel('F1 Score')
plt.ylim(0, 1.1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()