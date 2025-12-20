import matplotlib.pyplot as plt

stages = ['Stage 1', 'Stage 2', 'Stage 3']
t1_f1 = [1.0000, 0.6970, 0.7741] # GoNogo
t2_f1 = [0.0, 0.9163, 0.3601]    # DelayComp
t3_f1 = [0.0, 0.0, 0.9971]       # DMS

plt.figure(figsize=(9, 5))
plt.plot(stages, t1_f1, 'o-', label='Task 1 (GoNogo) - Protected', color='#d62728', linewidth=3)
plt.plot(stages, t2_f1, 's-', label='Task 2 (DelayComp)', color='#1f77b4', linewidth=2)
plt.plot(stages, t3_f1, '^-', label='Task 3 (DMS)', color='#2ca02c', linewidth=2)

# 画一条基准虚线，展示不加算法时的遗忘水平
plt.axhline(y=0.31, color='gray', linestyle='--', label='Unprotected Baseline')

plt.title('Sequential Learning Performance with EWC (1000 Epochs)', fontsize=12)
plt.ylabel('F1 Score')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left')
plt.show()