import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(task2losses, title):
    """绘制不同任务的loss曲线"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, (task, losses) in enumerate(task2losses.items()):
        epochs = list(range(0, 2000, 100))
        ax = axes[i]
        ax.plot(epochs, losses, 'b-', linewidth=2)
        ax.set_title(f'{task}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2000)
        ax.set_ylim(min(losses) - 1, max(losses) + 1)
    plt.suptitle(f'Loss Curves for 10 Tasks ({title})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_f1_matrix(task2f1, title):
    """绘制F1分数矩阵热图"""
    tasks = list(task2f1.keys())
    f1_matrix = np.array([task2f1[task] for task in tasks])
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(f1_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('F1 Score', rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels([f'Task {i+1}' for i in range(len(tasks))], rotation=45, ha='right')
    ax.set_yticklabels(tasks)
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            text = ax.text(j, i, f'{f1_matrix[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if f1_matrix[i, j] < 0.5 else "black",
                          fontsize=8)
    ax.set_title(f'F1 Scores After Each Task ({title})\nRows: Test Tasks, Columns: Training Stage', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

def visualize_f1_line(task2f1, title):
    """绘制F1分数折线图"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    tasks = list(task2f1.keys())
    for idx, task_name in enumerate(tasks):
        ax = axes[idx]
        stages = list(range(1, 11))
        f1_scores = task2f1[task_name]
        ax.plot(stages, f1_scores, 'o-', linewidth=2, markersize=6, color='tab:blue')
        task_index = idx + 1
        if task_index <= len(f1_scores):
            ax.plot(task_index, f1_scores[task_index-1], 'o', markersize=10, color='red', label='Learning Point')
        ax.set_title(f'{task_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Stage')
        ax.set_ylabel('F1 Score')
        ax.set_xticks(stages)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.suptitle(f'F1 Score Evolution During Training ({title})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    simple_task2losses = {
        'AntiReach':                    [3.4895, 0.7363, 0.5760, 0.3384, 0.2203, 0.1612, 0.1237, 0.0970, 0.0764, 0.0597, 0.0474, 0.0373, 0.0297, 0.0234, 0.0184, 0.0143, 0.0113, 0.0091, 0.0073, 0.0059],
        'ContextDecisionMaking':        [1.0973, 0.1726, 0.1690, 0.1822, 0.1699, 0.1706, 0.1703, 0.1655, 0.1698, 0.1713, 0.1690, 0.1673, 0.1651, 0.1687, 0.1679, 0.1652, 0.1665, 0.1607, 0.1606, 0.1516],
        'DelayComparison':              [0.4515, 0.0260, 0.0216, 0.0149, 0.0088, 0.0053, 0.0031, 0.0026, 0.0031, 0.0024, 0.0014, 0.0014, 0.0011, 0.0019, 0.0006, 0.0012, 0.0012, 0.0007, 0.0008, 0.0008],
        'DelayMatchCategory':           [1.2265, 0.0462, 0.0435, 0.0426, 0.0423, 0.0419, 0.0421, 0.0412, 0.0417, 0.0419, 0.0424, 0.0417, 0.0463, 0.0429, 0.0425, 0.0418, 0.0419, 0.0421, 0.0419, 0.0427],
        'DelayMatchSample':             [0.6148, 0.0510, 0.0237, 0.0031, 0.0014, 0.0008, 0.0251, 0.0021, 0.0011, 0.0008, 0.0005, 0.0004, 0.0004, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002],
        'DelayMatchSampleDistractor1D': [0.0200, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        'DelayPairedAssociation':       [0.8634, 0.0450, 0.0420, 0.0435, 0.0422, 0.0420, 0.0418, 0.0437, 0.0446, 0.0424, 0.0430, 0.0417, 0.0436, 0.0353, 0.0418, 0.0414, 0.0417, 0.0413, 0.0418, 0.0428],
        'DualDelayMatchSample':         [2.7142, 0.1341, 0.1364, 0.1294, 0.1250, 0.1244, 0.1254, 0.1299, 0.1252, 0.1261, 0.1247, 0.1144, 0.1123, 0.0871, 0.0681, 0.0670, 0.0432, 0.0271, 0.0239, 0.0185],
        'EconomicDecisionMaking':       [0.4728, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        'GoNogo':                       [0.6843, 0.0007, 0.0003, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
    }
    simple_task2f1 = {
        'AntiReach':                    [1.0000, 0.0371, 0.0289, 0.0361, 0.0289, 0.0289, 0.0321, 0.0278, 0.0289, 0.0376],
        'ContextDecisionMaking':        [0.0000, 0.3251, 0.3248, 0.3252, 0.3123, 0.3249, 0.3246, 0.3143, 0.3253, 0.3220],
        'DelayComparison':              [0.0000, 0.0000, 0.9856, 0.3283, 0.5013, 0.3283, 0.3283, 0.3040, 0.3283, 0.3283],
        'DelayMatchCategory':           [0.0000, 0.0000, 0.0000, 0.5571, 0.3728, 0.3230, 0.3371, 0.5065, 0.3230, 0.4452],
        'DelayMatchSample':             [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.3212, 0.3212, 0.2888, 0.3212, 0.3504],
        'DelayMatchSampleDistractor1D': [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.4613, 0.2947, 0.4987, 0.4859],
        'DelayPairedAssociation':       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8326, 0.3389, 0.4928, 0.6628],
        'DualDelayMatchSample':         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.3004, 0.3004],
        'EconomicDecisionMaking':       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.3429],
        'GoNogo':                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
    }
    HME_task2losses = {
        'AntiReach':                    [3.5038, 0.4130, 0.0963, 0.0289, 0.0103, 0.0042, 0.0022, 0.0014, 0.0009, 0.0007, 0.0005, 0.0004, 0.0004, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001],
        'ContextDecisionMaking':        [3.4710, 0.2561, 0.1895, 0.1776, 0.1688, 0.1666, 0.1649, 0.1645, 0.1649, 0.1631, 0.1700, 0.1607, 0.1604, 0.1647, 0.1510, 0.1446, 0.1417, 0.1355, 0.1289, 0.1209],
        'DelayComparison':              [3.4673, 0.1984, 0.1546, 0.1085, 0.0468, 0.0542, 0.0325, 0.0237, 0.0122, 0.0073, 0.0048, 0.0033, 0.0023, 0.0018, 0.0281, 0.0243, 0.0225, 0.0207, 0.0184, 0.0164],
        'DelayMatchCategory':           [3.4947, 0.3032, 0.2404, 0.1379, 0.0664, 0.0558, 0.0503, 0.0470, 0.0450, 0.0447, 0.0434, 0.0428, 0.0425, 0.0422, 0.0420, 0.0416, 0.0411, 0.0414, 0.0403, 0.0406],
        'DelayMatchSample':             [3.5406, 0.3433, 0.2533, 0.0907, 0.2452, 0.0772, 0.0642, 0.0606, 0.0557, 0.0537, 0.0517, 0.2979, 0.1845, 0.0680, 0.0624, 0.0562, 0.0529, 0.0508, 0.0500, 0.0517],
        'DelayMatchSampleDistractor1D': [3.5056, 0.0368, 0.0183, 0.0068, 0.0017, 0.0007, 0.0004, 0.0003, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        'DelayPairedAssociation':       [0.0664, 0.0415, 0.0414, 0.0404, 0.0397, 0.0229, 0.0409, 0.0376, 0.0102, 0.0137, 0.0184, 0.0003, 0.0001, 0.0001, 0.0339, 0.0299, 0.0180, 0.0086, 0.0082, 0.0027],
        'DualDelayMatchSample':         [3.4886, 0.5966, 0.3031, 0.1999, 0.1469, 0.1336, 0.5083, 0.2252, 0.1417, 0.1321, 0.1290, 0.1275, 0.1266, 0.2117, 0.1376, 0.1297, 0.1275, 0.1263, 0.1254, 0.1345],
        'EconomicDecisionMaking':       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        'GoNogo':                       [0.1549, 0.0010, 0.0002, 0.0001, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0260]
    }
    HME_task2f1 = {
        'AntiReach':                    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.1961, 0.2134],
        'ContextDecisionMaking':        [0.0000, 0.3514, 0.3578, 0.3921, 0.3664, 0.3457, 0.3702, 0.3593, 0.3519, 0.3444],
        'DelayComparison':              [0.0000, 0.0000, 0.9507, 0.9861, 0.9929, 0.9721, 0.4621, 0.4787, 0.4729, 0.5086],
        'DelayMatchCategory':           [0.0000, 0.0000, 0.0000, 0.6291, 0.6381, 0.6611, 0.6653, 0.6627, 0.6487, 0.6340],
        'DelayMatchSample':             [0.0000, 0.0000, 0.0000, 0.0000, 0.5721, 0.5502, 0.5556, 0.5709, 0.5646, 0.4303],
        'DelayMatchSampleDistractor1D': [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        'DelayPairedAssociation':       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.9713, 0.9900, 0.9898, 0.4918],
        'DualDelayMatchSample':         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6540, 0.6447, 0.6480],
        'EconomicDecisionMaking':       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000],
        'GoNogo':                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
    }
    # 绘制loss曲线
    visualize_loss(simple_task2losses, 'Without Continual Learning')
    visualize_loss(HME_task2losses, 'With Continual Learning (HME)')
    
    # 绘制F1矩阵
    visualize_f1_matrix(simple_task2f1, 'Without Continual Learning')
    visualize_f1_matrix(HME_task2f1, 'With Continual Learning (HME)')
    
    # 绘制F1折线图
    visualize_f1_line(simple_task2f1, 'Without Continual Learning')
    visualize_f1_line(HME_task2f1, 'With Continual Learning (HME)')
