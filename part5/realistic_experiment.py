import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import neurogym as ngym
import numpy as np
import random
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

# --- 1. 彻底杀死随机性 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> 确定性模式已开启，随机种子: {seed}")

set_seed(42)

# --- 2. 任务准备 (Realistic Version) ---
tasks = ['AntiReach-v0', 'ContextDecisionMaking-v0', 'DelayComparison-v0',
         'DelayMatchCategory-v0', 'DelayMatchSample-v0', 'DelayMatchSampleDistractor1D-v0',
         'DelayPairedAssociation-v0', 'DualDelayMatchSample-v0', 'EconomicDecisionMaking-v0', 'GoNogo-v0']

# 改进：创建具有噪声和可变时间的“真实”环境
# sigma: 观测噪声标准差 (通过 Wrapper 添加)
# timing: 任务阶段的时间参数，使用 uniform 分布模拟时间不确定性
timing_kwargs = {
    'timing': {
        'fixation': ('uniform', [100, 300]),
        'delay': ('uniform', [200, 1000]),
        'decision': ('uniform', [200, 600])
    }
}
sigma_noise = 0.2

print(f">>> 初始化 Realistic 环境 (Sigma={sigma_noise}, Variable Timing)...")
envs = []
for t in tasks:
    # 1. 尝试应用可变时间参数
    try:
        env = gym.make(t, **timing_kwargs)
    except Exception as e:
        print(f"Warning: Could not apply timing to {t}, using default. Error: {e}")
        env = gym.make(t)
    
    # 2. 不使用 Wrapper，手动添加噪声以避免 seed 问题
    # env = ngym.wrappers.Noise(env, std_noise=sigma_noise)
    
    env.action_space.seed(42)
    envs.append(env)

MAX_INPUT, MAX_OUTPUT = 33, 33
HIDDEN_SIZE = 128
NUM_TASKS = len(tasks)
MAX_SEQ_LEN = 50 # 保持与 Part1 一致，但需注意时间变长可能导致截断

# --- 3. 数据采集 ---
def get_batch(task_idx, batch_size=16, seed_offset=0):
    env = envs[task_idx]
    batch_obs = np.zeros((batch_size, MAX_SEQ_LEN, MAX_INPUT))
    batch_targets = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)
    
    # 使用 numpy 的随机状态来生成噪声，确保可复现
    rng = np.random.RandomState(task_idx * 1000 + seed_offset)

    for b in range(batch_size):
        obs, _ = env.reset(seed=task_idx * 1000 + seed_offset + b)
        for t in range(MAX_SEQ_LEN):
            # 手动添加噪声
            noise = rng.normal(0, sigma_noise, size=obs.shape)
            batch_obs[b, t, :obs.shape[0]] = obs + noise
            
            # 使用随机动作推进环境
            obs, _, _, _, info = env.step(env.action_space.sample())
            
            # 获取 Ground Truth
            # 注意：在变长任务中，gt 的时序也会变化，info['gt'] 是当前步的真实标签
            # 如果 info 中没有 gt，尝试从 unwrapped 获取 (兼容旧代码逻辑，但在变长任务中可能不准确，优先信赖 info)
            gt = info.get('gt', 0) 
            # Part1 代码使用了 getattr(env.unwrapped, 'gt', ...)[t % 5]，这在变长任务中可能失效
            # NeuroGym 的 info['gt'] 通常是可靠的
            
            batch_targets[b, t] = gt
            
    task_onehot = np.zeros((batch_size, NUM_TASKS))
    task_onehot[:, task_idx] = 1
    return torch.FloatTensor(batch_obs), torch.LongTensor(batch_targets), torch.FloatTensor(task_onehot)

# --- 4. 模型定义 ---

class CognitiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks):
        super().__init__()
        self.lstm = nn.LSTM(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, task_id):
        task_info = task_id.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.lstm(torch.cat((x, task_info), dim=2))
        return self.fc(out)

class CognitiveGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks):
        super().__init__()
        self.gru = nn.GRU(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, task_id):
        task_info = task_id.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.gru(torch.cat((x, task_info), dim=2))
        return self.fc(out)

class CTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks, dt=1.0, tau=10.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.alpha = dt / tau
        self.input2h = nn.Linear(input_size + num_tasks, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2out = nn.Linear(hidden_size, output_size)

    def forward(self, x, task_id):
        batch_size, seq_len, _ = x.size()
        task_info = task_id.unsqueeze(1).repeat(1, seq_len, 1)
        combined_input = torch.cat((x, task_info), dim=2)
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        outputs = []
        for t in range(seq_len):
            curr_input = combined_input[:, t, :]
            rec_input = self.input2h(curr_input) + self.h2h(torch.relu(h))
            h = (1 - self.alpha) * h + self.alpha * rec_input
            outputs.append(self.h2out(h))
        return torch.stack(outputs, dim=1)

# --- 5. 训练与评估函数 ---
def train_and_evaluate(model_class, model_name, epochs=500): # 减少 epoch 以节省演示时间，实际可设为 1000
    print(f"\n>>> 开始训练 {model_name} (Realistic Environment)...")
    model = model_class(MAX_INPUT, HIDDEN_SIZE, MAX_OUTPUT, NUM_TASKS)
    if torch.cuda.is_available(): model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    for epoch in range(epochs + 1):
        model.train()
        total_loss = 0
        for i in range(NUM_TASKS):
            obs, target, t_id = get_batch(i, seed_offset=epoch)
            if torch.cuda.is_available():
                obs, target, t_id = obs.cuda(), target.cuda(), t_id.cuda()
            
            output = model(obs, t_id)
            loss = criterion_ce(output.reshape(-1, MAX_OUTPUT), target.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Avg Loss: {total_loss/NUM_TASKS:.4f}")

    print(f"\n>>> {model_name} 最终评估 (Realistic Tasks)")
    print(f"{'Task Name':<30} | {'F1 Score':<10} | {'MSE Loss':<10}")
    print("-" * 60)
    
    model.eval()
    results = {'f1': [], 'mse': []}
    
    for i, t_name in enumerate(tasks):
        obs, target, t_id = get_batch(i, batch_size=50, seed_offset=9999) # 使用不同的 seed offset 进行测试
        if torch.cuda.is_available():
            obs, target, t_id = obs.cuda(), target.cuda(), t_id.cuda()
            
        with torch.no_grad():
            output = model(obs, t_id)
            pred = output.argmax(dim=-1).cpu().numpy().flatten()
            target_np = target.cpu().numpy().flatten()
            
            f1 = f1_score(target_np, pred, average='macro')
            
            target_oh = F.one_hot(target, num_classes=MAX_OUTPUT).float()
            mse = criterion_mse(F.softmax(output, dim=-1), target_oh).item()
            
            print(f"{t_name:<30} | {f1:.4f}     | {mse:.6f}")
            results['f1'].append(f1)
            results['mse'].append(mse)
            
    return results

import json
import os

# --- 6. 主程序 ---
if __name__ == "__main__":
    results_lstm = train_and_evaluate(CognitiveLSTM, "LSTM")
    results_gru = train_and_evaluate(CognitiveGRU, "GRU")
    results_ctrnn = train_and_evaluate(CTRNN, "CTRNN")
    
    # 简单的汇总对比
    print("\n>>> 模型性能汇总 (平均 F1 Score)")
    print(f"LSTM : {np.mean(results_lstm['f1']):.4f}")
    print(f"GRU  : {np.mean(results_gru['f1']):.4f}")
    print(f"CTRNN: {np.mean(results_ctrnn['f1']):.4f}")

    # 保存结果供可视化使用
    final_results = {
        "tasks": tasks,
        "LSTM": results_lstm,
        "GRU": results_gru,
        "CTRNN": results_ctrnn
    }
    
    output_file = os.path.join(os.path.dirname(__file__), 'results_realistic.json')
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\n>>> 实验结果已保存至: {output_file}")
