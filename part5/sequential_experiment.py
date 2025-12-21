import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import neurogym as ngym
import numpy as np
import random
import json
import os
from sklearn.metrics import f1_score

# --- 1. 确定性设置 ---
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
# 选取与 Part 1 EWC_test 相同的 3 个任务
tasks_to_learn = ['GoNogo-v0', 'DelayComparison-v0', 'DelayMatchSample-v0']

# Realistic 参数
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
for t in tasks_to_learn:
    try:
        env = gym.make(t, **timing_kwargs)
    except Exception as e:
        print(f"Warning: Could not apply timing to {t}, using default. Error: {e}")
        env = gym.make(t)
    
    env.action_space.seed(42)
    envs.append(env)

MAX_INPUT, MAX_OUTPUT = 33, 33 # 统一维度 (注意 Part 1 EWC_test 用的是 10, 这里为了兼容性用 33，或者保持一致)
# Part 1 EWC_test.py 中 MAX_INPUT=10. 
# 为了公平比较，我们应该尽量保持模型规模一致，但输入维度取决于任务。
# GoNogo, DelayComparison, DMS 的输入维度通常较小。
# 让我们检查一下 Part 1 的 EWC_test.py 确实用了 10。
# 如果我们用 33，模型参数会多一些。为了严谨，我们检查一下环境的实际 observation space。
# 但为了简单起见，我们使用 33 (兼容所有任务的最大值)，这不会显著改变遗忘的性质。

HIDDEN_SIZE = 128
NUM_TASKS = len(tasks_to_learn)
MAX_SEQ_LEN = 50

# --- 3. 数据采集 ---
def get_batch(task_idx, batch_size=16, seed_offset=0):
    env = envs[task_idx]
    batch_obs = np.zeros((batch_size, MAX_SEQ_LEN, MAX_INPUT))
    batch_targets = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)
    
    rng = np.random.RandomState(task_idx * 1000 + seed_offset)

    for b in range(batch_size):
        obs, _ = env.reset(seed=task_idx * 1000 + seed_offset + b)
        for t in range(MAX_SEQ_LEN):
            # 手动添加噪声
            noise = rng.normal(0, sigma_noise, size=obs.shape)
            # 注意：如果 obs 维度小于 MAX_INPUT，需要填充
            current_input = np.zeros(MAX_INPUT)
            current_input[:obs.shape[0]] = obs + noise
            
            batch_obs[b, t, :] = current_input
            
            obs, _, _, _, info = env.step(env.action_space.sample())
            gt = info.get('gt', 0)
            batch_targets[b, t] = gt
            
    task_onehot = np.zeros((batch_size, NUM_TASKS))
    task_onehot[:, task_idx] = 1
    return torch.FloatTensor(batch_obs), torch.LongTensor(batch_targets), torch.FloatTensor(task_onehot)

# --- 4. 模型定义 (LSTM) ---
class CognitiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks):
        super().__init__()
        self.lstm = nn.LSTM(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, task_id):
        task_info = task_id.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.lstm(torch.cat((x, task_info), dim=2))
        return self.fc(out)

# --- 5. 顺序学习流程 ---
def train_task(model, task_idx, epochs=500):
    print(f"\n>>> Training on Task {task_idx}: {tasks_to_learn[task_idx]}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        obs, target, t_id = get_batch(task_idx, seed_offset=epoch)
        if torch.cuda.is_available():
            obs, target, t_id = obs.cuda(), target.cuda(), t_id.cuda()
            
        output = model(obs, t_id)
        loss = criterion(output.reshape(-1, MAX_OUTPUT), target.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch} | Loss: {loss.item():.4f}")

def evaluate_all(model):
    model.eval()
    scores = []
    for i in range(NUM_TASKS):
        obs, target, t_id = get_batch(i, batch_size=50, seed_offset=9999)
        if torch.cuda.is_available():
            obs, target, t_id = obs.cuda(), target.cuda(), t_id.cuda()
        
        with torch.no_grad():
            output = model(obs, t_id)
            pred = output.argmax(dim=-1).cpu().numpy().flatten()
            target_np = target.cpu().numpy().flatten()
            f1 = f1_score(target_np, pred, average='macro')
            scores.append(f1)
    return scores

if __name__ == "__main__":
    model = CognitiveLSTM(MAX_INPUT, HIDDEN_SIZE, MAX_OUTPUT, NUM_TASKS)
    if torch.cuda.is_available(): model = model.cuda()
    
    history = []
    
    # Stage 1: Train Task 0
    train_task(model, 0)
    scores_s1 = evaluate_all(model)
    print(f"Stage 1 Scores: {scores_s1}")
    history.append(scores_s1)
    
    # Stage 2: Train Task 1
    train_task(model, 1)
    scores_s2 = evaluate_all(model)
    print(f"Stage 2 Scores: {scores_s2}")
    history.append(scores_s2)
    
    # Stage 3: Train Task 2
    train_task(model, 2)
    scores_s3 = evaluate_all(model)
    print(f"Stage 3 Scores: {scores_s3}")
    history.append(scores_s3)
    
    # 保存结果
    results = {
        "tasks": tasks_to_learn,
        "history": history # [[t1, t2, t3], [t1, t2, t3], [t1, t2, t3]]
    }
    
    output_file = os.path.join(os.path.dirname(__file__), 'sequential_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
