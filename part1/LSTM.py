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

# --- 2. 任务准备 ---
tasks = ['AntiReach-v0', 'ContextDecisionMaking-v0', 'DelayComparison-v0',
         'DelayMatchCategory-v0', 'DelayMatchSample-v0', 'DelayMatchSampleDistractor1D-v0',
         'DelayPairedAssociation-v0', 'DualDelayMatchSample-v0', 'EconomicDecisionMaking-v0', 'GoNogo-v0']

envs = [gym.make(t) for t in tasks]
for env in envs: env.action_space.seed(42) # 锁定动作空间采样

MAX_INPUT, MAX_OUTPUT, HIDDEN_SIZE, NUM_TASKS, MAX_SEQ_LEN = 33, 33, 128, 10, 50

# --- 3. 稳健的数据采集 (固定种子采样) ---
def get_batch(task_idx, batch_size=16, seed_offset=0):
    env = envs[task_idx]
    batch_obs = np.zeros((batch_size, MAX_SEQ_LEN, MAX_INPUT))
    batch_targets = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)
    for b in range(batch_size):
        obs, _ = env.reset(seed=task_idx * 1000 + seed_offset + b)
        for t in range(MAX_SEQ_LEN):
            batch_obs[b, t, :obs.shape[0]] = obs
            obs, _, _, _, info = env.step(env.action_space.sample())
            gt = info.get('gt', getattr(env.unwrapped, 'gt', [0]*MAX_SEQ_LEN)[t % 5])
            batch_targets[b, t] = gt
    task_onehot = np.zeros((batch_size, NUM_TASKS)); task_onehot[:, task_idx] = 1
    return torch.FloatTensor(batch_obs), torch.LongTensor(batch_targets), torch.FloatTensor(task_onehot)

# --- 4. LSTM 模型定义 ---
class CognitiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks):
        super().__init__()
        self.lstm = nn.LSTM(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, task_id):
        task_info = task_id.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.lstm(torch.cat((x, task_info), dim=2))
        return self.fc(out)

# --- 5. 训练与评估 ---
model = CognitiveLSTM(MAX_INPUT, HIDDEN_SIZE, MAX_OUTPUT, NUM_TASKS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()



print("\n>>> epoch为1000，开始 LSTM 并行训练...")
for epoch in range(1001):
    model.train(); total_loss = 0
    for i in range(NUM_TASKS):
        obs, target, t_id = get_batch(i, seed_offset=epoch)
        output = model(obs, t_id)
        loss = criterion_ce(output.view(-1, MAX_OUTPUT), target.view(-1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    if epoch % 100 == 0: print(f"Epoch {epoch:4d} | Avg Loss: {total_loss/NUM_TASKS:.4f}")

# 最终评估报表
print("\n" + "="*85)
print(f"{'任务名称 (LSTM)':<40} | {'F1 分数':<10} | {'MSE 损失':<10}")
print("-" * 85)
model.eval()
for i, t_name in enumerate(tasks):
    obs, target, t_id = get_batch(i, batch_size=20, seed_offset=999)
    with torch.no_grad():
        output = model(obs, t_id)
        pred = output.argmax(dim=-1).numpy().flatten()
        f1 = f1_score(target.numpy().flatten(), pred, average='macro')
        target_oh = F.one_hot(target, num_classes=MAX_OUTPUT).float()
        mse = criterion_mse(F.softmax(output, dim=-1), target_oh).item()
        print(f"{t_name:<40} | {f1:.4f}     | {mse:.6f}")