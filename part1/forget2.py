import torch
import torch.nn as nn
import gymnasium as gym
import neurogym as ngym
import numpy as np
import random
from sklearn.metrics import f1_score
import warnings

# 彻底屏蔽警告
warnings.filterwarnings("ignore")

# --- 1. 核心：彻底杀死随机性的函数 ---
def set_seed(seed=42):
    random.seed(seed) # 锁定 Python 基础随机库
    np.random.seed(seed) # 锁定 NumPy
    torch.manual_seed(seed) # 锁定 PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 锁定 PyTorch GPU
    # 锁定计算算子，确保每次矩阵运算结果完全一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> 随机种子已锁定为: {seed}")

set_seed(42) # 使用 42 作为幸运数字

# --- 2. 任务配置 ---
tasks_to_learn = ['GoNogo-v0', 'DelayComparison-v0', 'DelayMatchSample-v0']
MAX_INPUT, MAX_OUTPUT = 10, 10
HIDDEN_SIZE = 128
NUM_TASKS = len(tasks_to_learn)
MAX_SEQ_LEN = 50

# 初始化环境并为每个环境设置种子
envs = []
for i, t_name in enumerate(tasks_to_learn):
    env = gym.make(t_name)
    # 给环境分配固定种子，确保每个任务生成的 Trial 序列是确定的
    envs.append(env)

# --- 3. 稳健的数据采集函数 ---
def get_batch(task_idx, batch_size=16, seed_offset=0):
    env = envs[task_idx]
    batch_obs = np.zeros((batch_size, MAX_SEQ_LEN, MAX_INPUT))
    batch_targets = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)

    for b in range(batch_size):
        # 确保数据采集的过程也是可复现的
        obs, _ = env.reset(seed=task_idx + b + seed_offset)
        for t in range(MAX_SEQ_LEN):
            batch_obs[b, t, :obs.shape[0]] = obs
            obs, _, _, _, info = env.step(env.action_space.sample())
            gt = info.get('gt', getattr(env.unwrapped, 'gt', [0] * MAX_SEQ_LEN)[t % 5])
            batch_targets[b, t] = gt

    task_onehot = np.zeros((batch_size, NUM_TASKS))
    task_onehot[:, task_idx] = 1
    return torch.FloatTensor(batch_obs), torch.LongTensor(batch_targets), torch.FloatTensor(task_onehot)

# --- 4. 模型定义 (LSTM) ---
class CognitiveRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks):
        super().__init__()
        self.lstm = nn.LSTM(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, task_id_onehot):
        batch_size, seq_len, _ = x.size()
        task_info = task_id_onehot.unsqueeze(1).repeat(1, seq_len, 1)
        combined_input = torch.cat((x, task_info), dim=2)
        out, _ = self.lstm(combined_input)
        return self.fc(out)

# --- 5. 初始化与评估逻辑 ---
model = CognitiveRNN(MAX_INPUT, HIDDEN_SIZE, MAX_OUTPUT, NUM_TASKS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 降低一点学习率让遗忘更平滑
criterion = nn.CrossEntropyLoss()

def evaluate(task_idx):
    model.eval()
    # 使用固定的种子进行评估，保证每次测试的“试卷”是一样的
    obs, target, task_id = get_batch(task_idx, batch_size=32, seed_offset=1000)
    with torch.no_grad():
        output = model(obs, task_id)
        pred = output.argmax(dim=-1).numpy().flatten()
        true = target.numpy().flatten()
    return f1_score(true, pred, average='macro')

# --- 6. 顺序学习过程 ---
print("\n>>> 开始确定性顺序学习演示（演示灾难性遗忘）")

for i, task_name in enumerate(tasks_to_learn):
    print(f"\n[阶段 {i + 1}] 正在训练任务: {task_name} ...")

    for epoch in range(1001):
        model.train()
        # 训练时动态调整种子偏移，保证虽然确定但能看到多
        obs, target, task_id = get_batch(i, seed_offset=epoch)

        output = model(obs, task_id)
        loss = criterion(output.view(-1, MAX_OUTPUT), target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"  Epoch {epoch} | Loss: {loss.item():.4f}")

    print(f"\n>> 阶段 {i + 1} 结束性能报告：")
    print(f"{'已学习的任务':<25} | {'F1 分数':<10} | {'状态':<15}")
    print("-" * 60)
    for j in range(i + 1):
        f1 = evaluate(j)
        if j == i: status = "刚刚学会"
        else: status = "!!! 遗忘测试 !!!"
        print(f"{tasks_to_learn[j]:<25} | {f1:.4f}     | {status}")
    print("-" * 60)

print("\n【演示结果深度分析】")
print("1. 观察 GoNogo-v0 在阶段 1 为 1.0，阶段 2 暴跌？")
print("2. 阶段 3 即使有微弱回升（Backward Transfer），这个数值也将是变化不大的。")