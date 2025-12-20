import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import neurogym as ngym
import numpy as np
import random
from sklearn.metrics import f1_score
import warnings

# 彻底屏蔽所有干扰警告
warnings.filterwarnings("ignore")


# --- 1. 彻底杀死随机性（确保科研结果可复现） ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 锁定计算算子，保证底层矩阵运算的一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> 确定性模式已开启，随机种子: {seed}")


set_seed(42)

# --- 2. 任务准备（筛选 10 个认知任务） ---
tasks = ['AntiReach-v0', 'ContextDecisionMaking-v0', 'DelayComparison-v0',
         'DelayMatchCategory-v0', 'DelayMatchSample-v0', 'DelayMatchSampleDistractor1D-v0',
         'DelayPairedAssociation-v0', 'DualDelayMatchSample-v0', 'EconomicDecisionMaking-v0', 'GoNogo-v0']

envs = [gym.make(t) for t in tasks]
for env in envs:
    env.action_space.seed(42)  # 锁定动作空间采样

# 统一对齐维度
MAX_INPUT, MAX_OUTPUT = 33, 33  # 兼容 AntiReach
HIDDEN_SIZE = 128
NUM_TASKS = len(tasks)
MAX_SEQ_LEN = 50


# --- 3. 稳健的数据采集（固定种子采样） ---
def get_batch(task_idx, batch_size=16, seed_offset=0):
    env = envs[task_idx]
    batch_obs = np.zeros((batch_size, MAX_SEQ_LEN, MAX_INPUT))
    batch_targets = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)

    for b in range(batch_size):
        # 锁定环境生成的随机性
        obs, _ = env.reset(seed=task_idx * 1000 + seed_offset + b)
        for t in range(MAX_SEQ_LEN):
            batch_obs[b, t, :obs.shape[0]] = obs
            obs, _, _, _, info = env.step(env.action_space.sample())
            # 容错获取 Ground Truth (gt)
            gt = info.get('gt', getattr(env.unwrapped, 'gt', [0] * MAX_SEQ_LEN)[t % 5])
            batch_targets[b, t] = gt

    # 生成任务标识 (One-hot)
    task_onehot = np.zeros((batch_size, NUM_TASKS))
    task_onehot[:, task_idx] = 1

    return torch.FloatTensor(batch_obs), torch.LongTensor(batch_targets), torch.FloatTensor(task_onehot)


# --- 4. CTRNN 模型定义 ---
#
class CTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks, dt=1.0, tau=10.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau = tau
        self.alpha = dt / tau  # 时间常数相关的缩放因子

        # 权重定义
        self.input2h = nn.Linear(input_size + num_tasks, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2out = nn.Linear(hidden_size, output_size)

    def forward(self, x, task_id):
        batch_size, seq_len, _ = x.size()
        task_info = task_id.unsqueeze(1).repeat(1, seq_len, 1)
        combined_input = torch.cat((x, task_info), dim=2)

        # 初始化隐藏状态（神经元膜电位）
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        outputs = []

        # 使用欧拉法（Euler Method）迭代微分方程
        for t in range(seq_len):
            curr_input = combined_input[:, t, :]
            # CTRNN 动力学方程: h = (1 - alpha) * h + alpha * f(W_in*x + W_rec*h)
            # 这里 f 使用 ReLU 或 Tanh 模拟激活函数
            rec_input = self.input2h(curr_input) + self.h2h(torch.relu(h))
            h = (1 - self.alpha) * h + self.alpha * rec_input

            y = self.h2out(torch.relu(h))
            outputs.append(y)

        return torch.stack(outputs, dim=1)


# --- 5. 训练循环 ---
model = CTRNN(MAX_INPUT, HIDDEN_SIZE, MAX_OUTPUT, NUM_TASKS, dt=1.0, tau=10.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()

print("\n>>> 开始 CTRNN 多任务并行训练 (1000 Epochs) ...")
for epoch in range(1001):
    model.train()
    total_loss = 0
    for i in range(NUM_TASKS):
        obs, target, t_id = get_batch(i, seed_offset=epoch)

        output = model(obs, t_id)
        # 计算分类交叉熵损失
        loss = criterion_ce(output.view(-1, MAX_OUTPUT), target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | 平均 CE Loss: {total_loss / NUM_TASKS:.4f}")

# --- 6. 最终评估报表 (F1 分数 & MSE 损失) ---
print("\n" + "=" * 85)
print(f"{'任务名称 (CTRNN)':<40} | {'F1 分数':<10} | {'MSE 损失':<10}")
print("-" * 85)

model.eval()
for i, t_name in enumerate(tasks):
    # 使用评估专用种子
    obs, target, t_id = get_batch(i, batch_size=20, seed_offset=999)
    with torch.no_grad():
        output = model(obs, t_id)

        # 指标 1: F1 分数
        pred = output.argmax(dim=-1).numpy().flatten()
        f1 = f1_score(target.numpy().flatten(), pred, average='macro')

        # 指标 2: MSE 损失（基于概率分布）
        target_oh = F.one_hot(target, num_classes=MAX_OUTPUT).float()
        prob_out = F.softmax(output, dim=-1)
        mse = criterion_mse(prob_out, target_oh).item()

        print(f"{t_name:<40} | {f1:.4f}     | {mse:.6f}")
print("=" * 85)