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


# --- 1. 彻底杀死随机性（确保结果可复现） ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 锁定计算算子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> 确定性模式已开启，随机种子: {seed}")


set_seed(42)

# --- 2. 任务准备（筛选 10 个核心认知任务） ---
tasks = ['AntiReach-v0', 'ContextDecisionMaking-v0', 'DelayComparison-v0',
         'DelayMatchCategory-v0', 'DelayMatchSample-v0', 'DelayMatchSampleDistractor1D-v0',
         'DelayPairedAssociation-v0', 'DualDelayMatchSample-v0', 'EconomicDecisionMaking-v0', 'GoNogo-v0']

# 初始化环境并锁定动作空间随机性
envs = [gym.make(t) for t in tasks]
for env in envs:
    env.action_space.seed(42)

# 统一对齐维度
MAX_INPUT, MAX_OUTPUT = 33, 33  # 兼容所有任务的特征维度
HIDDEN_SIZE = 128
NUM_TASKS = len(tasks)
MAX_SEQ_LEN = 50


# --- 3. 稳健的数据采集（固定种子采样） ---
def get_batch(task_idx, batch_size=16, seed_offset=0):
    env = envs[task_idx]
    batch_obs = np.zeros((batch_size, MAX_SEQ_LEN, MAX_INPUT))
    batch_targets = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)

    for b in range(batch_size):
        # 训练和评估时使用特定种子，确保“试卷”内容固定
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


# --- 4. GRU 模型定义 ---

class CognitiveGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks):
        super().__init__()
        # 使用更新门 (Update Gate) 和重置门 (Reset Gate) 维护记忆
        self.gru = nn.GRU(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, task_id):
        # x 形状: (batch, seq, features)
        # task_id 形状: (batch, num_tasks)
        task_info = task_id.unsqueeze(1).repeat(1, x.size(1), 1)
        # 拼接原始输入与任务标识信号
        out, _ = self.gru(torch.cat((x, task_info), dim=2))
        return self.fc(out)


# --- 5. 训练循环 ---
model = CognitiveGRU(MAX_INPUT, HIDDEN_SIZE, MAX_OUTPUT, NUM_TASKS)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()

print("\n>>> 开始 GRU 多任务并行训练 (1000 Epochs) ...")
for epoch in range(1001):
    model.train()
    total_loss = 0
    for i in range(NUM_TASKS):
        obs, target, t_id = get_batch(i, seed_offset=epoch)

        output = model(obs, t_id)
        # 交叉熵用于训练分类性能
        loss = criterion_ce(output.view(-1, MAX_OUTPUT), target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | 平均 CE Loss: {total_loss / NUM_TASKS:.4f}")

# --- 6. 最终评估报表 (F1 分数 & MSE 损失) ---
print("\n" + "=" * 85)
print(f"{'任务名称 (GRU)':<40} | {'F1 分数':<10} | {'MSE 损失':<10}")
print("-" * 85)

model.eval()
for i, t_name in enumerate(tasks):
    # 使用评估专用种子，确保结果的唯一性
    obs, target, t_id = get_batch(i, batch_size=20, seed_offset=999)
    with torch.no_grad():
        output = model(obs, t_id)

        # 指标 1: F1 分数（衡量决策是非）
        pred = output.argmax(dim=-1).numpy().flatten()
        f1 = f1_score(target.numpy().flatten(), pred, average='macro')

        # 指标 2: MSE 损失（衡量输出概率的精度）
        target_onehot = F.one_hot(target, num_classes=MAX_OUTPUT).float()
        # 将输出通过 Softmax 转为概率分布后再与 One-hot 标签计算 MSE
        mse = criterion_mse(F.softmax(output, dim=-1), target_onehot).item()

        print(f"{t_name:<40} | {f1:.4f}     | {mse:.6f}")
print("=" * 85)