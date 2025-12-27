import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import neurogym as ngym
import numpy as np
import random
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

# ==================== 统一配置 ====================
# 任务配置
TASKS = ['GoNogo-v0', 'AntiReach-v0', 'DelayComparison-v0', 'DelayMatchSample-v0']
NUM_TASKS = len(TASKS)

# 模型配置
MAX_INPUT = 33
MAX_OUTPUT = 33
HIDDEN_SIZE = 128
MAX_SEQ_LEN = 50

# 训练配置
EPOCHS_PER_TASK = 600
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SEED = 42

# 评估配置
EVAL_BATCH_SIZE = 32
EVAL_SEED_OFFSET = 9999

# EWC特定配置
EWC_LAMBDA = 100
EWC_FISHER_SAMPLES = 200

# ==================== 随机种子设置 ====================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> 确定性模式已开启，随机种子: {seed}")

set_seed(SEED)

# ==================== 环境初始化 ====================
envs = [gym.make(t) for t in TASKS]
for env in envs:
    env.action_space.seed(SEED)

# ==================== 数据采集函数 ====================
def get_batch(task_idx, batch_size=BATCH_SIZE, seed_offset=0):
    """统一的数据采集函数"""
    env = envs[task_idx]
    obs_batch = np.zeros((batch_size, MAX_SEQ_LEN, MAX_INPUT))
    target_batch = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)

    for b in range(batch_size):
        rst = env.reset(seed=task_idx * 1000 + seed_offset + b)
        obs = rst[0] if isinstance(rst, tuple) else rst
        
        for t in range(MAX_SEQ_LEN):
            obs_batch[b, t, :obs.shape[0]] = obs
            step_ret = env.step(env.action_space.sample())
            if len(step_ret) == 4:
                obs, _, _, info = step_ret
            else:
                obs, _, _, _, info = step_ret
            gt = info.get("gt", 0)
            target_batch[b, t] = gt

    task_id = np.zeros((batch_size, NUM_TASKS))
    task_id[:, task_idx] = 1.0

    return (
        torch.tensor(obs_batch, dtype=torch.float32),
        torch.tensor(target_batch, dtype=torch.long),
        torch.tensor(task_id, dtype=torch.float32),
    )

# ==================== EWC类 ====================
# ==================== EWC类 (增加了数值归一化) ====================
class EWC:
    def __init__(self, model, num_samples=EWC_FISHER_SAMPLES):
        self.model = model
        self.num_samples = num_samples
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fisher = {}
        self.clamp_max = 5.0  # 设置最大截断阈值，防止惩罚项过大
        self.eps = 1e-8       # 防止除零

    def register_old_task(self, task_idx):
        """计算Fisher信息矩阵并保存当前权重"""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        # 1. 采样数据计算梯度平方期望
        for s in range(self.num_samples):
            # 注意：这里需要确保 get_batch 函数在外部是可见的，或者传入
            obs, target, task_id = get_batch(task_idx, batch_size=1, seed_offset=s)
            self.model.zero_grad()
            output = self.model(obs, task_id)
            loss = F.cross_entropy(output.view(-1, MAX_OUTPUT), target.view(-1))
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n].data += p.grad.data ** 2 / self.num_samples

        # ==================== 新增：数值归一化与截断 ====================
        # 2. 计算全局均值
        # 将所有参数的Fisher值展平并拼接，计算平均值
        all_vals = torch.cat([f.view(-1) for f in fisher.values()])
        mean_val = all_vals.mean().clamp_min(self.eps) 
        
        # 3. 归一化 + 截断
        # 让Fisher矩阵的数值分布更稳定，不再受梯度绝对大小的影响
        for n in fisher:
            fisher[n] = fisher[n] / mean_val  # 归一化
            fisher[n] = fisher[n].clamp(0.0, self.clamp_max) # 截断
        # ==========================================================

        self._fisher[task_idx] = fisher
        self._means[task_idx] = {n: p.clone().detach() for n, p in self.params.items()}

    def ewc_loss(self):
        """计算所有旧任务的EWC惩罚项"""
        loss = 0
        for task_idx in self._fisher.keys():
            for n, p in self.model.named_parameters():
                # 这里的 fisher 已经是归一化后的，数值通常在 0~5 之间
                # 配合 lambda=1000~2000 使用效果较好
                _loss = self._fisher[task_idx][n] * (p - self._means[task_idx][n]) ** 2
                loss += _loss.sum()
        return loss
# ==================== 模型定义 ====================
class CognitiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks):
        super().__init__()
        self.lstm = nn.LSTM(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, task_id):
        task_info = task_id.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.lstm(torch.cat((x, task_info), dim=2))
        return self.fc(out)

# ==================== 评估函数 ====================
def evaluate(model, task_idx):
    """统一的评估函数"""
    model.eval()
    obs, target, task_id = get_batch(task_idx, batch_size=EVAL_BATCH_SIZE, seed_offset=EVAL_SEED_OFFSET)
    with torch.no_grad():
        output = model(obs, task_id)
        pred = output.argmax(dim=-1).numpy().flatten()
        true = target.numpy().flatten()
    return f1_score(true, pred, average='macro')

# ==================== 训练初始化 ====================
model = CognitiveLSTM(MAX_INPUT, HIDDEN_SIZE, MAX_OUTPUT, NUM_TASKS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
ewc = EWC(model)

# 性能追踪矩阵
results_matrix = np.zeros((NUM_TASKS, NUM_TASKS))

# ==================== 顺序连续学习 + EWC ====================
print("\n" + "="*70)
print("方法: EWC连续学习 (Elastic Weight Consolidation)")
print(f"配置: λ={EWC_LAMBDA}, Fisher样本数={EWC_FISHER_SAMPLES}")
print("="*70)

for i in range(NUM_TASKS):
    current_task = TASKS[i]
    print(f"\n[阶段 {i+1}/{NUM_TASKS}] 训练任务: {current_task}")
    
    # 训练当前任务
    for epoch in range(EPOCHS_PER_TASK + 1):
        model.train()
        obs, target, task_id = get_batch(i, seed_offset=epoch)
        
        output = model(obs, task_id)
        loss_ce = F.cross_entropy(output.view(-1, MAX_OUTPUT), target.view(-1))
        
        # 计算EWC损失
        loss_ewc = ewc.ewc_loss()
        total_loss = loss_ce + EWC_LAMBDA * loss_ewc
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            ewc_penalty = EWC_LAMBDA * (loss_ewc.item() if torch.is_tensor(loss_ewc) else loss_ewc)
            print(f"  Epoch {epoch:4d} | CE Loss: {loss_ce.item():.4f} | EWC Penalty: {ewc_penalty:.4f}")

    # 注册为旧任务
    ewc.register_old_task(i)
    print(f"  → 任务 {current_task} 已注册到EWC保护")

    # 阶段性评估所有任务
    print(f"\n--- 阶段 {i+1} 评估结果 ---")
    for j in range(NUM_TASKS):
        f1 = evaluate(model, j)
        results_matrix[i, j] = f1
        status = "✓已学" if j <= i else "✗未学"
        print(f"  {TASKS[j]:<25} {status} | F1: {f1:.4f}")

# ==================== 最终分析 ====================
print("\n" + "="*70)
print(">>> 最终结果分析 <<<")
print("="*70)
print("\n性能矩阵 (行=学习阶段, 列=测试任务):")
print(results_matrix)

# 计算遗忘指标
forgetting_scores = []
for j in range(NUM_TASKS - 1):
    best_performance = results_matrix[j, j]
    final_performance = results_matrix[NUM_TASKS-1, j]
    forgetting = best_performance - final_performance
    forgetting_scores.append(forgetting)
    print(f"\n任务 {TASKS[j]}: 最佳={best_performance:.4f}, 最终={final_performance:.4f}, 遗忘={forgetting:.4f}")

avg_forgetting = np.mean(forgetting_scores)
print(f"\n平均遗忘程度 (Average Forgetting): {avg_forgetting:.4f}")
print(f"最终平均性能: {results_matrix[-1, :].mean():.4f}")
print("="*70)