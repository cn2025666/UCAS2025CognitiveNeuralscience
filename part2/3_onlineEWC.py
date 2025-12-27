import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import neurogym as ngym
import numpy as np
import random
from sklearn.metrics import f1_score
import warnings
from typing import Dict, List, Tuple

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

# Online EWC特定配置
ONLINE_EWC_LAMBDA = 200
ONLINE_EWC_FISHER_SAMPLES = 256
ONLINE_EWC_MEMORY_SIZE = 256
ONLINE_EWC_FISHER_BATCH_SIZE = 16
ONLINE_EWC_FISHER_CLAMP = 5.0
ONLINE_EWC_DECAY = 1.0
ONLINE_EWC_WARMUP_EPOCHS = 300

EPS = 1e-8

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

# ==================== 固定样本缓存 ====================
def build_task_memory(task_idx: int, total: int = ONLINE_EWC_MEMORY_SIZE, seed: int = 2025):
    """为某个任务采样固定数据，保存用于Fisher估计"""
    obs_list, tgt_list, tid_list = [], [], []
    bs = 32
    offset = seed + task_idx * 10000
    while len(obs_list) < total:
        obs, tgt, tid = get_batch(task_idx, batch_size=bs, seed_offset=offset + len(obs_list))
        for b in range(obs.size(0)):
            obs_list.append(obs[b].detach().clone())
            tgt_list.append(tgt[b].detach().clone())
            tid_list.append(tid[b].detach().clone())
            if len(obs_list) >= total:
                break
    return (torch.stack(obs_list), torch.stack(tgt_list), torch.stack(tid_list))

def iter_memory_batches(mem_obs, mem_tgt, mem_tid, batch_size: int):
    """迭代memory批次"""
    n = mem_obs.size(0)
    idx = torch.randperm(n)
    for s in range(0, n, batch_size):
        j = idx[s:s + batch_size]
        yield mem_obs[j], mem_tgt[j], mem_tid[j]

# ==================== Online EWC类 ====================
class OnlineEWC:
    def __init__(self, model: nn.Module):
        self.model = model
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}
        self.is_ready = False

    @torch.no_grad()
    def _snapshot_means(self):
        self._means = {n: p.detach().clone() for n, p in self.params.items()}

    def estimate_fisher_from_memory(self, mem_obs, mem_tgt, mem_tid,
                                    num_samples: int = ONLINE_EWC_FISHER_SAMPLES,
                                    batch_size: int = ONLINE_EWC_FISHER_BATCH_SIZE):
        """用固定memory估计对角Fisher"""
        self.model.eval()
        fisher = {n: torch.zeros_like(p, device=p.device) for n, p in self.params.items()}

        # 取前num_samples条（可复现）
        mem_obs = mem_obs[:num_samples]
        mem_tgt = mem_tgt[:num_samples]
        mem_tid = mem_tid[:num_samples]

        total_seen = 0
        for obs_b, tgt_b, tid_b in iter_memory_batches(mem_obs, mem_tgt, mem_tid, batch_size=batch_size):
            self.model.zero_grad(set_to_none=True)
            out = self.model(obs_b, tid_b)
            loss = F.cross_entropy(out.view(-1, MAX_OUTPUT), tgt_b.view(-1), reduction="mean")
            loss.backward()

            bs = obs_b.size(0)
            total_seen += bs
            for n, p in self.params.items():
                if p.grad is None:
                    continue
                fisher[n] += (p.grad.detach() ** 2) * bs

        # 归一化为均值
        for n in fisher:
            fisher[n] /= max(total_seen, 1)

        # 尺度控制：clamp + mean normalize
        all_vals = torch.cat([f.view(-1) for f in fisher.values()])
        mean_val = all_vals.mean().clamp_min(EPS)

        for n in fisher:
            fisher[n] = fisher[n] / mean_val
            fisher[n] = fisher[n].clamp(0.0, ONLINE_EWC_FISHER_CLAMP)

        return fisher
    
    def consolidate(self, fisher_new: Dict[str, torch.Tensor], decay: float = ONLINE_EWC_DECAY):
        """将新任务的Fisher合并进累计Fisher"""
        if not self.is_ready:
            self._fisher = {n: f.detach().clone() for n, f in fisher_new.items()}
            self._snapshot_means()
            self.is_ready = True
            return

        # online合并
        for n in self._fisher:
            self._fisher[n] = decay * self._fisher[n] + fisher_new[n]
        self._snapshot_means()

    def penalty(self):
        if not self.is_ready:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        loss = 0.0
        for n, p in self.params.items():
            loss = loss + (self._fisher[n] * (p - self._means[n]) ** 2).sum()
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

def evaluate_fixed_memory(model, task_idx: int, task_memory: List):
    """用固定memory做评估，避免每次get_batch随机导致指标抖动"""
    model.eval()
    mem_obs, mem_tgt, mem_tid = task_memory[task_idx]
    with torch.no_grad():
        out = model(mem_obs, mem_tid)
        pred = out.argmax(dim=-1).reshape(-1).cpu().numpy()
        true = mem_tgt.reshape(-1).cpu().numpy()
    return f1_score(true, pred, average="macro")

# ==================== 训练初始化 ====================
model = CognitiveLSTM(MAX_INPUT, HIDDEN_SIZE, MAX_OUTPUT, NUM_TASKS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
ewc = OnlineEWC(model)

# 性能追踪矩阵
results_matrix = np.zeros((NUM_TASKS, NUM_TASKS))

# 预先构建所有任务的固定memory
print(">>> 构建固定任务memory用于Fisher估计...")
task_memory = []
for ti in range(NUM_TASKS):
    mem = build_task_memory(ti)
    task_memory.append(mem)
    print(f"  任务 {TASKS[ti]}: 已缓存 {mem[0].size(0)} 条序列")

# ==================== 顺序连续学习 + Online EWC ====================
print("\n" + "="*70)
print("方法: Online EWC连续学习")
print(f"配置: λ={ONLINE_EWC_LAMBDA}, Fisher样本={ONLINE_EWC_FISHER_SAMPLES}, ")
print(f"      Memory={ONLINE_EWC_MEMORY_SIZE}, Decay={ONLINE_EWC_DECAY}")
print("="*70)

for i in range(NUM_TASKS):
    current_task = TASKS[i]
    print(f"\n[阶段 {i+1}/{NUM_TASKS}] 训练任务: {current_task}")
    
    # 训练当前任务
    for epoch in range(EPOCHS_PER_TASK + 1):
        model.train()
        obs, target, task_id = get_batch(i, seed_offset=epoch)
        output = model(obs, task_id)
        
        loss_ce = criterion(output.view(-1, MAX_OUTPUT), target.view(-1))
        
        # λ warm-up：前ONLINE_EWC_WARMUP_EPOCHS个epoch从0线性到ONLINE_EWC_LAMBDA
        if epoch < ONLINE_EWC_WARMUP_EPOCHS:
            lam = ONLINE_EWC_LAMBDA * (epoch / ONLINE_EWC_WARMUP_EPOCHS)
        else:
            lam = ONLINE_EWC_LAMBDA
        
        loss_ewc = ewc.penalty()
        total_loss = loss_ce + lam * loss_ewc
        
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d} | CE: {loss_ce.item():.4f} | EWC: {loss_ewc.item():.4f} | Total: {total_loss.item():.4f}")

    # 任务完成：用固定memory估计Fisher并consolidate
    mem_obs, mem_tgt, mem_tid = task_memory[i]
    fisher_i = ewc.estimate_fisher_from_memory(mem_obs, mem_tgt, mem_tid)
    ewc.consolidate(fisher_i)
    print(f"  → 任务 {current_task} 的Fisher已合并到累计Fisher")

    # 阶段性评估所有任务（使用固定memory评估）
    print(f"\n--- 阶段 {i+1} 评估结果 (固定评估集) ---")
    for j in range(NUM_TASKS):
        f1 = evaluate_fixed_memory(model, j, task_memory)
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