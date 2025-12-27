import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import neurogym as ngym
import numpy as np
import random
from collections import deque
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

# A-GEM特定配置
AGEM_BUFFER_SIZE = 20  # 每个任务存储的样本数
AGEM_REF_BATCH_SIZE = 8  # 计算参考梯度的批次大小

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

# ==================== A-GEM Memory Buffer ====================
class AGEMBuffer:
    """A-GEM的episodic memory buffer"""
    def __init__(self, max_size=AGEM_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, obs, target, task_id):
        self.buffer.append((obs, target, task_id))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, target, task_id = zip(*samples)
        return (
            torch.stack(obs),
            torch.stack(target),
            torch.stack(task_id),
        )

    def is_empty(self):
        return len(self.buffer) == 0
    
    def __len__(self):
        return len(self.buffer)

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

# ==================== A-GEM梯度投影函数 ====================
def project_gradient(current_grad, ref_grad):
    """
    将当前梯度投影到参考梯度的半空间
    如果 g · g_ref < 0，则投影；否则保持原样
    """
    # 计算点积
    dot_product = torch.dot(current_grad, ref_grad)
    
    # 如果梯度方向与参考梯度冲突（点积为负），进行投影
    if dot_product < 0:
        # 投影公式: g_new = g - (g · g_ref / ||g_ref||^2) * g_ref
        ref_norm_sq = torch.dot(ref_grad, ref_grad)
        projected_grad = current_grad - (dot_product / ref_norm_sq) * ref_grad
        return projected_grad
    else:
        return current_grad

def flatten_gradients(model):
    """将模型所有梯度展平为一维向量"""
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)

def set_gradients(model, flat_grad):
    """将一维梯度向量设置回模型参数"""
    idx = 0
    for param in model.parameters():
        if param.grad is not None:
            param_size = param.grad.numel()
            param.grad.copy_(flat_grad[idx:idx + param_size].view(param.grad.shape))
            idx += param_size

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
criterion = nn.CrossEntropyLoss()

# 为每个任务创建一个buffer
buffers = [AGEMBuffer(max_size=AGEM_BUFFER_SIZE) for _ in range(NUM_TASKS)]

# 性能追踪矩阵
results_matrix = np.zeros((NUM_TASKS, NUM_TASKS))

# ==================== A-GEM连续学习 ====================
print("\n" + "="*70)
print("方法: A-GEM (Averaged Gradient Episodic Memory) 连续学习")
print(f"配置: Buffer大小={AGEM_BUFFER_SIZE}, 参考批次={AGEM_REF_BATCH_SIZE}")
print("="*70)

for i in range(NUM_TASKS):
    current_task = TASKS[i]
    print(f"\n[阶段 {i+1}/{NUM_TASKS}] 训练任务: {current_task}")
    
    # 训练当前任务
    for epoch in range(EPOCHS_PER_TASK + 1):
        model.train()
        
        # 获取当前任务数据
        obs, target, task_id = get_batch(i, seed_offset=epoch)
        
        # 存入Buffer (用于未来的参考梯度计算)
        for b in range(obs.size(0)):
            buffers[i].add(obs[b], target[b], task_id[b])
        
        # 前向传播和损失计算
        output = model(obs, task_id)
        loss = criterion(output.view(-1, MAX_OUTPUT), target.view(-1))
        
        # 计算当前任务的梯度
        optimizer.zero_grad()
        loss.backward()
        
        # A-GEM核心: 如果有旧任务，计算参考梯度并投影
        if i > 0:
            # 保存当前梯度
            current_grad = flatten_gradients(model).clone()
            
            # 计算所有旧任务的平均参考梯度
            ref_grads = []
            for old_task_idx in range(i):
                if not buffers[old_task_idx].is_empty():
                    # 从旧任务buffer采样
                    old_obs, old_target, old_task_id = buffers[old_task_idx].sample(AGEM_REF_BATCH_SIZE)
                    
                    # 计算旧任务的梯度
                    optimizer.zero_grad()
                    old_output = model(old_obs, old_task_id)
                    old_loss = criterion(
                        old_output.view(-1, MAX_OUTPUT),
                        old_target.view(-1)
                    )
                    old_loss.backward()
                    
                    # 收集旧任务的梯度
                    ref_grads.append(flatten_gradients(model).clone())
            
            # 计算平均参考梯度
            if len(ref_grads) > 0:
                avg_ref_grad = torch.stack(ref_grads).mean(dim=0)
                
                # 投影当前梯度
                projected_grad = project_gradient(current_grad, avg_ref_grad)
                
                # 设置投影后的梯度
                set_gradients(model, projected_grad)
        
        # 更新参数
        optimizer.step()
        
        if epoch % 200 == 0:
            buffer_info = f"[Buffer: {', '.join([str(len(buffers[j])) for j in range(i+1)])}]"
            print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f} {buffer_info}")

    print(f"  → 任务 {current_task} 的数据已加入Memory Buffer")

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