import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import neurogym as ngym
import numpy as np
import random
import copy
from sklearn.metrics import f1_score, pairwise_distances
from collections import defaultdict
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

tasks_to_learn = ['AntiReach-v0', 'ContextDecisionMaking-v0', 'DelayComparison-v0',
                 'DelayMatchCategory-v0', 'DelayMatchSample-v0', 'DelayMatchSampleDistractor1D-v0',
                 'DelayPairedAssociation-v0', 'DualDelayMatchSample-v0', 'EconomicDecisionMaking-v0', 'GoNogo-v0']
MAX_INPUT, MAX_OUTPUT = 33, 33
HIDDEN_SIZE = 512
NUM_TASKS = len(tasks_to_learn)
MAX_SEQ_LEN = 100
BATCH_SIZE = 32

envs = [gym.make(t) for t in tasks_to_learn]

def get_batch(task_idx, batch_size=BATCH_SIZE, seed_offset=0):
    """获取一个批次的数据，保持确定性"""
    env = envs[task_idx]
    batch_obs = np.zeros((batch_size, MAX_SEQ_LEN, MAX_INPUT))
    batch_targets = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)
    for b in range(batch_size):
        obs, _ = env.reset(seed=task_idx + b + seed_offset)
        for t in range(MAX_SEQ_LEN):
            batch_obs[b, t, :obs.shape[0]] = obs
            obs, _, _, _, info = env.step(env.action_space.sample())
            gt = info.get('gt', getattr(env.unwrapped, 'gt', [0] * MAX_SEQ_LEN)[t % 5])
            batch_targets[b, t] = gt
    task_onehot = np.zeros((batch_size, NUM_TASKS))
    task_onehot[:, task_idx] = 1
    return torch.FloatTensor(batch_obs), torch.LongTensor(batch_targets), torch.FloatTensor(task_onehot)

class CognitiveRNN(nn.Module):
    """LSTM认知神经网络"""
    def __init__(self, input_size, hidden_size, output_size, num_tasks):
        super().__init__()
        self.lstm = nn.LSTM(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # 存储每个任务的隐藏层激活，用于之后的相似性计算
        self.task_activations = {}
    def forward(self, x, task_id_onehot, return_activations=False):
        """前向传播"""
        batch_size, seq_len, _ = x.size()
        task_info = task_id_onehot.unsqueeze(1).repeat(1, seq_len, 1)
        combined_input = torch.cat((x, task_info), dim=2)
        out, (hidden, cell) = self.lstm(combined_input)
        # 获取最终隐藏状态
        final_hidden = hidden[-1]
        if return_activations:
            return self.fc(out), final_hidden
        return self.fc(out)

class SWIL:
    """相似性加权交错学习 (Similarity-Weighted Interleaved Learning)"""
    def __init__(self, model, num_memory_samples=100, temperature=1.0, similarity_threshold=0.2):
        self.model = model
        self.num_memory_samples = num_memory_samples
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold # 存储阈值
        self.memory_bank = {} # 存储旧任务的数据样本（记忆库）
        self.task_representations = {} # 存储每个任务的隐藏层激活表示
    def compute_similarity(self, current_task_idx, old_task_idx, method='centered_cosine'):
        """
        计算任务间相似度\n
        method: 
            'centered_cosine' - 中心化后的余弦相似度（推荐）
            'correlation' - 皮尔逊相关系数
            'euclidean' - 欧几里得距离转换的相似度
            'others' - 不规范参数统一为默认余弦相似度
        """
        if old_task_idx not in self.task_representations:
            return 0.0
        curr_rep = self.task_representations[current_task_idx]
        old_rep = self.task_representations[old_task_idx]
        curr_rep_np = curr_rep.detach().cpu().numpy() if isinstance(curr_rep, torch.Tensor) else curr_rep
        old_rep_np = old_rep.detach().cpu().numpy() if isinstance(old_rep, torch.Tensor) else old_rep
        min_dim = min(len(curr_rep_np), len(old_rep_np))
        curr_rep_np = curr_rep_np[:min_dim]
        old_rep_np = old_rep_np[:min_dim]
        n_components = min(50, min_dim)
        try:
            if len(self.task_representations) >= 5:
                all_reps = []
                for idx in self.task_representations.keys():
                    rep = self.task_representations[idx]
                    rep_np = rep.detach().cpu().numpy() if isinstance(rep, torch.Tensor) else rep
                    all_reps.append(rep_np[:min_dim])
                all_reps = np.array(all_reps)
                pca = PCA(n_components=min(n_components, len(all_reps)-1), whiten=True)
                pca.fit(all_reps)
                curr_transformed = pca.transform(curr_rep_np.reshape(1, -1))[0]
                old_transformed = pca.transform(old_rep_np.reshape(1, -1))[0]
                curr_rep_np, old_rep_np = curr_transformed, old_transformed
            else:
                if min_dim > 50:
                    random_proj = np.random.randn(50, min_dim) / np.sqrt(min_dim)
                    curr_rep_np = random_proj @ curr_rep_np
                    old_rep_np = random_proj @ old_rep_np
        except Exception as e:
            curr_rep_np = (curr_rep_np - np.mean(curr_rep_np)) / (np.std(curr_rep_np) + 1e-8)
            old_rep_np = (old_rep_np - np.mean(old_rep_np)) / (np.std(old_rep_np) + 1e-8)
        if method == 'centered_cosine':
            curr_centered = curr_rep_np - np.mean(curr_rep_np)
            old_centered = old_rep_np - np.mean(old_rep_np)
            dot_product = np.dot(curr_centered, old_centered)
            norm_curr = np.linalg.norm(curr_centered)
            norm_old = np.linalg.norm(old_centered)
            similarity = dot_product / (norm_curr * norm_old) if norm_curr * norm_old > 0 else 0.0
        elif method == 'correlation':
            similarity = np.corrcoef(curr_rep_np, old_rep_np)[0, 1] if len(curr_rep_np) > 1 else 0.0
            similarity = 0.0 if np.isnan(similarity) else similarity
        elif method == 'euclidean':
            distance = np.linalg.norm(curr_rep_np - old_rep_np)
            similarity = 1.0 / (1.0 + distance)
        else:
            dot_product = np.dot(curr_rep_np, old_rep_np)
            norm_curr = np.linalg.norm(curr_rep_np)
            norm_old = np.linalg.norm(old_rep_np)
            similarity = dot_product / (norm_curr * norm_old) if norm_curr * norm_old > 0 else 0.0
        return float(np.clip(similarity, -1.0, 1.0))
    def register_task(self, task_idx, data_loader_func=None):
        """注册已学习的任务到记忆库"""
        if data_loader_func:
            obs, target, task_id = data_loader_func(task_idx, batch_size=self.num_memory_samples, seed_offset=9999)
            self.memory_bank[task_idx] = (obs, target, task_id)
            self.model.eval()
            with torch.no_grad():
                all_hidden_states = []
                for i in range(min(10, len(obs))):
                    sample_output, hidden = self.model(obs[i:i+1], task_id[i:i+1], return_activations=True)
                    all_hidden_states.append(hidden)
                full_output, final_hidden = self.model(obs, task_id, return_activations=True)
                output_features = full_output.mean(dim=[0, 1])
                task_representation = torch.cat([final_hidden.mean(dim=0), output_features, torch.std(full_output, dim=[0, 1])])
                self.task_representations[task_idx] = task_representation
                rep_np = task_representation.numpy()
                print(f"  任务{task_idx}表示: 均值={np.mean(rep_np):.4f}, 标准差={np.std(rep_np):.4f}, 范围=[{np.min(rep_np):.4f}, {np.max(rep_np):.4f}]")
    def get_interleaved_batch(self, current_task_idx, current_batch, similarity_threshold=0.2):
        """根据相似度权重选择旧任务样本，生成交错学习批次。\n
        核心：只回放相似度高于threshold的旧任务"""
        current_obs, current_target, current_task_id = current_batch
        if current_task_idx not in self.task_representations:
            self.model.eval()
            with torch.no_grad():
                _, task_rep = self.model(current_obs, current_task_id, return_activations=True)
                self.task_representations[current_task_idx] = task_rep.mean(dim=0)
            print(f"  已计算任务 {current_task_idx} 的表示")
        if not self.memory_bank:
            return current_obs, current_target, current_task_id
        valid_old_indices = []
        valid_similarities = []
        for old_idx in self.memory_bank.keys():
            sim = self.compute_similarity(current_task_idx, old_idx)
            if sim >= similarity_threshold:
                valid_old_indices.append(old_idx)
                valid_similarities.append(sim)
        if not valid_old_indices:
            return current_obs, current_target, current_task_id
        valid_similarities = np.array(valid_similarities)
        valid_similarities = np.maximum(valid_similarities, 0)
        similarities_exp = np.exp(valid_similarities / self.temperature)
        probs = similarities_exp / (similarities_exp.sum() + 1e-8)
        avg_similarity = np.mean(valid_similarities)
        old_sample_ratio = min(0.7, avg_similarity)
        num_old_samples = int(BATCH_SIZE * old_sample_ratio)
        num_current_samples = BATCH_SIZE - num_old_samples
        all_obs = []
        all_targets = []
        all_task_ids = []
        if num_current_samples > 0:
            indices = np.random.choice(len(current_obs), num_current_samples, replace=True)
            all_obs.append(current_obs[indices])
            all_targets.append(current_target[indices])
            all_task_ids.append(current_task_id[indices])
        if num_old_samples > 0 and len(valid_old_indices) > 0:
            chosen_old_indices = np.random.choice(
                valid_old_indices, 
                size=num_old_samples, 
                p=probs, 
                replace=True
            )
            for old_idx in chosen_old_indices:
                old_obs, old_target, old_task_id = self.memory_bank[old_idx]
                sample_idx = np.random.randint(0, len(old_obs))
                all_obs.append(old_obs[sample_idx:sample_idx+1])
                all_targets.append(old_target[sample_idx:sample_idx+1])
                all_task_ids.append(old_task_id[sample_idx:sample_idx+1])
        if len(all_obs) > 0:
            interleaved_obs = torch.cat(all_obs, dim=0)
            interleaved_targets = torch.cat(all_targets, dim=0)
            interleaved_task_ids = torch.cat(all_task_ids, dim=0)
        else:
            interleaved_obs, interleaved_targets, interleaved_task_ids = current_obs, current_target, current_task_id
        if len(interleaved_obs) != BATCH_SIZE:
            if len(interleaved_obs) > BATCH_SIZE:
                interleaved_obs = interleaved_obs[:BATCH_SIZE]
                interleaved_targets = interleaved_targets[:BATCH_SIZE]
                interleaved_task_ids = interleaved_task_ids[:BATCH_SIZE]
            else:
                needed = BATCH_SIZE - len(interleaved_obs)
                indices = np.random.choice(len(current_obs), needed, replace=True)
                interleaved_obs = torch.cat([interleaved_obs, current_obs[indices]], dim=0)
                interleaved_targets = torch.cat([interleaved_targets, current_target[indices]], dim=0)
                interleaved_task_ids = torch.cat([interleaved_task_ids, current_task_id[indices]], dim=0)
        return interleaved_obs, interleaved_targets, interleaved_task_ids

def evaluate(model, task_idx):
    """评估模型在特定任务上的性能"""
    model.eval()
    obs, target, task_id = get_batch(task_idx, batch_size=32, seed_offset=9999)
    with torch.no_grad():
        output = model(obs, task_id)
        pred = output.argmax(dim=-1).numpy().flatten()
        true = target.numpy().flatten()
    return f1_score(true, pred, average='macro')

def run_experiment(use_swil=True):
    """运行顺序学习"""
    model = CognitiveRNN(MAX_INPUT, HIDDEN_SIZE, MAX_OUTPUT, NUM_TASKS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.CrossEntropyLoss()
    swil = None
    if use_swil:
        swil = SWIL(model, num_memory_samples=100, temperature=1.0, similarity_threshold=0.2)
    performance_history = [] # 存储每个阶段的性能
    # 顺序学习每个任务
    for i, task_name in enumerate(tasks_to_learn):
        print(f"\n训练任务: {task_name}")
        for epoch in range(3001):
            model.train()
            # 获取当前任务的训练数据
            current_batch = get_batch(i, seed_offset=epoch)
            if use_swil and i > 0:
                obs, target, task_id = swil.get_interleaved_batch(i, current_batch)
            else:
                obs, target, task_id = current_batch
            # 前向传播
            output = model(obs, task_id)
            loss = criterion(output.view(-1, MAX_OUTPUT), target.view(-1))
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 输出训练损失
            if epoch % 100 == 0:
                print(f"  Epoch {epoch} | Loss: {loss.item():.4f}")
        if use_swil:
            swil.register_task(i, get_batch)
        # 评估所有已学任务
        print(f"\n>> 阶段 {i+1} 性能报告:")
        print(f"{'任务':<25} | {'F1分数':<10}")
        stage_performance = []
        for j in range(i + 1):
            f1 = evaluate(model, j)
            stage_performance.append(f1)
            print(f"{tasks_to_learn[j]:<25} | {f1:.4f}")
        performance_history.append(stage_performance)
    return performance_history

def main():
    """主函数：比较普通学习和SWIL"""
    print(">>> 普通顺序学习（基线，演示灾难性遗忘）")
    baseline_history = run_experiment(use_swil=False)
    print(">>> SWIL顺序学习（缓解灾难性遗忘）")
    swil_history = run_experiment(use_swil=True)
    for i, task_name in enumerate(tasks_to_learn):
        print(f"任务: {task_name}")
        print(f"  - 普通学习最终F1: {baseline_history[-1][i]:.4f}")
        print(f"  - SWIL学习最终F1: {swil_history[-1][i]:.4f}")
        improvement = swil_history[-1][i] - baseline_history[-1][i]
        if improvement > 0:
            print(f"  - SWIL改进: +{improvement:.4f}")
        else:
            print(f"  - SWIL改进: {improvement:.4f}")
    baseline_retention = np.mean(baseline_history[-1][:-1]) if len(baseline_history[-1]) > 1 else 0
    swil_retention = np.mean(swil_history[-1][:-1]) if len(swil_history[-1]) > 1 else 0
    print(f"旧任务平均保留率:")
    print(f"  - 普通学习: {baseline_retention:.4f}")
    print(f"  - SWIL学习: {swil_retention:.4f}")
    print(f"  - 提升: {(swil_retention - baseline_retention):.4f}")

if __name__ == "__main__":
    main()