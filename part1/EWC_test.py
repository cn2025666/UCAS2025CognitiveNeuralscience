import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import neurogym as ngym
import numpy as np
import random
import copy
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">>> 确定性模式已开启，随机种子: {seed}")


set_seed(42)

# --- 任务配置 ---
tasks_to_learn = ['GoNogo-v0', 'DelayComparison-v0', 'DelayMatchSample-v0']
MAX_INPUT, MAX_OUTPUT = 10, 10
HIDDEN_SIZE = 128
NUM_TASKS = len(tasks_to_learn)
MAX_SEQ_LEN = 50

envs = [gym.make(t) for t in tasks_to_learn]
for env in envs:
    env.action_space.seed(42)


def get_batch(task_idx, batch_size=16, seed_offset=0):
    env = envs[task_idx]
    batch_obs = np.zeros((batch_size, MAX_SEQ_LEN, MAX_INPUT))
    batch_targets = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)
    for b in range(batch_size):
        obs, _ = env.reset(seed=task_idx * 1000 + seed_offset + b)
        for t in range(MAX_SEQ_LEN):
            batch_obs[b, t, :obs.shape[0]] = obs
            obs, _, _, _, info = env.step(env.action_space.sample())
            gt = info.get('gt', getattr(env.unwrapped, 'gt', [0] * MAX_SEQ_LEN)[t % 5])
            batch_targets[b, t] = gt
    task_onehot = np.zeros((batch_size, NUM_TASKS))
    task_onehot[:, task_idx] = 1
    return torch.FloatTensor(batch_obs), torch.LongTensor(batch_targets), torch.FloatTensor(task_onehot)


# --- 3. 增强版 EWC 核心算法 ---
class EWC(object):
    def __init__(self, model: nn.Module, dataset_func, num_samples=200):
        self.model = model
        self.dataset_func = dataset_func
        self.num_samples = num_samples
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = {}

    def _diag_fisher(self, task_idx):
        precision_matrices = {}
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()
        for _ in range(self.num_samples):
            input, target, task_id = self.dataset_func(task_idx, batch_size=1, seed_offset=random.randint(0, 5000))
            self.model.zero_grad()
            output = self.model(input, task_id)
            loss = F.cross_entropy(output.view(-1, MAX_OUTPUT), target.view(-1))
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / self.num_samples

        # 【核心改进：归一化】
        # 将 Fisher 值缩放到 [0, 1] 之间，防止数值淹没
        for n in precision_matrices:
            max_val = precision_matrices[n].max()
            if max_val > 0:
                precision_matrices[n].data /= max_val

        return precision_matrices

    def register_old_task(self, task_idx):
        self._means[task_idx] = {n: p.clone().detach() for n, p in self.params.items()}
        self._precision_matrices[task_idx] = self._diag_fisher(task_idx)

    def ewc_loss(self):
        if not self._means:
            return torch.tensor(0.0).to(next(self.model.parameters()).device)
        loss = 0
        for task_idx in self._means.keys():
            for n, p in self.model.named_parameters():
                _loss = self._precision_matrices[task_idx][n] * (p - self._means[task_idx][n]) ** 2
                loss += _loss.sum()
        return loss


# --- 4. 模型定义 ---
class CognitiveRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks):
        super().__init__()
        self.lstm = nn.LSTM(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, task_id_onehot):
        task_info = task_id_onehot.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.lstm(torch.cat((x, task_info), dim=2))
        return self.fc(out)


model = CognitiveRNN(MAX_INPUT, HIDDEN_SIZE, MAX_OUTPUT, NUM_TASKS)
# 【核心改进：降低学习率】
# 学习率过大会让模型暴力冲破 EWC 的阻力
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

ewc = EWC(model, get_batch)
# 由于做了归一化，Lambda 不需要 200万那么夸张，1000-5000 即可
EWC_LAMBDA = 1200


def evaluate(task_idx):
    model.eval()
    obs, target, task_id = get_batch(task_idx, batch_size=32, seed_offset=9999)
    with torch.no_grad():
        output = model(obs, task_id)
        pred = output.argmax(dim=-1).numpy().flatten()
        true = target.numpy().flatten()
    return f1_score(true, pred, average='macro')


print("\n>>> 开始顺序学习 (增强版 EWC)")

for i, task_name in enumerate(tasks_to_learn):
    print(f"\n[阶段 {i + 1}] 训练: {task_name} ...")
    for epoch in range(1001):
        model.train()
        obs, target, task_id = get_batch(i, seed_offset=epoch)
        output = model(obs, task_id)

        loss_ce = criterion(output.view(-1, MAX_OUTPUT), target.view(-1))
        loss_ewc = ewc.ewc_loss()
        total_loss = loss_ce + EWC_LAMBDA * loss_ewc

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            ewc_val = loss_ewc.item() if torch.is_tensor(loss_ewc) else loss_ewc
            print(f"  Epoch {epoch} | CE: {loss_ce.item():.4f} | EWC Penalty: {EWC_LAMBDA * ewc_val:.4f}")

    ewc.register_old_task(i)

    print(f"\n>> 阶段 {i + 1} 性能报告：")
    for j in range(i + 1):
        f1 = evaluate(j)
        print(f"{tasks_to_learn[j]:<20} | F1: {f1:.4f}")