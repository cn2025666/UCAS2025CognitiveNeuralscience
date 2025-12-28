import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import neurogym as ngym
import numpy as np
import random
from sklearn.metrics import f1_score
import warnings
import copy
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

tasks_to_learn = ['GoNogo-v0', 'DelayComparison-v0', 'DelayMatchSample-v0']
MAX_INPUT, MAX_OUTPUT = 33, 33
HIDDEN_SIZE_PRIMARY = 64
HIDDEN_SIZE_SECONDARY = 128
NUM_TASKS = len(tasks_to_learn)
MAX_SEQ_LEN = 50
BATCH_SIZE = 16

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

class PrimaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # 直接使用MLP，因为特征已经取平均值
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x: (batch, features)
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        return self.fc2(out)

class SecondaryClassifier(nn.Module):
    """次级分类器：完成具体任务的LSTM模型"""
    def __init__(self, input_size, hidden_size, output_size, num_tasks=1):
        super().__init__()
        # 每个次级分类器只处理一个任务，所以num_tasks=1
        self.lstm = nn.LSTM(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, task_id_onehot):
        # x: (batch, seq_len, input_size)
        # task_id_onehot: (batch, 1) 因为每个次级分类器只对应一个任务
        task_info = task_id_onehot.unsqueeze(1).repeat(1, x.size(1), 1)
        combined_input = torch.cat((x, task_info), dim=2)
        out, _ = self.lstm(combined_input)
        return self.fc(out)

class HierarchicalContinualLearner:
    """分层混合专家模型 - 核心实现"""
    def __init__(self, new_task_threshold=0.6, max_memory=2000):
        """
        初始化分层学习器
        Args:
            new_task_threshold: 判断是否为新任务的F1阈值
            max_memory: 例题库最大容量
        """
        self.new_task_threshold = new_task_threshold
        self.max_memory = max_memory
        # 一级分类器（初始时为空，将在第一个任务后创建）
        self.primary_classifier = None
        # 次级分类器列表，每个元素对应一个任务专家
        self.secondary_classifiers = []
        # 例题库：存储(observation, task_id)用于训练一级分类器
        self.memory = []
        # 当前任务数量
        self.num_tasks = 0
        # 记录每个次级分类器对应的原始任务
        self.task_mapping = {}  # 专家ID -> 原始任务ID
        print(f">>> 初始化分层连续学习器")
        print(f"    新任务阈值: {new_task_threshold}")
        print(f"    最大记忆容量: {max_memory}")
    def _extract_features(self, obs):
        """从观测序列中提取特征，用于一级分类器输入"""
        # 简单策略：取序列的平均值
        if len(obs.shape) == 3:  # (batch, seq_len, features)
            return obs.mean(dim=1)  # (batch, features)
        else:
            return obs
    def _evaluate_expert(self, expert, data, expert_task_id):
        """
        评估专家在数据上的表现
        Args:
            expert: 次级分类器
            data: (obs, target, task_onehot) 元组
            expert_task_id: 专家对应的任务ID
        """
        obs, target, _ = data
        # 为这个专家创建合适的任务ID（全为0，因为每个专家只对应一个任务）
        batch_size = obs.size(0)
        expert_task_onehot = torch.zeros(batch_size, 1)
        expert.eval()
        with torch.no_grad():
            outputs = expert(obs, expert_task_onehot)
            preds = outputs.argmax(dim=-1)
            # 计算F1分数
            preds_np = preds.cpu().numpy().flatten()
            target_np = target.cpu().numpy().flatten()
            f1 = f1_score(target_np, preds_np, average='macro')
        return f1
    def _train_secondary_classifier(self, expert, data, epochs=10):
        """训练次级分类器"""
        obs, target, _ = data
        batch_size = obs.size(0)
        # 为这个专家创建任务ID（全为0）
        task_onehot = torch.zeros(batch_size, 1)
        expert.train()
        optimizer = torch.optim.Adam(expert.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = expert(obs, task_onehot)
            loss = criterion(outputs.view(-1, MAX_OUTPUT), target.view(-1))
            loss.backward()
            optimizer.step()
    def _retrain_primary_classifier(self):
        """重新训练一级分类器"""
        if not self.memory:
            return
        print(f"重新训练一级分类器，使用 {len(self.memory)} 个例题")
        # 准备训练数据
        X_list = []
        y_list = []
        for obs, task_id in self.memory:
            # 提取特征
            features = self._extract_features(obs)
            X_list.append(features)
            y_list.append(task_id)
        # 堆叠成张量
        X = torch.cat(X_list, dim=0)  # (total_samples, features)
        y = torch.tensor(y_list, dtype=torch.long)
        # 创建新的一级分类器
        self.primary_classifier = PrimaryClassifier(
            input_size=MAX_INPUT,
            hidden_size=HIDDEN_SIZE_PRIMARY,
            num_classes=self.num_tasks
        )
        # 训练一级分类器
        optimizer = torch.optim.Adam(self.primary_classifier.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()
        # 简单训练几轮（因为数据量不大）
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = self.primary_classifier(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f"    一级分类器训练 Epoch {epoch}, Loss: {loss.item():.4f}")
    def _update_memory(self, data, task_id, num_samples=10):
        """
        更新例题库
        Args:
            data: (obs, target, task_onehot) 元组
            task_id: 任务ID
            num_samples: 存储的样本数量
        """
        obs, _, _ = data
        batch_size = obs.size(0)
        # 随机选择一些样本存储
        num_to_store = min(num_samples, batch_size)
        indices = np.random.choice(batch_size, num_to_store, replace=False)
        for idx in indices:
            obs_sample = obs[idx].unsqueeze(0)  # 保持batch维度
            self.memory.append((obs_sample, task_id))
        # 如果超过最大容量，随机丢弃一些旧的例题
        if len(self.memory) > self.max_memory:
            num_to_remove = len(self.memory) - self.max_memory
            remove_indices = np.random.choice(len(self.memory), num_to_remove, replace=False)
            # 按索引从大到小删除，避免索引错位
            for idx in sorted(remove_indices, reverse=True):
                del self.memory[idx]
            print(f"    记忆库已满，随机丢弃 {num_to_remove} 个例题")
    def learn_first_task(self, task_idx):
        """学习第一个任务"""
        print(f"\n>>> 学习第一个任务: {tasks_to_learn[task_idx]}")
        # 获取训练数据
        train_data = get_batch(task_idx, batch_size=BATCH_SIZE, seed_offset=0)
        # 1. 创建并训练第一个次级分类器
        print("创建第一个次级分类器...")
        first_expert = SecondaryClassifier(
            input_size=MAX_INPUT,
            hidden_size=HIDDEN_SIZE_SECONDARY,
            output_size=MAX_OUTPUT,
            num_tasks=1
        )
        self._train_secondary_classifier(first_expert, train_data, epochs=100)
        # 2. 添加到专家列表
        self.secondary_classifiers.append(first_expert)
        self.task_mapping[0] = task_idx  # 专家0对应原始任务task_idx
        self.num_tasks = 1
        # 3. 存储例题
        self._update_memory(train_data, task_id=0, num_samples=20)
        # 4. 训练一级分类器（当前只有一类）
        self._retrain_primary_classifier()
        return 0  # 返回分配的任务ID
    def learn_new_task(self, task_idx, stage):
        """
        学习新任务
        Args:
            task_idx: 原始任务索引
            stage: 学习阶段（用于seed_offset）
        """
        print(f"\n>>> 学习新任务 [{stage}]: {tasks_to_learn[task_idx]}")
        # 获取训练数据
        train_data = get_batch(task_idx, batch_size=BATCH_SIZE, seed_offset=stage*1000)
        # 1. 评估现有专家
        f1_scores = []
        for i, expert in enumerate(self.secondary_classifiers):
            f1 = self._evaluate_expert(expert, train_data, expert_task_id=i)
            f1_scores.append(f1)
            print(f"    专家 {i} (原始任务: {tasks_to_learn[self.task_mapping[i]]}) F1: {f1:.4f}")
        # 2. 判断是否为新任务
        if not f1_scores or max(f1_scores) < self.new_task_threshold:
            # 创建新专家
            print(f"创建新专家 #{len(self.secondary_classifiers)}")
            new_expert = SecondaryClassifier(
                input_size=MAX_INPUT,
                hidden_size=HIDDEN_SIZE_SECONDARY,
                output_size=MAX_OUTPUT,
                num_tasks=1
            )
            self._train_secondary_classifier(new_expert, train_data, epochs=100)
            # 添加到专家列表
            new_expert_id = len(self.secondary_classifiers)
            self.secondary_classifiers.append(new_expert)
            self.task_mapping[new_expert_id] = task_idx
            self.num_tasks += 1
            assigned_task_id = new_expert_id
        else:
            # 强化现有最佳专家
            best_idx = np.argmax(f1_scores)
            print(f"强化现有专家 #{best_idx} (原始任务: {tasks_to_learn[self.task_mapping[best_idx]]})")
            self._train_secondary_classifier(self.secondary_classifiers[best_idx], train_data, epochs=100)
            assigned_task_id = best_idx
        # 3. 存储例题
        self._update_memory(train_data, task_id=assigned_task_id, num_samples=20)
        # 4. 重新训练一级分类器
        self._retrain_primary_classifier()
        return assigned_task_id
    def evaluate_task(self, task_idx, eval_seed_offset=9999):
        """评估指定任务"""
        if self.primary_classifier is None or not self.secondary_classifiers:
            return 0.0
        # 获取测试数据
        test_data = get_batch(task_idx, batch_size=32, seed_offset=eval_seed_offset)
        obs, target, _ = test_data
        # 1. 提取特征用于一级分类
        features = self._extract_features(obs)  # (batch, features)
        # 1. 一级分类：判断任务类型
        self.primary_classifier.eval()
        with torch.no_grad():
            primary_output = self.primary_classifier(features)
            predicted_task = primary_output.argmax(dim=1).mode().values.item()
        # 2. 二级分类：选择对应专家进行预测
        if predicted_task < len(self.secondary_classifiers):
            selected_expert = self.secondary_classifiers[predicted_task]
            # 为选中的专家准备任务ID（全为0）
            batch_size = obs.size(0)
            expert_task_onehot = torch.zeros(batch_size, 1)
            selected_expert.eval()
            with torch.no_grad():
                outputs = selected_expert(obs, expert_task_onehot)
                preds = outputs.argmax(dim=-1)
                # 计算F1分数
                preds_np = preds.cpu().numpy().flatten()
                target_np = target.cpu().numpy().flatten()
                f1 = f1_score(target_np, preds_np, average='macro')
        else:
            # 如果一级分类器预测了不存在的任务，使用第一个专家
            print(f"警告：一级分类器预测了不存在的任务 {predicted_task}，使用专家0")
            selected_expert = self.secondary_classifiers[0]
            batch_size = obs.size(0)
            expert_task_onehot = torch.zeros(batch_size, 1)
            selected_expert.eval()
            with torch.no_grad():
                outputs = selected_expert(obs, expert_task_onehot)
                preds = outputs.argmax(dim=-1)
                preds_np = preds.cpu().numpy().flatten()
                target_np = target.cpu().numpy().flatten()
                f1 = f1_score(target_np, preds_np, average='macro')
        return f1

def main():
    # 创建分层学习器
    learner = HierarchicalContinualLearner(
        new_task_threshold=0.6,  # F1低于0.6认为是新任务
        max_memory=2000          # 最大存储2000个例题
    )
    # 记录每个阶段的性能
    history = []
    # 阶段1: 学习第一个任务 (GoNogo)
    learner.learn_first_task(task_idx=0)  # GoNogo
    # 评估所有任务
    stage1_scores = []
    for i in range(len(tasks_to_learn)):
        if i == 0:
            f1 = learner.evaluate_task(i, eval_seed_offset=9999)
        else:
            f1 = 0.0  # 未学习的任务
        stage1_scores.append(f1)
    history.append(stage1_scores)
    print(f"\n阶段1评估结果:")
    for i, task in enumerate(tasks_to_learn):
        print(f"  {task}: F1 = {stage1_scores[i]:.4f}")
    # 阶段2: 学习第二个任务 (DelayComparison)
    learner.learn_new_task(task_idx=1, stage=2)  # DelayComparison
    # 评估所有任务
    stage2_scores = []
    for i in range(len(tasks_to_learn)):
        f1 = learner.evaluate_task(i, eval_seed_offset=9999)
        stage2_scores.append(f1)
    history.append(stage2_scores)
    print(f"\n阶段2评估结果:")
    for i, task in enumerate(tasks_to_learn):
        print(f"  {task}: F1 = {stage2_scores[i]:.4f}")
    # 阶段3: 学习第三个任务 (DMS)
    learner.learn_new_task(task_idx=2, stage=3)  # DMS
    # 评估所有任务
    stage3_scores = []
    for i in range(len(tasks_to_learn)):
        f1 = learner.evaluate_task(i, eval_seed_offset=9999)
        stage3_scores.append(f1)
    history.append(stage3_scores)
    print(f"\n阶段3评估结果:")
    for i, task in enumerate(tasks_to_learn):
        print(f"  {task}: F1 = {stage3_scores[i]:.4f}")
    # 打印最终统计信息
    print("\n" + "=" * 80)
    print("实验总结")
    print("=" * 80)
    print(f"创建的专家数量: {len(learner.secondary_classifiers)}")
    print(f"一级分类器类别数: {learner.num_tasks}")
    print(f"例题库大小: {len(learner.memory)}")
    print(f"专家映射: {learner.task_mapping}")
    print("\n性能历史:")
    print(f"{'阶段':<10} {'GoNogo':<12} {'DelayComp':<12} {'DMS':<12}")
    for i, scores in enumerate(history):
        print(f"阶段 {i+1}:  {scores[0]:<10.4f}  {scores[1]:<10.4f}  {scores[2]:<10.4f}")
    # 分析遗忘情况
    print("\n遗忘分析:")
    if len(history) >= 3:
        # Task 1 (GoNogo) 的遗忘
        forgetting_task1 = history[0][0] - history[2][0]
        print(f"任务1 (GoNogo) 遗忘: {history[0][0]:.4f} → {history[2][0]:.4f} (Δ = {forgetting_task1:+.4f})")
        # Task 2 (DelayComp) 的遗忘
        if len(history) > 1:
            forgetting_task2 = history[1][1] - history[2][1]
            print(f"任务2 (DelayComp) 遗忘: {history[1][1]:.4f} → {history[2][1]:.4f} (Δ = {forgetting_task2:+.4f})")
    return history

if __name__ == "__main__":
    history = main()