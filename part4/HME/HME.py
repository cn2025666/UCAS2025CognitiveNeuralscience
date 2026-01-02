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
HIDDEN_SIZE_PRIMARY = 512
HIDDEN_SIZE_SECONDARY = 512
NUM_TASKS = len(tasks_to_learn)
MAX_SEQ_LEN = 100
BATCH_SIZE = 64
EPOCHS = 2000

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
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        return self.fc2(out)

class SecondaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_tasks=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size + num_tasks, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # 存储任务特征统计
        self.task_features_mean = None
        self.original_samples = None  # 存储一些原任务样本
    def forward(self, x, task_id_onehot, return_features=False):
        task_info = task_id_onehot.unsqueeze(1).repeat(1, x.size(1), 1)
        combined_input = torch.cat((x, task_info), dim=2)
        lstm_out, (hidden_state, cell_state) = self.lstm(combined_input)
        features = hidden_state[-1]  # 最后一个层的隐藏状态
        if return_features:
            return self.fc(lstm_out), features
        else:
            return self.fc(lstm_out)
    def update_task_features(self, data):
        """更新任务特征统计"""
        obs, target, _ = data
        batch_size = obs.size(0)
        task_onehot = torch.zeros(batch_size, 1)
        with torch.no_grad():
            _, features = self.forward(obs, task_onehot, return_features=True)
            self.task_features_mean = features.mean(dim=0)
            # 存储少量原任务样本
            if self.original_samples is None:
                self.original_samples = (obs[:5].clone(), target[:5].clone())

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
    def _extract_features(self, obs):
        """从观测序列中提取特征，用于一级分类器输入"""
        if len(obs.shape) == 3:
            return obs.mean(dim=1)
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
    def _train_secondary_classifier(self, expert, data, epochs=1000):
        """训练次级分类器"""
        obs, target, _ = data
        batch_size = obs.size(0)
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
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    def _retrain_primary_classifier(self, epochs=1000):
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
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.primary_classifier(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"一级分类器训练 Epoch {epoch}, Loss: {loss.item():.4f}")
    def _update_memory(self, data, task_id, num_samples=1000):
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
        num_to_store = min(num_samples, batch_size * 16)
        indices = np.random.choice(batch_size * 16, num_to_store, replace=False)
        for idx in indices:
            obs_sample = obs[idx // 16].unsqueeze(0)  # 保持batch维度
            self.memory.append((obs_sample, task_id))
        # 如果超过最大容量，随机丢弃一些旧的例题
        if len(self.memory) > self.max_memory:
            num_to_remove = len(self.memory) - self.max_memory
            remove_indices = np.random.choice(len(self.memory), num_to_remove, replace=False)
            # 按索引从大到小删除，避免索引错位
            for idx in sorted(remove_indices, reverse=True):
                del self.memory[idx]
            print(f"记忆库已满，随机丢弃 {num_to_remove} 个例题")
    def learn_first_task(self, task_idx, epochs=1000):
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
        self._train_secondary_classifier(first_expert, train_data, epochs=epochs)
        # 2. 添加到专家列表
        self.secondary_classifiers.append(first_expert)
        self.task_mapping[0] = task_idx  # 专家0对应原始任务task_idx
        self.num_tasks = 1
        # 3. 存储例题
        self._update_memory(train_data, task_id=0, num_samples=2000)
        # 4. 训练一级分类器（当前只有一类）
        self._retrain_primary_classifier(epochs=epochs)
        return 0  # 返回分配的任务ID
    def _evaluate_expert_similarity(self, expert, data):
        """使用特征空间的相似性判断任务相似性"""
        obs, target, _ = data
        batch_size = obs.size(0)
        expert_task_onehot = torch.zeros(batch_size, 1)
        expert.eval()
        with torch.no_grad():
            # 1. 获取模型的中间特征
            # 修改SecondaryClassifier以暴露特征
            task_info = expert_task_onehot.unsqueeze(1).repeat(1, obs.size(1), 1)
            combined_input = torch.cat((obs, task_info), dim=2)
            # 获取LSTM的隐藏状态特征
            lstm_out, (hidden_state, cell_state) = expert.lstm(combined_input)
            # 使用最后一个时间步的隐藏状态作为特征
            features = hidden_state[-1]  # (batch_size, hidden_size)
            # 2. 计算特征分布统计量
            mean_features = features.mean(dim=0)  # (hidden_size,)
            # 3. 如果有历史特征，计算相似性
            if hasattr(expert, 'task_features_mean'):
                # 计算余弦相似度
                similarity = F.cosine_similarity(
                    mean_features.unsqueeze(0), 
                    expert.task_features_mean.unsqueeze(0)
                ).item()
                # 4. 考虑F1分数但加权降低
                outputs = expert.fc(lstm_out)
                preds = outputs.argmax(dim=-1)
                preds_np = preds.cpu().numpy().flatten()
                target_np = target.cpu().numpy().flatten()
                f1 = f1_score(target_np, preds_np, average='macro')
                # 综合分数：相似性权重70%，F1权重30%
                combined_score = 0.7 * similarity + 0.3 * f1
                return combined_score
            else:
                # 第一次评估，只计算F1
                outputs = expert.fc(lstm_out)
                preds = outputs.argmax(dim=-1)
                preds_np = preds.cpu().numpy().flatten()
                target_np = target.cpu().numpy().flatten()
                return f1_score(target_np, preds_np, average='macro')
    def learn_new_task(self, task_idx, stage, epochs=1000):
        """学习新任务"""
        print(f"\n>>> 学习新任务 [{stage}]: {tasks_to_learn[task_idx]}")
        train_data = get_batch(task_idx, batch_size=BATCH_SIZE, seed_offset=stage*1000)
        # 1. 评估现有专家（使用改进的相似性度量）
        similarity_scores = []
        for i, expert in enumerate(self.secondary_classifiers):
            # 使用多指标融合
            similarity = self._evaluate_expert_similarity(expert, train_data, i)
            similarity_scores.append(similarity)
            print(f"    专家 {i} 相似性: {similarity:.4f} (任务: {tasks_to_learn[self.task_mapping[i]]})")
        # 2. 判断是否为新任务（动态阈值）
        max_similarity = max(similarity_scores) if similarity_scores else 0
        threshold = self._get_dynamic_threshold(stage, max_similarity)
        if not similarity_scores or max_similarity < threshold:
            # 创建新专家
            print(f"创建新专家 #{len(self.secondary_classifiers)}")
            new_expert = SecondaryClassifier(
                input_size=MAX_INPUT,
                hidden_size=HIDDEN_SIZE_SECONDARY,
                output_size=MAX_OUTPUT,
                num_tasks=1
            )
            self._train_secondary_classifier(new_expert, train_data, epochs=epochs)
            new_expert.update_task_features(train_data)  # 更新特征统计
            new_expert_id = len(self.secondary_classifiers)
            self.secondary_classifiers.append(new_expert)
            self.task_mapping[new_expert_id] = task_idx
            self.num_tasks += 1
            assigned_task_id = new_expert_id
        else:
            # 强化现有最佳专家，但使用保护机制
            best_idx = np.argmax(similarity_scores)
            best_similarity = similarity_scores[best_idx]
            print(f"强化现有专家 #{best_idx} (相似性: {best_similarity:.4f})")
            # 检查是否需要保护原任务知识
            if best_similarity < 0.85:  # 相似性不够高，需要谨慎
                print("警告：任务相似性中等，使用保护性训练")
                self._train_with_protection(
                    self.secondary_classifiers[best_idx], 
                    train_data, 
                    epochs=epochs
                )
            else:
                # 高度相似，可以安全训练
                self._train_secondary_classifier(
                    self.secondary_classifiers[best_idx], 
                    train_data, 
                    epochs=epochs
                )
            # 更新专家特征统计
            self.secondary_classifiers[best_idx].update_task_features(train_data)
            assigned_task_id = best_idx
        # 3. 存储例题
        self._update_memory(train_data, task_id=assigned_task_id, num_samples=2000)
        # 4. 重新训练一级分类器
        self._retrain_primary_classifier(epochs=1000)
        return assigned_task_id
    def _train_with_protection(self, expert, new_data, epochs=500):
        """保护性训练：防止完全覆盖原任务知识"""
        # 获取原任务样本
        if expert.original_samples is None:
            # 如果没有存储，简单训练
            self._train_secondary_classifier(expert, new_data, epochs//2)
            return
        obs_original, target_original = expert.original_samples
        obs_new, target_new, _ = new_data
        # 混合训练：新旧任务数据一起训练
        for epoch in range(epochs):
            # 交替训练新旧任务
            if epoch % 2 == 0:
                # 训练新任务
                task_onehot = torch.zeros(obs_new.size(0), 1)
                outputs = expert(obs_new, task_onehot)
                loss_new = F.cross_entropy(
                    outputs.view(-1, MAX_OUTPUT), 
                    target_new.view(-1)
                )
            else:
                # 训练原任务
                task_onehot = torch.zeros(obs_original.size(0), 1)
                outputs = expert(obs_original, task_onehot)
                loss_original = F.cross_entropy(
                    outputs.view(-1, MAX_OUTPUT), 
                    target_original.view(-1)
                )
            # 组合损失
            if epoch % 2 == 0:
                loss = loss_new
            else:
                loss = loss_original
            # 反向传播
            optimizer = torch.optim.Adam(expert.parameters(), lr=0.001)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"保护性训练 Epoch {epoch}, Loss: {loss.item():.4f}")
    def _get_dynamic_threshold(self, stage, max_similarity):
        """动态调整阈值：随着任务增多，阈值降低"""
        base_threshold = self.new_task_threshold
        # 阶段数越多，阈值越低（因为可能越来越难区分）
        decay_factor = 0.95 ** stage
        dynamic_threshold = base_threshold * decay_factor
        # 如果最大相似性很高，可以稍微提高阈值
        if max_similarity > 0.9:
            dynamic_threshold = min(dynamic_threshold * 1.1, 0.9)
        return dynamic_threshold
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
            probs = F.softmax(primary_output, dim=1)
            avg_probs = probs.mean(dim=0)
            predicted_task = avg_probs.argmax().item()
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
    learner = HierarchicalContinualLearner(
        new_task_threshold=0.6,  # F1低于0.6认为是新任务
        max_memory=10000          # 最大存储10000个例题
    )
    history = []
    for i in range(len(tasks_to_learn)):
        if i == 0:
            learner.learn_first_task(task_idx=i, epochs=EPOCHS)
        else:
            learner.learn_new_task(task_idx=i, stage=i+1, epochs=EPOCHS)
        stage_i_scores = []
        for j in range(len(tasks_to_learn)):
            if j <= i:
                f1 = learner.evaluate_task(j, eval_seed_offset=9999)
            else:
                f1 = 0.0
            stage_i_scores.append(f1)
        history.append(stage_i_scores)
        print(f"\n阶段{i+1}评估结果:")
        for j, task in enumerate(tasks_to_learn):
            print(f"  {task}: F1 = {stage_i_scores[j]:.4f}")
    print(f"\n创建的专家数量: {len(learner.secondary_classifiers)}")
    print(f"一级分类器类别数: {learner.num_tasks}")
    print(f"例题库大小: {len(learner.memory)}")
    print(f"专家映射: {learner.task_mapping}")
    return history

if __name__ == "__main__":
    history = main()
