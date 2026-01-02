# 认知神经科学 part4

本目录聚焦于探索新的连续学习方法，希望能从其他角度降低灾难性遗忘的风险，基于之前综合性能最好的模型（LSTM），在此基础上进行改进，配套了可视化脚本。这套脚本统一依赖确定性数据采集流程，方便复现实验指标与遗忘趋势。

## 说明
- **任务来源**：采用 `neurogym` 实现的一系列认知任务（例如 `GoNogo-v0`, `DelayComparison-v0` 等），筛选出 10 个任务用于对比训练。
- **数据采集**：所有训练脚本都定义了 `get_batch(task_idx, batch_size, seed_offset)`，通过固定种子、最大序列长度（100 步）和统一的输入/输出维度（33/33）采集对齐数据。
- **随机性控制**：在每个训练脚本里都调用 `set_seed(42)`，锁定 Python/NumPy/PyTorch、CuDNN 的随机状态，确保每次运行结果一致。
- **评估指标**：训练结束后打印每个任务的 F1 分数和 MSE 损失并可视化。
- **学习方法优化思路**：类似学生对旧的相似知识的“复习”。把模型权重参数整体视为一个参数空间，假设每一轮训练数据学习对参数空间中参数点的作用是线性的，参数点被柔性限制在几个方向构成的子空间内（类似罚函数机制），在补空间内不移动或移动很少，可以认为子空间特征向量方向对旧任务是关键的，新的学习鼓励改变补空间方向的参数点移动，惩罚子空间方向的参数点移动，为了限制子空间方向移动，在新的训练数据中加入与新数据（在倒数第二个特征层的权重参数的余弦）相似度较大的旧数据一起训练，起到用旧数据限制旧任务主导参数方向的作用。这个连续学习方法已有类似先例，称为相似性加权交错学习（SWIL）

## 数据集说明
本项目不依赖外部静态数据集文件，而是通过 `neurogym` 库实时生成认知任务数据。为了保证实验的可复现性，所有数据生成过程均受到严格的随机种子控制。
### 数据生成机制
- **动态生成**：在训练过程中调用 `get_batch` 函数实时生成数据。
- **确定性**：通过 `set_seed(42)` 锁定全局随机种子，并为每个 batch 的生成指定固定的 `seed_offset`，确保每次运行生成的样本序列完全一致。
- **预处理**：
  - **序列长度**：统一截断或填充至 `MAX_SEQ_LEN = 100`。
  - **维度对齐**：输入和输出维度根据任务集合的最大需求进行统一（10 任务集合统一为 33 维），不足部分补零。

### 任务集合
#### 1. 多任务基线对比 (经典LSTM)
包含 10 个经典的认知神经科学任务，涵盖决策、记忆、抑制控制等维度：
- `AntiReach-v0`
- `ContextDecisionMaking-v0`
- `DelayComparison-v0`
- `DelayMatchCategory-v0`
- `DelayMatchSample-v0`
- `DelayMatchSampleDistractor1D-v0`
- `DelayPairedAssociation-v0`
- `DualDelayMatchSample-v0`
- `EconomicDecisionMaking-v0`
- `GoNogo-v0`

#### 2. 顺序学习演示 (SWIL)
使用连续学习方法SWIL，使用相同数据集（随机种子相同），进行并行实验：
- `AntiReach-v0`
- `ContextDecisionMaking-v0`
- `DelayComparison-v0`
- `DelayMatchCategory-v0`
- `DelayMatchSample-v0`
- `DelayMatchSampleDistractor1D-v0`
- `DelayPairedAssociation-v0`
- `DualDelayMatchSample-v0`
- `EconomicDecisionMaking-v0`
- `GoNogo-v0`

## 快速上手
```bash
pip install torch gymnasium neurogym matplotlib scikit-learn numpy
python SWIL.py      # 全任务 LSTM 基线训练和 SWIL 连续学习方法训练
python visualize.py # 可视化训练结果
``` 

## 环境依赖
- **Python**：推荐使用 3.11
- **虚拟环境**：优先使用 `python -m venv venv` 并 `venv\Scripts\activate`（Windows）后再安装依赖，避免与系统包冲突。
- **核心库**：`torch`（1.15+）、`gymnasium==0.29.1`、`neurogym==2.2.0`、`matplotlib`、`scikit-learn==1.4.2`、`numpy==1.26.4`。

## 重要机制细节
1. **`set_seed`**：封装了随机库、PyTorch、CuDNN 的全部固定手段，并在第一行打印锁定提示，提醒用户结果可重复。
2. **`get_batch`**：以 `MAX_INPUT`, `MAX_OUTPUT`, `MAX_SEQ_LEN` 等常量规范每个批次；默认 batch=32、seq_len=100，内部用 `env.step(env.action_space.sample())` 采样多步观测并自动补全标签。
3. **任务 one-hot**：每个 `get_batch` 还附加了一份任务标识符，作为 RNN 输入的一部分，帮助模型区分当前任务。
4. **评估**：在 `.eval()` 模式下，用 `f1_score(..., average='macro')` 评价整局；同时以交叉熵和 MSE 观察概率分布差异。

## 可能的扩展方向
- 将 `tasks` 列表扩展至新的 `neurogym` 任务或自定义任务，保持 `MAX_INPUT/OUTPUT` 对齐即可复用训练管线。
- 引入 `tensorboard` 记录指标，或把 `F1 + MSE` 结果保存成 CSV 便于后续分析。
