# 认知神经科学1 and 2 研习代码库

本目录聚焦于“认知神经科学”相关的对比与连续学习实验，涵盖三类基线模型（LSTM / GRU / CTRNN）、EWC 强化的顺序学习演示，以及配套的可视化脚本。这套脚本统一依赖确定性数据采集流程，方便复现实验指标与遗忘趋势。

## 说明
- **任务来源**：采用 `neurogym` 实现的一系列认知任务（例如 `GoNogo-v0`, `DelayComparison-v0` 等），10 个任务用于对比训练，3 个任务用于 EWC 顺序学习演示。
- **数据采集**：所有训练脚本都定义了 `get_batch(task_idx, batch_size, seed_offset)`，通过固定种子、最大序列长度（50 步）和统一的输入/输出维度（33/33 或 10/10）采集对齐数据。
- **随机性控制**：在每个训练脚本里都调用 `set_seed(42)`，锁定 Python/NumPy/PyTorch、CuDNN 的随机状态，确保每次运行结果一致。
- **评估指标**：训练结束后打印每个任务的 F1 分数和 MSE 损失；某些脚本（如 `EWC_test.py`）还以阶段方式评估已有任务，凸显遗忘。

## 数据集说明
本项目不依赖外部静态数据集文件，而是通过 `neurogym` 库实时生成认知任务数据。为了保证实验的可复现性，所有数据生成过程均受到严格的随机种子控制。
如需静态数据集，我们已导出为dataset_10tasks.pt
### 数据生成机制
- **动态生成**：在训练过程中调用 `get_batch` 函数实时生成数据。
- **确定性**：通过 `set_seed(42)` 锁定全局随机种子，并为每个 batch 的生成指定固定的 `seed_offset`，确保每次运行生成的样本序列完全一致。
- **预处理**：
  - **序列长度**：统一截断或填充至 `MAX_SEQ_LEN = 50`。
  - **维度对齐**：输入和输出维度根据任务集合的最大需求进行统一（例如 10 任务集合统一为 33 维，3 任务集合统一为 10 维），不足部分补零。

### 任务集合
#### 1. 多任务基线对比 (CTRNN/GRU/LSTM)
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

#### 2. 顺序学习演示 (EWC)
选取了 3 个代表性任务用于演示灾难性遗忘与 EWC 的保护作用：
1. `GoNogo-v0` (任务 1)
2. `DelayComparison-v0` (任务 2)
3. `DelayMatchSample-v0` (任务 3)

## 快速上手
```bash
pip install torch gymnasium neurogym matplotlib scikit-learn numpy
python CTRNN.py      # 全任务 CTRNN 基线训练
python GRU.py        # 多任务 GRU 基线
python LSTM.py       # 多任务 LSTM 基线
python EWC_test.py   # 3 任务顺序学习，展示 EWC 机制
python visible.py    # 绘制各模型在 10 个任务上的 F1 对比柱状图
python compare.py    # 导出 F1/损失趋势图
python forget_visible.py  # 分类遗忘趋势可视化
python EWC_visible.py     # EWC 顺序学习性能曲线
``` 

## 环境依赖
- **Python**：推荐使用 3.10或3.11
- **虚拟环境**：优先使用 `python -m venv venv` 并 `venv\Scripts\activate`（Windows）后再安装依赖，避免与系统包冲突。
- **核心库**：`torch`（1.15+）、`gymnasium`、`neurogym`、`matplotlib`、`scikit-learn`、`numpy`。
- **NeuroGym 依赖**：`neurogym` 会自动拉取 `gymnasium>=0.31`、`numpy` 等，若出现编译报错，可先 `pip install gymnasium==0.31.0` 再安装 `neurogym`。
- **绘图支持**：`matplotlib` 负责渲染所有可视化；结果保存为 `compare.png`, `loss.png`，可用于 LaTeX  `forget_visible.py` 等脚本也依赖。
- **额外工具**：如需导出结果表格，可额外安装 `pandas`、`seaborn`，但运行本仓库的脚本不是必须。

## 结构概览
- `CTRNN.py`：定义 `CTRNN` 模型（时间常数 `tau`、欧拉法）并并行训练 10 个任务，评估阶段呈现每个任务的 F1/MSE。`get_batch` 融合任务 one-hot 附加输入，并在 `forward` 中逐步更新膜电位。
- `GRU.py` / `LSTM.py`：分别替换核心 RNN 模块。训练/评估流程、数据采集、超参（隐藏维度 128、学习率 0.002）与 `CTRNN` 保持一致，便于横向对比。
- `EWC_test.py`：封装了归一化的 `EWC` 类（Fisher 信息矩阵元素先归一化再累加），在每个阶段训练完后调用 `register_old_task` 记录旧任务参数与 Fisher，训练损失由交叉熵＋$	ext{EWC}_	ext{penalty}$ 组成。每训练完一个任务就输出阶段性能评价。
- `EWC_visible.py` / `forget_visible.py` / `forget2.py`：基于实测 F1 段值绘制继发学习趋势图，可用于补充报告或直接嵌入 LaTeX。
- `visible.py`、`compare.py`：分别绘制多个模型在 10 个任务上 1) F1 比较柱状图；2) 训练损失收敛曲线；3) 折线图。输出文件（`compare.png`, `loss.png`）可直接用于论文或演示。

## 重要机制细节
1. **`set_seed`**：封装了随机库、PyTorch、CuDNN 的全部固定手段，并在第一行打印锁定提示，提醒用户结果可重复。
2. **`get_batch`**：以 `MAX_INPUT`, `MAX_OUTPUT`, `MAX_SEQ_LEN` 等常量规范每个批次；默认 batch=16、seq_len=50，内部用 `env.step(env.action_space.sample())` 采样多步观测并自动补全标签。
3. **任务 one-hot**：每个 `get_batch` 还附加了一份任务标识符，作为 RNN 输入的一部分，帮助模型区分当前任务。
4. **评估**：在 `.eval()` 模式下，用 `f1_score(..., average='macro')` 评价整局；同时以交叉熵和 MSE 观察概率分布差异。
5. **EWC 归一化**：Fisher 信息矩阵在 `register_old_task` 时先归一化到 [0, 1]，避免惩罚项因尺度而失控；`EWC_LAMBDA = 1200` 设定较高值补偿归一化后的缩小。

## 建议的使用顺序
1. 先运行 `CTRNN.py/GRU.py/LSTM.py` 获取基线集，检查 `compare.py` 和 `visible.py` 生成的图像。
2. 用 `EWC_test.py` 观察 EWC 在顺序任务上的抵抗遗忘效果，配合 `EWC_visible.py` 可视化阶段性能。
3. `forget_visible.py` 与 `forget2.py` 记录未经保护的灾难性遗忘，便于与 EWC 曲线对比。

## 可能的扩展方向
- 将 `tasks` 列表扩展至新的 `neurogym` 任务或自定义任务，保持 `MAX_INPUT/OUTPUT` 对齐即可复用训练管线。
- 引入 `tensorboard` 记录指标，或把 `F1 + MSE` 结果保存成 CSV 便于后续分析。
- 把 `EWC` 类包装成可复用回调，进而支持其他模型（如 GRU）在顺序训练中加 EWC 惩罚。

