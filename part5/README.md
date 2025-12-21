# 课题任务 (5) - 改进认知任务仿真与分析

本文件夹包含了针对课题任务 (5) 的完整实现代码、实验数据及可视化结果。我们通过引入感知噪声和时间变异性，构建了更接近现实世界的“真实环境”，并在此基础上评估了模型的鲁棒性和连续学习能力。

## 1. 改进思路与生物学启发

为了弥补标准 NeuroGym 任务过于理想化的缺陷，我们对实验环境进行了以下核心改进：

### 1.1 引入感知噪声 (Sensory Noise)
*   **改进**: 在模型接收的观测数据中添加高斯噪声 (`sigma=0.2`)。
*   **生物学意义**: 模拟生物神经系统中的感觉噪声（Sensory Noise）和突触传递的随机性。大脑通过群体编码（Population Coding）和吸引子动力学（Attractor Dynamics）在噪声中维持稳定表征。这测试了模型是否学习到了鲁棒的神经表征。

### 1.2 引入时间变异性 (Temporal Variability)
*   **改进**: 将任务中的关键时间阶段（如 `fixation`, `delay`, `decision`）从固定时长改为在一定范围内随机采样（Uniform Distribution）。
*   **生物学意义**: 现实世界事件具有时间不确定性。前额叶皮层（PFC）和海马体需在不确定时间间隔内维持工作记忆。可变延迟迫使模型学习通用的动力学机制（如积分器），而非简单地“计数”时间步。

## 2. 文件结构说明

*   **`realistic_experiment.py`**:
    *   **功能**: 在改进后的环境中训练并评估 LSTM, GRU, CTRNN 三种模型。
    *   **输出**: 生成 `results_realistic.json` 保存实验数据。
*   **`visualize_results.py`**:
    *   **功能**: 读取 Part 1 (基线) 和 Part 5 (真实) 的数据，生成对比图表。
    *   **输出**: `realistic_comparison.png` (模型性能对比), `impact_analysis.png` (环境影响深度分析)。
*   **`sequential_experiment.py`**:
    *   **功能**: 在真实环境中执行顺序学习实验 (Task 1 -> Task 2 -> Task 3)，模拟灾难性遗忘过程。
    *   **输出**: 生成 `sequential_results.json`。
*   **`visualize_forgetting.py`**:
    *   **功能**: 对比 Part 1 和 Part 5 的遗忘曲线。
    *   **输出**: `forgetting_comparison.png`。

## 3. 运行指南

请确保已激活 `cn` 环境：
```bash
conda activate cn
```

### 步骤 1: 运行多任务并行训练实验
```bash
python part5/realistic_experiment.py
# 运行结束后，生成 results_realistic.json
```

### 步骤 2: 生成性能对比可视化
```bash
python part5/visualize_results.py
# 生成 realistic_comparison.png 和 impact_analysis.png
```

### 步骤 3: 运行顺序学习与遗忘实验
```bash
python part5/sequential_experiment.py
# 运行结束后，生成 sequential_results.json
```

### 步骤 4: 生成遗忘对比可视化
```bash
python part5/visualize_forgetting.py
# 生成 forgetting_comparison.png
```

## 4. 实验结果与分析

### 4.1 环境复杂度对性能的影响
*   **观察**: 参见 `impact_analysis.png`。在引入噪声和可变时间后，所有模型的性能普遍下降，尤其是对噪声敏感的任务（如 `AntiReach`）。
*   **结论**: 真实环境显著增加了任务难度，要求模型具备更强的抗噪能力和时间灵活性。LSTM 和 GRU 凭借门控机制表现出比 CTRNN 更好的鲁棒性。

### 4.2 对灾难性遗忘的影响
*   **观察**: 参见 `forgetting_comparison.png`。
    *   **Baseline (Part 1)**: Task 1 (GoNogo) 在学习新任务后 F1 分数从 1.00 骤降至 ~0.30 (Drop: -0.69)。
    *   **Realistic (Part 5)**: Task 1 的 F1 分数从 0.97 降至 ~0.53 (Drop: -0.44)。
*   **结论**: 有趣的是，**真实环境下的灾难性遗忘程度反而较轻**。这可能是因为噪声和时间变异性起到了类似“正则化”的作用，迫使模型学习到更通用、更鲁棒的特征表示，这些特征在任务切换时更不容易被覆盖或干扰。这一发现符合“噪声训练提高泛化能力”的机器学习理论，也与生物大脑在嘈杂环境中进化出强适应性的现象相呼应。
