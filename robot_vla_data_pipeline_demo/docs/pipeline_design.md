# Pipeline 设计说明

本项目的核心目标是展示一个机器人 VLA 数据处理链路如何从 raw episode 数据走到可训练、可评估、可转换的统一数据资产。

```text
Raw HDF5 Data
-> Episode Parser
-> Unified Observation-Action-Language Schema
-> QA Pipeline
-> Visualization
-> Format Conversion
-> Baseline Training
-> Evaluation Report
```

## 1. Raw HDF5 Data

机器人模仿学习数据通常以 HDF5 存储，因为它适合保存多 episode、多模态、大数组数据。例如一个 episode 可能同时包含 RGB、深度图、机器人 proprioception、action、reward、done、language instruction 等。

真实 LIBERO / RoboMimic 数据的字段和 group layout 不完全一样。本 demo 的 mock 数据采用 RoboMimic-like 结构：

```text
data/demo_000/obs/rgb
data/demo_000/obs/state
data/demo_000/actions
data/demo_000.attrs["language_instruction"]
```

`src/hdf5_loader.py` 对常见路径做了候选匹配，因此后续接真实数据时，可以在 loader/parser 层扩展字段映射，而不影响 QA、可视化、训练和转换模块。

## 2. Episode Parser

Episode parser 的职责是把不同来源的数据转换为统一结构：

- RGB 统一为 `[T, H, W, 3]`
- state 统一为 `[T, state_dim]`
- action 统一为 `[T, action_dim]`
- language instruction 统一为字符串
- metadata 记录来源格式、机器人、相机名、fps 等

这一步是整个 pipeline 的接口边界。只要 parser 输出的 schema 稳定，后续模块就不需要关心原始数据来自 LIBERO、RoboMimic 还是 mock。

## 3. 为什么机器人数据要按 episode 组织

Episode 表示一次完整任务轨迹，例如“拿起红色方块”从初始观察到动作序列，再到任务完成。按 episode 组织有几个好处：

- 保留时序上下文：动作是否平滑、是否突变，需要比较相邻帧。
- 保留任务语义：一条语言指令通常对应一个完整 episode，而不是孤立帧。
- 支持轨迹级 QA：短轨迹、缺失语言、异常 episode 需要在 episode 级统计。
- 方便训练切分：可以按 episode 或 frame 展开给不同模型使用。

## 4. Observation / Action / Language 是什么

- observation：机器人在某个时间步看到和感知到的信息。在本 demo 中包含 RGB 图像和 state 向量。真实项目中还可能包含深度图、多相机、关节角、末端位姿、夹爪状态等。
- action：机器人策略输出或数据集中记录的控制量。在本 demo 中是 7 维连续向量，可理解为末端位姿增量和夹爪控制的简化表示。
- language：任务自然语言指令，例如 `pick up the red block`。VLA 模型需要将语言、视觉和动作关联起来，因此语言字段是 VLA 数据中非常关键的模态。

## 5. 为什么需要数据清洗

机器人数据来自真实采集、仿真导出或多来源合并时，很容易出现质量问题：

- 某些 episode 没有语言指令；
- 控制器或记录器造成 action 突变；
- 相机帧为空、全黑或缺失；
- episode 太短，不足以表达一个完整任务；
- state/action 中出现 NaN 或 Inf；
- 不同来源 action 维度不一致。

这些问题如果直接进入训练，会导致模型学到错误关联，甚至让训练过程出现 NaN loss 或 batch shape 错误。

## 6. Action spike 为什么影响训练

Action spike 指相邻时间步 action 差异突然变大。它可能来自控制器重置、数据记录错误、轨迹拼接错误或仿真异常。

对行为克隆或 VLA action prediction 来说，action spike 会产生几个问题：

- MSE loss 对大误差敏感，异常 spike 会主导梯度；
- 模型可能学到不平滑动作，影响机器人执行安全性；
- spike 通常不是任务语义本身，而是数据噪声；
- 在小数据集上，一个异常 episode 就可能明显影响 baseline 指标。

因此 QA 中需要统计 spike，并在真实训练前决定过滤、截断或降权。

## 7. Language missing 为什么影响 VLA

VLA 模型的关键是将 visual observation、language instruction 和 action 对齐。如果 language missing：

- 模型无法知道当前轨迹对应的任务意图；
- 多任务数据会退化成“看图猜动作”，缺少语言条件；
- 同一视觉状态下可能有不同任务目标，缺语言会造成标签歧义；
- OpenVLA 等语言条件模型的数据格式通常要求 instruction 字段存在。

因此语言缺失是 VLA 数据质量评估中的基础规则。

## 8. QA Pipeline

本 demo 的 QA pipeline 包含：

- language missing
- action spike
- empty RGB
- short episode
- NaN/Inf
- action dim mismatch

输出包括 episode 级 CSV 和全局 JSON summary。CSV 方便人工筛查，JSON 方便后续训练脚本读取，例如 `evaluate_baselines.py` 会把 failed episode ratio 合并进最终评估表。

## 9. Visualization

逐帧可视化是机器人数据工程中很实用的调试工具。只看数组统计很难发现图像是否黑屏、任务是否对齐、action 是否突然跳变。可视化模块把 RGB、语言、state 和 action 放在同一帧里，便于人工快速检查。

## 10. Format Conversion 的意义

不同训练框架对数据格式有不同要求：

- LeRobot 通常强调 episode/frame 级 observation、action、timestamp 和 metadata。
- ACT 常使用图像、qpos、action 等数组训练序列策略。
- OpenVLA 类模型需要 image/instruction/action 这样的语言条件样本。

真实官方格式通常包含更多 metadata、视频编码、索引文件、tokenization 或特定目录约定。本 demo 的转换器只实现 schema-compatible 简化格式，用于展示“统一 schema 可以适配多个下游训练栈”的工程思路。

## 11. Baseline Training

Baseline 实验的目的不是追求高精度，而是验证数据能否被模型正常读取和训练：

- MLP baseline 验证 state/action 数组是否能组成监督学习样本。
- CNN + MLP baseline 验证 RGB/state/action 三模态是否能被 PyTorch Dataset 和模型同时消费。
- checkpoint 验证模型权重、维度 metadata 和评估脚本是否闭环。

训练集会过滤不可训练的帧，例如 action 维度错误或 NaN/Inf。QA 报告仍保留这些异常，以便展示数据质量问题与训练过滤策略是分开的。

## 12. Evaluation Report

最终评估输出 `outputs/baseline_eval.csv`，包含：

- MSE
- MAE
- action smoothness
- per-dimension MSE
- failed episode ratio from QA

这些指标组合能回答两个问题：

1. 数据是否能被模型正常读取、训练、预测；
2. 数据质量问题在当前 mock 数据集中占多大比例。

在真实项目中，可以进一步扩展为按任务、按相机、按采集批次、按机器人类型统计指标。
