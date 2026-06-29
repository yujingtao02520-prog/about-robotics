# 技术报告：机器人 VLA 数据清洗与训练评测 Pipeline

## 1. 摘要

本报告说明 `robot_vla_data_pipeline_demo` 的技术原理、执行步骤，以及它与机器人学习和 Vision-Language-Action (VLA) 前沿研究的关系。

这个项目不是一个大模型训练工程，而是一个面向 VLA 训练前置环节的数据工程 demo。它解决的问题是：把来自 LIBERO / RoboMimic 风格的 HDF5 episode 数据，统一成 observation-action-language schema，然后完成质量检查、可视化、格式转换、baseline 训练和评测闭环。

在真实 VLA 系统中，模型能力往往高度依赖数据质量。即使使用 OpenVLA、Octo、RT-X、ACT 或 Diffusion Policy 这类更强的策略模型，如果原始 episode 存在语言缺失、action 突变、图像黑屏、维度不一致或 NaN，训练和评估都会被污染。本项目展示的正是这类基础但关键的数据基础设施。

## 2. 项目定位

本 demo 对应的工程定位是：

```text
Raw HDF5 robot demonstrations
-> unified episode schema
-> data quality assessment
-> human-readable visualization
-> downstream format conversion
-> lightweight training verification
-> evaluation report
```

它面向三类场景：

1. 数据接入：将不同机器人数据源映射到统一 episode 表示。
2. 数据清洗：发现不能直接进入训练的数据异常。
3. 训练验证：用小模型快速确认数据能被 PyTorch 正常读取、batch、forward、backward 和评估。

与真实生产级 VLA pipeline 相比，本项目做了简化：图像只有单相机 RGB，action 是固定连续向量，格式转换是 schema-compatible demo，不复刻官方框架的完整目录约定、视频编码、tokenizer、metadata 或大规模分布式训练逻辑。

## 3. 核心数据原理

机器人模仿学习数据通常按 episode 组织。一个 episode 表示一次完整任务轨迹：

```text
instruction: "pick up the red block"

t=0: observation_0 -> action_0
t=1: observation_1 -> action_1
...
t=T-1: observation_T-1 -> action_T-1
```

本项目使用的统一 schema 是：

```python
episode = {
    "episode_id": str,
    "task_name": str,
    "language_instruction": str,
    "num_steps": int,
    "observations": {
        "rgb": np.ndarray,        # [T, H, W, 3]
        "state": np.ndarray,      # [T, state_dim]
    },
    "actions": np.ndarray,        # [T, action_dim]
    "metadata": {
        "source_format": str,
        "robot": str,
        "camera_names": list,
        "fps": int,
    }
}
```

其中：

- observation 是机器人在当前时间步看到和感知到的信息，例如相机图像、关节角、末端位姿、夹爪状态。
- action 是机器人需要执行的控制量，例如末端位姿增量、关节目标或夹爪开合。
- language_instruction 是任务条件，让同一视觉场景可以对应不同目标。
- metadata 保留来源格式和采集信息，方便后续追踪和分组评估。

统一 schema 的价值是把“原始数据格式差异”和“下游训练逻辑”解耦。只要 parser 输出稳定，QA、可视化、转换器和 baseline 都不需要知道原始 HDF5 是 mock、LIBERO-like 还是 RoboMimic-like。

## 4. Pipeline 步骤

### Step 1：生成或接入原始数据

入口：

```bash
python scripts/generate_mock_data.py
```

该脚本生成 `data/mock/mock_robot_data.hdf5`，模拟 RoboMimic-like HDF5 结构：

```text
data/demo_000/obs/rgb
data/demo_000/obs/state
data/demo_000/actions
data/demo_000.attrs["language_instruction"]
```

为了验证 QA，mock 数据会主动注入异常：

- 缺失语言指令；
- action 突变；
- RGB 全零帧；
- episode 过短；
- action 维度错误；
- action 包含 NaN。

### Step 2：解析到统一 schema

入口：

```bash
python scripts/parse_dataset.py --input data/mock/mock_robot_data.hdf5
```

解析过程包括：

1. 识别 HDF5 格式；
2. 遍历 episode group；
3. 在候选路径中读取 RGB、state、actions、language；
4. 将 RGB 规范为 `[T, H, W, 3]`；
5. 将 state/action 规范为 `[T, D]`；
6. 生成统一 episode dict；
7. 保存为 `data/processed/episodes.pkl` 和 `episodes.jsonl`。

这一步对应真实工程中的 adapter 层。接入真实 LIBERO 或 RoboMimic 数据时，主要扩展 `src/hdf5_loader.py` 和 `src/episode_parser.py`。

### Step 3：执行 QA 检查

入口：

```bash
python scripts/run_qa.py --input data/processed/episodes.pkl
```

QA 规则如下：

| 规则 | 原理 | 训练风险 |
|---|---|---|
| language missing | language_instruction 为空 | VLA 失去任务条件，无法学习语言-视觉-动作对齐 |
| action spike | 相邻 action 的 L2 距离超过阈值 | MSE loss 被异常大动作主导，策略可能学到不平滑控制 |
| empty RGB | 图像缺失或全零 | 视觉 encoder 接收无信息输入，污染视觉条件分布 |
| short episode | 轨迹长度小于最小阈值 | 不能表达完整任务，影响时序模型和成功率统计 |
| NaN/Inf | state/action 有非有限值 | 训练 loss 可能变成 NaN，batch 计算失败 |
| action dim mismatch | action 维度不一致 | batch 堆叠失败，模型输出维度无法对齐 |

输出：

```text
outputs/qa_reports/qa_report.csv
outputs/qa_reports/qa_summary.json
```

### Step 4：逐帧可视化

入口：

```bash
python scripts/visualize_episode.py --episode_id episode_000
```

输出：

```text
outputs/visualizations/episode_000.gif
```

可视化不是装饰功能，而是机器人数据工程中的核心调试工具。很多问题只看统计很难发现，例如图像是否黑屏、语言是否和动作一致、轨迹中是否有突变帧。GIF 将 RGB、instruction、state 和 action 放在一起，便于人工快速检查。

### Step 5：转换为下游格式

入口：

```bash
python scripts/convert_formats.py --input data/processed/episodes.pkl --target all
```

输出：

```text
outputs/converted/lerobot/
outputs/converted/act/
outputs/converted/openvla/
```

本项目实现三种简化格式：

- LeRobot-like JSON：适合表达 frame-level observation/action/timestamp。
- ACT-like NPZ：适合表达 `images`、`qpos`、`actions` 这种 imitation learning 数组结构。
- OpenVLA-like JSONL：适合表达 image/instruction/action 的语言条件样本。

这些格式是教学展示版，不是官方完整格式。真实接入时还需要处理视频编码、episode 索引、parquet/metadata、action tokenizer、机器人 embodiment 信息等。

### Step 6：baseline 训练

入口：

```bash
python scripts/train_baseline_mlp.py --input data/processed/episodes.pkl
python scripts/train_baseline_cnn.py --input data/processed/episodes.pkl
```

Baseline 1：MLP Policy

```text
state [state_dim]
-> Linear(128) -> ReLU
-> Linear(128) -> ReLU
-> action [action_dim]
```

Baseline 2：CNN + MLP Policy

```text
RGB [3, H, W] -> CNN encoder
state [state_dim] -> MLP encoder
concat(image_feature, state_feature) -> MLP head -> action
```

训练数据集会过滤不可训练样本，例如 NaN 帧和 action 维度错误 episode。注意：过滤训练样本不等于忽略数据问题，异常仍然保留在 QA 报告中。

### Step 7：评估

入口：

```bash
python scripts/evaluate_baselines.py
```

输出：

```text
outputs/baseline_eval.csv
```

指标包括：

- MSE；
- MAE；
- per-dimension MSE；
- action smoothness；
- failed episode ratio from QA。

Baseline 评估的目的不是证明模型强，而是验证数据闭环强：数据能被读入、能被打 batch、能训练、能保存 checkpoint、能恢复并评估。

## 5. 与前沿研究的关系

### 5.1 VLA 的主线：从视觉语言预训练到机器人动作

RT-2 提出了将视觉语言模型直接适配到机器人控制的思路：模型输入图像和语言，输出机器人动作 token，从而把互联网规模视觉语言知识迁移到机器人控制中。OpenVLA 进一步开源了 7B 参数 VLA，并在大规模机器人 demonstrations 上训练，强调开放模型、微调和部署效率。

本 demo 不实现大规模 VLA 模型，但它服务于同一个核心假设：VLA 需要高质量的 image/state/action/language 对齐数据。没有稳定的数据 schema 和 QA，后面的 VLA 训练很难可靠。

### 5.2 大规模机器人数据：Open X-Embodiment、DROID 与数据标准化

Open X-Embodiment / RT-X 汇聚了多机构、多机器人、多任务数据，研究跨机器人迁移和 generalist policy。DROID 则强调真实世界、大规模、in-the-wild 机器人操作数据。它们共同说明一个趋势：机器人学习正在从单任务小数据，走向多来源、多 embodiment、大规模数据。

这种趋势让数据工程变成核心瓶颈。不同数据源的相机、状态、action 定义、语言标注、采样频率都不同。本 demo 的统一 schema、metadata、parser 和 converter，正是大规模机器人数据融合前必须具备的基础能力。

### 5.3 通用机器人策略：Octo、OpenVLA 与可微调策略

Octo 强调开源 generalist robot policy，可以通过语言或目标图像条件化，并适配新的 observation/action spaces。OpenVLA 强调开源 VLA、参数高效微调和量化部署。这些工作都依赖一个前提：下游用户能够把自己的机器人数据整理成模型期望的数据格式。

本 demo 的贡献点不是模型结构，而是把数据准备成“可微调策略能消费”的形式，包括：

- episode 级组织；
- frame 级 observation/action 对齐；
- language instruction 保留；
- action/state 维度检查；
- 可视化人工审计；
- 转换到 LeRobot / ACT / OpenVLA-like 下游格式。

### 5.4 行为克隆与动作序列建模：ACT、Diffusion Policy

ACT 使用 transformer 预测 action chunks，缓解逐步预测中的误差累积问题。Diffusion Policy 用条件扩散过程建模 action 分布，适合多模态动作和高维动作空间。

本 demo 的 MLP/CNN baseline 比 ACT 或 Diffusion Policy 简单得多，但它们验证的是同一个数据前提：

- action 是否连续且维度一致；
- image/state/action 是否时间对齐；
- 模型是否能从 observation 预测 action；
- action smoothness 是否处于合理范围。

如果后续要升级，本项目可以把 baseline 模块替换为 ACT-style action chunking、Diffusion Policy 或 transformer policy，而无需重写 parser、QA 和 converter。

### 5.5 Benchmarks：LIBERO 与 RoboMimic

LIBERO 面向 lifelong robot learning，强调知识迁移、任务套件和高质量演示数据。RoboMimic 提供了离线人类示范数据和可复现实验框架，强调 demonstration quality 对离线学习的重要影响。

本项目以 LIBERO / RoboMimic 风格 HDF5 为输入背景，原因是它们代表了机器人模仿学习中常见的 episode 数据组织方式。QA pipeline 中的 action spike、短轨迹、NaN、空图像等检查，与这些 benchmark 中“数据质量决定训练稳定性”的经验是一致的。

### 5.6 LeRobot 与端到端机器人学习工具链

LeRobot 代表了机器人学习工具链开源化、标准化的趋势。它关注数据采集、存储、训练、推理和部署的一体化流程。

本 demo 的 LeRobot-like 导出只是最小教学格式，但工程思想与 LeRobot 方向一致：机器人学习不只需要模型，还需要可复用的数据表示、加载、可视化、训练和评估工具。

### 5.7 新近方向：长时程推理、数据管理与高质量执行

截至 2026-06-29，近期研究还在关注三个方向：

- 长时程任务：例如利用文本与视觉中间轨迹，让模型在长 horizon 任务中保持全局计划和局部动作一致。
- 数据管理：例如面向大规模机器人轨迹的视频压缩、元数据索引和快速 streaming。
- 执行质量：不仅看任务是否成功，还看动作是否平滑、是否优雅、是否满足隐式约束。

本 demo 中的 QA 和 action smoothness 虽然简单，但已经触及这些方向的工程基础。真实系统中可以进一步加入轨迹分段、关键帧标注、多相机时间同步、成功标签、接触/力觉信号和执行质量评分。

## 6. 可扩展路线

### 6.1 接入真实 LIBERO / RoboMimic

扩展点：

- `src/hdf5_loader.py`：增加更多候选 dataset path。
- `src/episode_parser.py`：处理多相机、多 state 字段、多 action convention。
- `config/default.yaml`：增加真实数据路径和 schema 映射配置。

### 6.2 增强 QA

可新增规则：

- camera timestamp drift；
- state/action 长度不一致；
- gripper action 饱和；
- action jerk 过大；
- episode success label 缺失；
- language 与 task_name 不一致；
- 多相机帧不同步；
- 图像模糊或曝光异常。

### 6.3 升级训练模型

可替换为：

- ACT action chunking；
- Diffusion Policy；
- transformer sequence policy；
- OpenVLA fine-tuning adapter；
- Octo-style generalist policy adapter。

### 6.4 转换为真实框架格式

当前转换器是简化版。要接入真实训练框架，可以进一步实现：

- LeRobot dataset writer；
- ACT 原始训练目录结构；
- OpenVLA / Open X-Embodiment 风格数据 writer；
- RLDS 或 parquet metadata；
- 视频编码与懒加载；
- action tokenizer 和 normalization stats。

## 7. 复现命令

完整运行：

```bash
pip install -r requirements.txt

python scripts/generate_mock_data.py
python scripts/parse_dataset.py --input data/mock/mock_robot_data.hdf5
python scripts/run_qa.py --input data/processed/episodes.pkl
python scripts/visualize_episode.py --episode_id episode_000
python scripts/convert_formats.py --input data/processed/episodes.pkl --target all
python scripts/train_baseline_mlp.py --input data/processed/episodes.pkl
python scripts/train_baseline_cnn.py --input data/processed/episodes.pkl
python scripts/evaluate_baselines.py
```

预期生成：

```text
data/mock/mock_robot_data.hdf5
data/processed/episodes.pkl
outputs/qa_reports/qa_report.csv
outputs/qa_reports/qa_summary.json
outputs/visualizations/episode_000.gif
outputs/converted/lerobot/
outputs/converted/act/
outputs/converted/openvla/
outputs/checkpoints/mlp_policy.pt
outputs/checkpoints/cnn_policy.pt
outputs/baseline_eval.csv
```

## 8. 结论

本项目展示的是 VLA 训练前的数据基础设施：统一 schema、质量检查、可视化、跨格式转换、baseline 验证和评估报告。它与前沿 VLA 研究的关系可以概括为：

- RT-2 / OpenVLA / Octo 代表模型能力上限；
- Open X-Embodiment / DROID 代表数据规模化趋势；
- ACT / Diffusion Policy 代表动作序列建模方法；
- LeRobot 代表工具链标准化趋势；
- 本 demo 代表这些前沿工作落地前必须完成的数据工程底座。

一个可靠的 VLA 系统不只是“大模型 + 数据”，还需要清晰的数据接口、可复现的清洗流程、可解释的 QA 报告和能快速验证的 baseline。本项目就是这条工程链路的最小可运行版本。

## 9. 参考资料

- RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control, 2023. https://arxiv.org/abs/2307.15818
- Open X-Embodiment: Robotic Learning Datasets and RT-X Models, 2023. https://arxiv.org/abs/2310.08864
- RoboMimic / What Matters in Learning from Offline Human Demonstrations for Robot Manipulation, 2021. https://arxiv.org/abs/2108.03298
- LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning, 2023. https://arxiv.org/abs/2306.03310
- Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware / ACT, 2023. https://arxiv.org/abs/2304.13705
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, 2023. https://arxiv.org/abs/2303.04137
- DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset, 2024. https://arxiv.org/abs/2403.12945
- Octo: An Open-Source Generalist Robot Policy, 2024. https://arxiv.org/abs/2405.12213
- OpenVLA: An Open-Source Vision-Language-Action Model, 2024. https://arxiv.org/abs/2406.09246
- LeRobot: An Open-Source Library for End-to-End Robot Learning, 2026. https://arxiv.org/abs/2602.22818
- Robo-DM: Data Management For Large Robot Datasets, 2025. https://arxiv.org/abs/2505.15558
