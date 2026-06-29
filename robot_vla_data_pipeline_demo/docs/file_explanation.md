# 文件说明

本文档逐个说明 `robot_vla_data_pipeline_demo` 中各文件的作用，方便阅读代码和扩展到真实机器人数据。

## 根目录文件

- `README.md`：项目说明与运行入口，包含环境安装、完整运行顺序、统一 schema、QA 规则、格式转换和 baseline 实验说明。
- `requirements.txt`：项目依赖库，包括 HDF5 读取、数值处理、可视化、PyTorch 训练和配置读取所需包。

## config

- `config/default.yaml`：默认配置文件。集中管理路径、mock 数据参数、QA 阈值、可视化参数、训练参数和评估参数。脚本默认从这里读取配置，命令行参数可覆盖关键路径或训练参数。

## data

- `data/raw/`：预留给真实 LIBERO / RoboMimic HDF5 数据。
- `data/mock/`：保存 mock HDF5 数据，默认输出 `mock_robot_data.hdf5`。
- `data/processed/`：保存解析后的统一 episode 数据，默认包含 `episodes.pkl` 和便于查看的 `episodes.jsonl`。

## outputs

- `outputs/qa_reports/`：保存 QA 输出，包括 `qa_report.csv` 和 `qa_summary.json`。
- `outputs/visualizations/`：保存 episode 可视化 GIF 或 PNG 序列。
- `outputs/converted/`：保存 LeRobot-like、ACT-like、OpenVLA-like 转换结果。
- `outputs/checkpoints/`：保存 MLP 与 CNN baseline checkpoint。
- `outputs/baseline_eval.csv`：保存 baseline 评估指标。

## scripts

- `scripts/generate_mock_data.py`：生成模拟 LIBERO / RoboMimic 风格的 HDF5 数据。默认生成 20 个 episode，包含 64x64 RGB、10 维 state、7 维 action，并注入语言缺失、动作突变、空 RGB、短轨迹、action 维度错误和 NaN 等异常。
- `scripts/parse_dataset.py`：读取原始 HDF5，并转换为统一 observation-action-language episode schema。解析后保存 `data/processed/episodes.pkl`，同时打印 episode 数量、平均长度、state/action 维度和语言覆盖率。
- `scripts/run_qa.py`：执行数据质量检查并生成报告。每个 episode 输出 pass/fail、失败原因和 issue 类型，最终写入 CSV 与 JSON。
- `scripts/visualize_episode.py`：将单个 episode 逐帧可视化。画面左侧显示 RGB，右侧显示 frame index、language_instruction、state 前几维和 action 向量。
- `scripts/convert_formats.py`：将统一 schema 转换为 LeRobot / ACT / OpenVLA 简化格式，用于展示跨训练栈数据适配能力。
- `scripts/train_baseline_mlp.py`：训练 state-to-action 的 MLP baseline。输入 state，输出 action，使用 MSELoss，保存 `mlp_policy.pt`。
- `scripts/train_baseline_cnn.py`：训练 RGB + state-to-action 的 CNN baseline。CNN 编码图像，MLP 编码 state，拼接后预测 action，保存 `cnn_policy.pt`。
- `scripts/evaluate_baselines.py`：评估 baseline 的预测误差和动作平滑性，输出 MSE、MAE、per-dimension MSE、action smoothness 和 QA failed episode ratio。

## src

- `src/__init__.py`：将 `src` 标记为 Python package，便于脚本导入模块。
- `src/data_schema.py`：定义统一 VLA episode 数据结构的创建、校验、保存、加载和 JSON 安全转换工具。核心 schema 包含 `episode_id`、`task_name`、`language_instruction`、`observations.rgb`、`observations.state`、`actions` 和 `metadata`。
- `src/hdf5_loader.py`：封装 HDF5 读取逻辑。负责识别 mock / RoboMimic-like / LIBERO-like 结构，列出 episode group，并从常见路径中读取 RGB、state、actions 和语言指令。
- `src/episode_parser.py`：负责不同原始格式到统一 schema 的映射。它会规范 RGB 到 `[T,H,W,3]`，规范 state/action 到 `[T,D]`，并生成统一 episode dict。
- `src/qa_rules.py`：实现语言缺失、动作突变、空图像、短轨迹、NaN/Inf、action 维度检查等质量规则，并负责保存 QA 报告。
- `src/visualizer.py`：实现帧级图像、状态、动作可视化。优先输出 GIF，如果 GIF writer 不可用，则回退保存 PNG 帧序列。
- `src/train_utils.py`：训练相关工具函数。包含 YAML 配置读取、路径解析、随机种子、train/val split、训练循环、评估循环、checkpoint 保存和加载。
- `src/metrics.py`：MSE、MAE、action smoothness、per-dimension MSE 和 QA failed ratio 读取等指标函数。

## src/converters

- `src/converters/__init__.py`：转换器 package 初始化文件。
- `src/converters/lerobot_converter.py`：导出 LeRobot-like JSON 数据格式。每个 episode 一个 JSON 文件，内部按 frame 保存 observation、action、语言和 timestamp。
- `src/converters/act_converter.py`：导出 ACT-like NPZ 数据格式。保存 `images`、`qpos`、`actions`、`language_instruction` 和 metadata。
- `src/converters/openvla_converter.py`：导出 OpenVLA-like JSONL 格式。每行表示一帧，包含 image placeholder、instruction、state 和 action。

## src/datasets

- `src/datasets/__init__.py`：dataset package 初始化文件。
- `src/datasets/torch_dataset.py`：将 episode 数据封装成 PyTorch Dataset。它会将 episode 展平为 frame samples，并过滤 action 维度错误、NaN/Inf 等不可训练样本。

## src/models

- `src/models/__init__.py`：model package 初始化文件。
- `src/models/mlp_policy.py`：MLP action prediction 模型。结构为 `Linear -> ReLU -> Linear -> ReLU -> Linear`。
- `src/models/cnn_policy.py`：CNN + MLP action prediction 模型。CNN 编码 RGB，MLP 编码 state，拼接后预测 action。

## docs

- `docs/file_explanation.md`：当前文件，逐个解释项目文件的作用。
- `docs/pipeline_design.md`：解释整个数据管线设计思路，包括 episode 组织方式、observation/action/language 语义、QA 必要性、格式转换意义和 baseline 实验目的。
- `docs/technical_report.md`：技术报告，系统说明项目原理、清晰运行步骤、QA 与训练验证逻辑，以及与 RT-2、Open X-Embodiment、OpenVLA、Octo、LeRobot、DROID、ACT、Diffusion Policy、LIBERO 和 RoboMimic 等前沿研究/工具链的关系。
