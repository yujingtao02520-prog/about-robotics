# robot_vla_data_pipeline_demo

一个轻量级但完整可运行的机器人 VLA 数据清洗与训练评测 Pipeline demo。项目覆盖从 HDF5 episode 数据读取、统一 observation-action-language schema、QA 质量检查、逐帧可视化、格式转换，到两个 baseline action prediction 实验与评估报告的完整链路。

## 1. 项目简介

这个 demo 面向机器人学习 / VLA 数据工程场景，用一套 mock LIBERO / RoboMimic-like HDF5 数据模拟真实 episode 数据流。它不训练真实 OpenVLA 大模型，也不追求完全复刻官方格式；重点是展示数据工程能力：原始 episode 如何进入统一 schema，如何做质量检查，如何导出给不同训练栈，如何用小模型验证数据能被正常读取和训练。

## 2. 对应简历描述

机器人 VLA 数据清洗与训练评测 Pipeline | 清华大学科研项目 | 指导老师：杨名扬

- 构建 VLA episode 数据处理管线：基于 LIBERO / RoboMimic 解析 episode，并统一 observation-action-language 数据接口。
- 实现 episode 可视化与数据质量评估系统：开发逐帧可视化工具与 QA pipeline，支持动作突变、语言缺失、空图像、短轨迹、NaN/Inf 等质量规则。
- 完成跨格式转换与小规模验证：实现数据向 LeRobot / ACT / OpenVLA 简化格式转换，并完成 MLP 与 CNN+MLP 两个 baseline 数据验证实验。

## 3. 环境安装

建议 Python 3.10+。在项目根目录执行：

```bash
pip install -r requirements.txt
```

依赖包含 `numpy`、`pandas`、`h5py`、`opencv-python`、`matplotlib`、`torch`、`torchvision`、`tqdm`、`pyyaml`、`scikit-learn`。

## 4. 完整运行顺序

从干净目录开始，按下面命令即可跑通完整 demo：

```bash
python scripts/generate_mock_data.py
python scripts/parse_dataset.py --input data/mock/mock_robot_data.hdf5
python scripts/run_qa.py --input data/processed/episodes.pkl
python scripts/visualize_episode.py --episode_id episode_000
python scripts/convert_formats.py --input data/processed/episodes.pkl --target all
python scripts/train_baseline_mlp.py --input data/processed/episodes.pkl
python scripts/train_baseline_cnn.py --input data/processed/episodes.pkl
python scripts/evaluate_baselines.py
```

如果 `parse_dataset.py` 发现默认 mock HDF5 不存在，会自动生成 mock 数据，保证本地没有真实数据时也能演示。

## 5. 脚本说明

| 脚本 | 作用 |
|---|---|
| `scripts/generate_mock_data.py` | 生成 RoboMimic-like HDF5 mock 数据，并故意注入 QA 异常。 |
| `scripts/parse_dataset.py` | 读取 HDF5，自动识别 mock / RoboMimic-like / LIBERO-like 结构，并保存统一 schema。 |
| `scripts/run_qa.py` | 对处理后的 episode 执行质量检查，输出 CSV 和 JSON 报告。 |
| `scripts/visualize_episode.py` | 将指定 episode 渲染为 GIF，展示 RGB、语言、state、action。 |
| `scripts/convert_formats.py` | 导出 LeRobot-like JSON、ACT-like NPZ、OpenVLA-like JSONL。 |
| `scripts/train_baseline_mlp.py` | 训练 state-to-action 的 MLP baseline。 |
| `scripts/train_baseline_cnn.py` | 训练 RGB + state-to-action 的 CNN baseline。 |
| `scripts/evaluate_baselines.py` | 评估 checkpoint，输出 MSE、MAE、smoothness、per-dim MSE 和 QA 失败比例。 |

## 6. 统一数据接口

所有来源的数据会被转换成如下 episode schema：

```python
episode = {
    "episode_id": str,
    "task_name": str,
    "language_instruction": str,
    "num_steps": int,
    "observations": {
        "rgb": np.ndarray,    # [T, H, W, 3]
        "state": np.ndarray,  # [T, state_dim]
    },
    "actions": np.ndarray,    # [T, action_dim]
    "metadata": {
        "source_format": str,
        "robot": str,
        "camera_names": list,
        "fps": int,
    }
}
```

其中 observation 表示机器人看到的 RGB 和感知到的状态；action 表示控制输出，例如末端位置增量或夹爪开合；language_instruction 表示任务语言指令；episode 表示一次完整轨迹。

## 7. QA 规则

QA pipeline 默认包含 6 类规则：

- `language_missing_check`：检查语言指令是否为空。
- `action_spike_check`：检查相邻 action 的 L2 跳变是否超过阈值，默认 `3.0`。
- `empty_rgb_check`：检查 RGB 是否缺失或出现全零帧。
- `short_episode_check`：检查 episode 长度是否小于最小长度，默认 `10`。
- `nan_check`：检查 state/action 是否包含 NaN 或 Inf。
- `action_dim_check`：检查 action 维度是否等于默认 `7`。

输出文件：

- `outputs/qa_reports/qa_report.csv`
- `outputs/qa_reports/qa_summary.json`

## 8. 格式转换说明

本项目导出的是教学与简历展示用的 schema-compatible 简化格式，不是官方格式的完整复刻。

- LeRobot-like：`outputs/converted/lerobot/episode_xxx.json`，按 frame 保存 `observation.images.front`、`observation.state`、`action`、`language_instruction`、`timestamp`。
- ACT-like：`outputs/converted/act/episode_xxx.npz`，保存 `images`、`qpos`、`actions`、`language_instruction`。
- OpenVLA-like：`outputs/converted/openvla/episode_xxx.jsonl`，每行表示一帧，包含 image placeholder、instruction、state、action。

## 9. Baseline 实验

Baseline 的目标不是追求高精度，而是验证数据能被模型正常读取、batch、forward、loss、backward 和评估。

- MLP Policy：输入 `state [state_dim]`，输出 `action [action_dim]`。
- CNN + MLP Policy：输入 `RGB [3,H,W]` 与 `state [state_dim]`，融合后输出 action。

训练默认设置：

- train / val split = 8:2
- epochs = 5
- batch_size = 32
- device = CPU
- checkpoints 输出到 `outputs/checkpoints/`

训练数据集会过滤 action 维度错误、NaN/Inf 等不能直接训练的帧；这些异常仍会保留在 QA 报告中。

## 10. 输出结果示例

完整流程会生成：

- `data/mock/mock_robot_data.hdf5`
- `data/processed/episodes.pkl`
- `data/processed/episodes.jsonl`
- `outputs/qa_reports/qa_report.csv`
- `outputs/qa_reports/qa_summary.json`
- `outputs/visualizations/episode_000.gif` 或 PNG 序列
- `outputs/converted/lerobot/`
- `outputs/converted/act/`
- `outputs/converted/openvla/`
- `outputs/checkpoints/mlp_policy.pt`
- `outputs/checkpoints/cnn_policy.pt`
- `outputs/baseline_eval.csv`

`evaluate_baselines.py` 会打印类似表格：

```text
baseline  num_samples      mse      mae  failed_episode_ratio_from_qa
     mlp          900  0.00231  0.03812                           0.35
     cnn          900  0.00274  0.04108                           0.35
```

具体数值会随随机种子、PyTorch 版本略有变化。

## 11. 与真实数据/格式的关系

- LIBERO / RoboMimic：真实项目中常见 HDF5 group layout 会更复杂，例如多相机、多模态观测、不同 action 定义。本 demo 的 parser 提供了可扩展入口，核心思路是将不同原始结构映射到统一 episode schema。
- LeRobot / ACT / OpenVLA：真实训练框架对目录、视频编码、parquet/json、metadata、tokenization 等有更严格要求。本 demo 只保留最小字段，目的是展示跨格式转换的工程接口和数据语义对应关系。
- Baseline：MLP 与 CNN+MLP 不代表 VLA 大模型能力，只用于验证数据闭环是否能被 PyTorch 正常消费。

## 12. 技术报告

更完整的原理、运行步骤和前沿研究关系说明见 `docs/technical_report.md`。该报告解释了本项目与 RT-2、Open X-Embodiment/RT-X、OpenVLA、Octo、LeRobot、DROID、ACT、Diffusion Policy、LIBERO 和 RoboMimic 等工作的关系。
