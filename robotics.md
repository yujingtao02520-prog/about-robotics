# robotics

这是我的机器人学习主线仓库。每个主题尽量按一个可复现的小步骤推进：先看数据、接口和仿真闭环，再逐步进入 policy、train、eval。

## 学习路线

```text
dataset -> replay -> simulation -> control -> policy -> train -> eval
```

当前已经整理了三个主题：

- `01_lerobot`: 先理解机器人数据集。
- `02_mujoco`: 先理解物理仿真、关节和执行器。
- `robot_vla_data_pipeline_demo`: 理解 VLA 数据工程 pipeline。

## 01 LeRobot

LeRobot 是一个重要的机器人学习框架，很多工作会直接使用或借鉴它的数据集、策略接口、训练和评估工具链。

当前已经完成：

- 下载 LeRobot 数据集：`lerobot/pusht`
- 读取数据：`LeRobotDataset`
- 回放 episode：导出 GIF
- 查看 action：shape、dtype、数值范围
- 查看 observation：state、image、key 和 shape
- 不训练，不跑 eval，不保存 checkpoint

入口：

- [01_lerobot/README.md](01_lerobot/README.md)
- [01_lerobot/explore_dataset.py](01_lerobot/explore_dataset.py)
- [01_lerobot/RUN_RESULT.md](01_lerobot/RUN_RESULT.md)

复现命令：

```powershell
cd "E:\program\learning\robotics being\01_lerobot"
.\.venv\Scripts\python.exe explore_dataset.py --repo-id lerobot/pusht --episode 0 --max-frames 120
```

核心结构：

```text
action              shape = [2]
observation.state   shape = [2]
observation.image   shape = [3, 96, 96]
```

下一步可以进入 policy：先写一个最小策略接口，但仍然不急着训练。

## 02 MuJoCo

MuJoCo 是机器人学习里常用的物理仿真器。这个主题先从最小闭环开始，再进入关节、执行器和简单控制。

入口：

- [02_mujoco/README.md](02_mujoco/README.md)
- [02_mujoco/stage1_minimal_loop.py](02_mujoco/stage1_minimal_loop.py)
- [02_mujoco/stage2_joint_actuator.py](02_mujoco/stage2_joint_actuator.py)

第一阶段目标：

- 知道 `model`、`data`、`mj_step` 是什么
- 会打开 viewer
- 会让一个物体掉落
- 会读取 `qpos`、`qvel`

第一阶段练习：

- 方块自由下落
- 小球滚动
- 单摆摆动
- 打印位置和速度

复现命令：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py box
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py ball
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py pendulum
```

第二阶段目标：

- 知道 `body`、`joint`、`geom`、`actuator` 的关系
- 会给一个关节施加控制输入
- 会控制单摆或二连杆机械臂

第二阶段练习：

- 单摆 torque 控制
- 二连杆机械臂
- 末端到达指定点

复现命令：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py pendulum --pendulum-target 35
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py arm --shoulder-torque 0.5 --elbow-torque -0.2
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py reach --target-x 0.55 --target-y 0.25
```

打开 viewer 示例：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py reach --viewer --seconds 8
```

## Robot VLA 数据清洗与训练评测 Pipeline Demo

这是一个完整但轻量级的 VLA 数据工程 demo，围绕 raw HDF5 episode 到统一 observation-action-language schema 的处理链路展开。

当前已经完成：

- 生成 LIBERO / RoboMimic 风格 mock HDF5 数据
- 解析 episode 到统一 schema
- 执行 QA 质量检查：语言缺失、动作突变、空图像、短轨迹、NaN/Inf、action 维度错误
- 逐帧 episode GIF 可视化
- 导出 LeRobot-like、ACT-like、OpenVLA-like 简化格式
- 训练 MLP 和 CNN+MLP 两个 action prediction baseline
- 输出 baseline 评估指标

入口：

- [robot_vla_data_pipeline_demo/README.md](robot_vla_data_pipeline_demo/README.md)
- [robot_vla_data_pipeline_demo/docs/pipeline_design.md](robot_vla_data_pipeline_demo/docs/pipeline_design.md)
- [robot_vla_data_pipeline_demo/docs/technical_report.md](robot_vla_data_pipeline_demo/docs/technical_report.md)

复现命令：

```powershell
cd robot_vla_data_pipeline_demo
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

这个主题的重点不是训练真实 OpenVLA 大模型，而是把机器人 VLA 数据从 raw episode、schema、QA、visualization、format conversion、baseline training 到 evaluation report 的工程链路跑通。
