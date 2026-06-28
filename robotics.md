# robotics

这是我的机器人学习主线仓库。每个主题尽量按一个可复现的小步骤推进：先看数据和接口，再逐步进入 policy、train、eval。

## 学习路线

```text
dataset -> replay -> policy -> train -> eval
```

第一阶段先不训练，先把数据看明白。

## 01 LeRobot

LeRobot 是目前很重要的机器人学习框架之一，很多新工作会直接使用或借鉴它的数据集、策略接口、训练和评估工具链。

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

本次看到的核心结构：

```text
action              shape = [2]
observation.state   shape = [2]
observation.image   shape = [3, 96, 96]
```

下一步再进入 policy：先写一个最小策略接口，但仍然不急着训练。

