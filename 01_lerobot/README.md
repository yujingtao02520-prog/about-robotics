# 第一课：LeRobot 数据集读取与回放

目标很小：先认识 LeRobot 的数据，不训练。

这一步只做：

- 下载一个 LeRobot 数据集
- 用 `LeRobotDataset` 读取数据
- 回放一个 episode，导出 GIF
- 打印 `action` 的形状、数值、范围
- 打印 `observation` 的 key、形状、数值、图像尺寸

暂时不做：

- 不创建 policy
- 不训练 train
- 不跑 eval
- 不保存 checkpoint

## 为什么先看数据

LeRobot 常见主线是：

```text
dataset -> replay -> policy -> train -> eval
```

现在只走前两步。policy 的输入就是 observation，输出就是 action；所以在写训练代码之前，先确认 observation/action 到底长什么样。

## 安装

当前 workspace 里普通 `python` 不在 PATH，可以先用已有的 Python 3.12 环境创建一个本目录专用虚拟环境：

```powershell
cd "E:\program\learning\robotics being\01_lerobot"
..\being_h07_tiny_demo\runenv\Scripts\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果 LeRobot 下载 Hugging Face 数据集时网络较慢，这是正常的；第一次会下载 metadata、parquet 和视频文件，之后会使用本地缓存。

## 运行

默认读取 `lerobot/pusht` 的第 0 个 episode：

```powershell
.\.venv\Scripts\python.exe explore_dataset.py
```

也可以显式指定：

```powershell
.\.venv\Scripts\python.exe explore_dataset.py --repo-id lerobot/pusht --episode 0 --max-frames 120
```

脚本会把数据放在：

```text
01_lerobot/data/
```

回放 GIF 会输出到：

```text
01_lerobot/outputs/
```

## 重点看什么

运行输出里先看这些块：

- `Dataset`：数据集总帧数、episode 数、fps、features
- `One sample`：第一个样本里有哪些 key
- `Action`：action 的 shape、dtype、前几个数值、min/max/mean
- `Observation`：`observation.*` 的 key、shape、dtype
- `Replay`：导出的 GIF 路径

对 `lerobot/pusht` 来说，通常会看到：

- `action` 是一个低维向量，表示下一步控制目标
- `observation.state` 是低维状态
- `observation.image` 或 `observation.images.*` 是图像观测

不同 LeRobot 数据集的 action/observation 命名和维度会不同，所以以后换数据集时先跑这个脚本，而不是直接写训练。

