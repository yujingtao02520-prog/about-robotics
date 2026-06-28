# 本次运行结果

运行命令：

```powershell
.\.venv\Scripts\python.exe explore_dataset.py --repo-id lerobot/pusht --episode 0 --max-frames 120
```

没有训练，没有 policy，没有 eval，只读取和回放数据。

## Dataset

- repo id: `lerobot/pusht`
- LeRobot: `0.5.1`
- 选中的 episode: `0`
- 选中 episode 帧数: `161`
- 数据集 metadata 总帧数: `25650`
- 数据集 metadata 总 episode 数: `206`
- fps: `10`
- 图像 key: `observation.image`

## Action 长什么样

`action` 是一个 `torch.float32` 的 2 维向量：

```text
shape = [2]
first action = [233.0, 71.0]
```

这个数据集里它表示 PushT 任务的二维控制目标。第 0 个 episode 的几个位置：

```text
frame 0:   action = [233.0, 71.0]
frame 80:  action = [260.0, 191.0]
frame 160: action = [164.0, 355.0]
```

## Observation 长什么样

这个 episode 里主要有两个 observation：

```text
observation.state
shape = [2]
dtype = torch.float32
first value = [222.0, 97.0]
```

```text
observation.image
shape = [3, 96, 96]
dtype = torch.float32
range ~= [0.2353, 1.0]
```

也就是说，这个最小例子里 observation 由低维状态和一张 RGB 图像组成；policy 以后会吃这些 observation，再预测 action。

## Replay

已经导出 GIF：

```text
outputs/lerobot_pusht_episode_000000.gif
```

本次写入 `120` 帧。

