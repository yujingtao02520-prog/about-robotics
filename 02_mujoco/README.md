# MuJoCo 学习笔记：第一阶段和第二阶段

这个目录是 MuJoCo 入门实验区。目标不是一下子做复杂机器人，而是先把最小闭环跑顺：

1. 从 XML/MJCF 加载模型。
2. 创建仿真状态。
3. 每一帧调用 `mujoco.mj_step(model, data)`。
4. 读取 `qpos`、`qvel`、`site_xpos` 等状态。
5. 通过 `data.ctrl` 给 actuator 输入控制量。

所有命令都从仓库根目录运行：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py box
```

## 核心概念

`model` 是模型说明书。它来自 XML/MJCF，里面保存 body、joint、geom、actuator、质量、阻尼、重力、时间步长等信息。一般创建一次后不频繁改。

`data` 是当前仿真状态。它会随着仿真不断变化，常用字段包括：

- `data.time`: 当前仿真时间，单位秒。
- `data.qpos`: 广义位置，比如自由关节的位置和四元数、hinge 关节角度。
- `data.qvel`: 广义速度，比如线速度、角速度、关节角速度。
- `data.ctrl`: actuator 的控制输入。
- `data.xpos`: body 在世界坐标系下的位置。
- `data.site_xpos`: site 在世界坐标系下的位置，常用来表示末端执行器。
- `data.qfrc_bias`: MuJoCo 计算出的偏置力，包含重力、科氏力等，做重力补偿时很有用。

`mujoco.mj_step(model, data)` 是物理仿真的一步更新。它读取 `model` 和当前 `data`，然后原地修改 `data`。

最小代码长这样：

```python
import mujoco

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

for _ in range(1000):
    mujoco.mj_step(model, data)
    print(data.qpos, data.qvel)
```

## XML 元素关系

`body` 是刚体节点，也是坐标系节点。子 body 会挂在父 body 下面，形成层级。

`joint` 定义 body 相对父 body 怎么运动：

- `freejoint`: 6 自由度，能平移也能旋转。`qpos` 有 7 个量，`qvel` 有 6 个量。
- `hinge`: 1 自由度旋转关节。`qpos` 是角度，`qvel` 是角速度。

`geom` 是几何形状，可以负责显示、碰撞、质量。常见类型有 `box`、`sphere`、`capsule`、`plane`。

`site` 是一个轻量标记点。它通常不参与碰撞，适合用来标记目标点、末端执行器、传感器位置。

`actuator` 是执行器。这里主要用 `motor`，它把 `data.ctrl[i]` 映射成某个 joint 上的力或扭矩。

## 第一阶段：物理仿真最小闭环

脚本：

[stage1_minimal_loop.py](./stage1_minimal_loop.py)

模型：

- [stage1_free_fall_box.xml](./stage1_free_fall_box.xml)
- [stage1_rolling_ball.xml](./stage1_rolling_ball.xml)
- [pendulum.xml](./pendulum.xml)

### 第一阶段参数

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py DEMO [options]
```

`DEMO` 可选：

- `box`: 方块自由下落。
- `ball`: 小球滚动。
- `pendulum`: 单摆摆动。

通用参数：

- `--seconds 2.0`: 仿真持续时间。
- `--sample-dt 0.1`: 每隔多少秒打印一次状态。
- `--viewer`: 打开 MuJoCo 可视化窗口。

`ball` 参数：

- `--speed 1.2`: 小球初始 x 方向速度。

`pendulum` 参数：

- `--initial-angle 35`: 单摆初始角度，单位 degree。

### 练习 1：方块自由下落

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py box
```

打开 viewer：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py box --viewer --seconds 5
```

观察：

- 方块使用 `freejoint`，所以它有完整的 3D 位姿。
- `qpos[0:3]` 是 `[x, y, z]`。
- `qvel[0:3]` 是 `[vx, vy, vz]`。
- 落地前 `z` 下降，`vz` 变成负数。

### 练习 2：小球滚动

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py ball
```

换初速度：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py ball --speed 2.0
```

打开 viewer：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py ball --viewer --seconds 6
```

观察：

- `qpos = [x, y, z, qw, qx, qy, qz]`，后四个是四元数姿态。
- `qvel = [vx, vy, vz, wx, wy, wz]`，后三个是角速度。
- 小球接触地面后，摩擦会把线速度和角速度联系起来。

### 练习 3：单摆摆动

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py pendulum
```

换初始角度：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py pendulum --initial-angle 60
```

打开 viewer：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage1_minimal_loop.py pendulum --viewer --seconds 8
```

观察：

- 单摆用 `hinge` 关节。
- `qpos` 只有一个角度，单位是弧度。
- `qvel` 只有一个角速度，单位是弧度每秒。
- 脚本打印时转成了 degree，方便阅读。

## 第二阶段：关节和执行器

脚本：

[stage2_joint_actuator.py](./stage2_joint_actuator.py)

模型：

- [stage2_single_pendulum.xml](./stage2_single_pendulum.xml)
- [stage2_two_link_arm.xml](./stage2_two_link_arm.xml)

这一阶段的关键是：`actuator` 不直接控制 body，它通常绑定到某个 `joint`。你写入 `data.ctrl[i]`，MuJoCo 根据 actuator 的配置把它变成关节力或关节扭矩。

### 第二阶段参数

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py DEMO [options]
```

`DEMO` 可选：

- `pendulum`: 单摆 torque 控制。
- `arm`: 二连杆机械臂手动扭矩控制。
- `reach`: 二连杆末端到达目标点。

通用参数：

- `--seconds 3.0`: 仿真持续时间。
- `--sample-dt 0.2`: 每隔多少秒打印一次状态。
- `--viewer`: 打开 MuJoCo 可视化窗口。

`pendulum` 参数：

- `--pendulum-torque 1.0`: 给单摆 motor 一个常量扭矩。
- `--pendulum-target 35`: 用 PD 加重力补偿控制单摆去目标角度，单位 degree。

`arm` 参数：

- `--shoulder-torque 0.5`: shoulder 关节扭矩。
- `--elbow-torque -0.2`: elbow 关节扭矩。

`reach` 参数：

- `--target-x 0.55`: 目标点 x 坐标。
- `--target-y 0.25`: 目标点 y 坐标。
- `--kp 12.0`: PD 位置增益。越大越想快速到目标，但太大会抖。
- `--kd 2.0`: PD 速度阻尼。越大越稳，但太大会慢。

### 练习 1：单摆 torque 控制

直接给 motor 一个常量扭矩：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py pendulum --pendulum-torque 1.0
```

控制到目标角度：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py pendulum --pendulum-target 35
```

打开 viewer：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py pendulum --pendulum-target 35 --viewer --seconds 8
```

这里的目标角控制用了：

```python
torque = data.qfrc_bias[dof] + kp * (target - qpos) - kd * qvel
data.ctrl[0] = torque
```

解释：

- `kp * (target - qpos)`: 角度离目标越远，推得越用力。
- `-kd * qvel`: 速度越快，阻尼越大，防止一直振荡。
- `data.qfrc_bias[dof]`: 重力补偿。没有它时，单摆会被重力拉偏，停不到目标角。
- `ctrlrange="-4 4"`: XML 里限制了最大扭矩，所以你会看到控制量有时被夹到 `4.0`。

### 练习 2：二连杆机械臂

给两个关节手动施加扭矩：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py arm --shoulder-torque 0.5 --elbow-torque -0.2
```

打开 viewer：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py arm --viewer --seconds 8
```

XML 结构可以这样理解：

```text
world
└── link1 body
    ├── shoulder hinge joint
    ├── link1 capsule geom
    └── link2 body
        ├── elbow hinge joint
        ├── link2 capsule geom
        └── end_effector site
```

控制对应关系：

- `data.ctrl[0]`: shoulder motor。
- `data.ctrl[1]`: elbow motor。
- `data.qpos[0]`: shoulder 角度。
- `data.qpos[1]`: elbow 角度。
- `data.site_xpos[end_effector]`: 末端执行器世界坐标。

### 练习 3：末端到达指定点

默认目标点：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py reach --target-x 0.55 --target-y 0.25
```

换目标点：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py reach --target-x 0.25 --target-y 0.55
```

打开 viewer：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py reach --target-x 0.55 --target-y 0.25 --viewer --seconds 8
```

实现逻辑：

1. 用二连杆解析 IK，根据 `(target_x, target_y)` 算出 shoulder 和 elbow 目标角。
2. 用 PD 控制两个关节向目标角运动。
3. 用 `data.site_xpos` 读取绿色末端点的位置。
4. 打印 `error`，也就是末端点到目标点的距离。

可视化中：

- 红色小球是目标点 `target site`。
- 绿色小球是末端执行器 `end_effector site`。
- 蓝色杆是第一段 link。
- 黄色杆是第二段 link。

## 常用调参实验

让 reach 更激进：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py reach --kp 30 --kd 3
```

让 reach 更柔和：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py reach --kp 6 --kd 1
```

目标点放远一点，观察 reach 自动夹到可达范围：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py reach --target-x 1.2 --target-y 0.1
```

给二连杆一个更大的手动扭矩：

```powershell
.\.venv_mujoco\Scripts\python.exe .\02_mujoco\stage2_joint_actuator.py arm --shoulder-torque 2.0 --elbow-torque -1.0 --viewer --seconds 10
```

## 推荐阅读代码顺序

1. `stage1_minimal_loop.py`: 看最小 `model -> data -> mj_step -> print` 闭环。
2. `stage1_free_fall_box.xml`: 看 `freejoint` 和 `geom`。
3. `stage2_single_pendulum.xml`: 看 `hinge joint` 和 `motor actuator`。
4. `stage2_joint_actuator.py` 的 `set_controls`: 看如何写入 `data.ctrl`。
5. `stage2_joint_actuator.py` 的 `ik_2link`: 看二连杆解析 IK。
6. `stage2_two_link_arm.xml`: 看 body 嵌套、site、两个 actuator 的对应关系。
