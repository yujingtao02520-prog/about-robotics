# -*- coding: utf-8 -*-
"""第一课：在写 policy / train / eval 之前，先把 LeRobot 数据看清楚。

这个脚本故意只做“读数据”和“看数据”，不做训练：

1. 从 Hugging Face 下载一个 LeRobot 数据集 episode。
2. 用 LeRobotDataset 把 episode 读成 Python 字典。
3. 打印 action 和 observation 的 key、shape、dtype、数值范围。
4. 把 observation.image 解码成图片，导出一个 GIF 回放。

学习机器人策略时，最重要的接口关系是：

    observation -> policy -> action

所以第一步不是急着训练模型，而是先确认 observation 和 action 到底长什么样。
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent


def import_lerobot_dataset():
    """兼容不同 LeRobot 版本，导入 LeRobotDataset。

    LeRobot 还在快速迭代，不同版本里 LeRobotDataset 的导入路径可能不完全一样。
    这里按多个候选路径依次尝试，后面的主流程就不用关心安装的是哪一个小版本。

    返回值：
        LeRobotDataset 类。后面会用它创建 dataset 对象。
    """

    candidates = (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
        "lerobot.datasets",
    )

    errors: list[str] = []
    for module_name in candidates:
        try:
            # __import__ 允许用字符串动态导入模块，适合这种“兼容多个路径”的场景。
            module = __import__(module_name, fromlist=["LeRobotDataset"])
            return module.LeRobotDataset
        except Exception as exc:  # noqa: BLE001 - 学习脚本里保留完整错误，便于排查环境。
            errors.append(f"{module_name}: {exc}")

    print("Cannot import LeRobotDataset.", file=sys.stderr)
    print("Install dependencies first:", file=sys.stderr)
    print(r"  .\.venv\Scripts\python.exe -m pip install -r requirements.txt", file=sys.stderr)
    print("\nImport attempts:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(2)


def sanitize_repo_id(repo_id: str) -> str:
    """把 Hugging Face repo id 转成适合作为本地文件夹名的字符串。

    例如：
        lerobot/pusht -> lerobot_pusht

    这样数据会默认下载到：
        01_lerobot/data/lerobot_pusht/
    """

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", repo_id).strip("_")


def to_builtin(value: Any) -> Any:
    """把 tensor / ndarray / 标量转换成适合打印的小型 Python 值。

    LeRobotDataset 返回的样本里经常混有 torch.Tensor、numpy.ndarray、字符串和布尔值。
    直接 print 大 tensor 会刷屏，所以这里：

    - 0 维或单元素 tensor 转成普通 Python 标量。
    - 多元素 tensor / ndarray 只展示前 8 个值。
    - 其他对象退化成字符串。
    """

    if hasattr(value, "detach"):
        # torch.Tensor 有 detach() 方法。先 detach 再 cpu，避免依赖 GPU 或 autograd。
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.flatten()[:8].tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.flatten()[:8].tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compact_json(value: Any, limit: int = 180) -> str:
    """把 feature metadata 压成一行，避免终端输出太长。

    dataset.features 里会包含 dtype、shape、fps、video_info 等信息。
    video_info 通常比较长，所以限制显示长度，只保留最关键的前半段。
    """

    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def tensor_stats(value: Any) -> str:
    """生成一个字段的统计摘要。

    这个函数是“看数据”的核心之一。对 action / observation 来说，我们最关心：

    - shape：维度是多少，例如 action 是 [2] 还是 [7]。
    - dtype：数据类型，例如 float32 / int64 / bool。
    - min/max/mean：数值范围是否合理。
    - preview：前几个具体数值，帮助建立直觉。
    """

    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        if tensor.numel() == 0:
            return f"Tensor shape={shape} dtype={dtype} empty"

        numeric = tensor
        if not getattr(tensor, "is_floating_point", lambda: False)():
            try:
                # bool / int tensor 也可以转成 float 来计算 min/max/mean。
                numeric = tensor.float()
            except Exception:  # noqa: BLE001 - 某些对象无法数值化时，只展示预览值。
                return f"Tensor shape={shape} dtype={dtype} preview={to_builtin(tensor)}"

        try:
            min_value = numeric.min().item()
            max_value = numeric.max().item()
            mean_value = numeric.mean().item()
            preview = to_builtin(tensor)
            return (
                f"Tensor shape={shape} dtype={dtype} "
                f"min={min_value:.4g} max={max_value:.4g} mean={mean_value:.4g} "
                f"preview={preview}"
            )
        except Exception:  # noqa: BLE001 - 保证学习脚本尽量不中断。
            return f"Tensor shape={shape} dtype={dtype} preview={to_builtin(tensor)}"

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return f"ndarray shape={list(value.shape)} dtype={value.dtype} empty"
        numeric = value.astype(np.float32, copy=False) if np.issubdtype(value.dtype, np.number) else value
        if np.issubdtype(value.dtype, np.number):
            return (
                f"ndarray shape={list(value.shape)} dtype={value.dtype} "
                f"min={numeric.min():.4g} max={numeric.max():.4g} mean={numeric.mean():.4g} "
                f"preview={to_builtin(value)}"
            )
        return f"ndarray shape={list(value.shape)} dtype={value.dtype} preview={to_builtin(value)}"

    if isinstance(value, Image.Image):
        return f"PIL.Image mode={value.mode} size={value.size}"

    return f"{type(value).__name__} value={to_builtin(value)}"


def print_section(title: str) -> None:
    """打印一个分隔标题，让终端输出更容易扫读。"""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_features(dataset: Any) -> None:
    """打印 LeRobot 数据集的 features 元信息。

    features 是理解数据集 schema 的入口。它告诉我们每个 key 的含义和形状，
    例如：

    - action: shape [2]
    - observation.state: shape [2]
    - observation.image: video / image shape [H, W, C]
    """

    features = getattr(dataset, "features", None)
    if not features:
        print("No dataset.features found.")
        return

    for key in sorted(features):
        print(f"- {key}: {compact_json(features[key])}")


def print_sample(sample: dict[str, Any]) -> None:
    """打印单帧样本里所有 key 的摘要。

    LeRobotDataset 的 dataset[0] 返回的是一个字典。这个字典就是后面训练时
    DataLoader 每次取出的基础数据结构。
    """

    for key in sorted(sample):
        print(f"- {key}: {tensor_stats(sample[key])}")


def print_key_group(title: str, sample: dict[str, Any], keys: list[str]) -> None:
    """按分组打印样本字段。

    例如把 action 单独打印，把 observation.* 单独打印。这样比混在一起看更清楚：

    - action：将来 policy 要预测的输出。
    - observation：将来 policy 要读取的输入。
    """

    print_section(title)
    if not keys:
        print("No matching keys.")
        return
    for key in keys:
        print(f"- {key}: {tensor_stats(sample[key])}")


def as_image(value: Any) -> Image.Image | None:
    """把 LeRobot 的图像字段转换成 PIL RGB 图片。

    不同数据集、不同 LeRobot 版本返回图像的方式可能不同：

    - PIL.Image
    - float tensor，范围通常是 [0, 1]
    - uint8 tensor，范围通常是 [0, 255]
    - channel-first: [C, H, W]，PyTorch 常见格式
    - channel-last: [H, W, C]，PIL / numpy 常见格式

    GIF 回放只需要统一成 PIL RGB，所以这里集中处理所有格式差异。
    """

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if not hasattr(value, "detach"):
        return None

    tensor = value.detach().cpu()

    # 有些数据集会返回类似 batch 的图像字段 [B, C, H, W] 或 [B, H, W, C]。
    # 回放时一次只需要当前帧，所以取第 0 张。
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 2:
        array = tensor.numpy()
    elif tensor.ndim == 3:
        # PyTorch 图像常见格式是 [C, H, W]，PIL 需要 [H, W, C]。
        # 如果第一个维度像通道数，并且最后一个维度不像通道数，就做 permute。
        if tensor.shape[0] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
    else:
        return None

    if np.issubdtype(array.dtype, np.floating):
        # 大多数 LeRobot 图像 tensor 是 [0, 1] 的 float。
        # GIF / PIL 需要 [0, 255] 的 uint8，所以先放大再裁剪。
        if np.nanmax(array) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[-1] == 1:
        return Image.fromarray(array[..., 0], mode="L").convert("RGB")
    if array.ndim == 3 and array.shape[-1] >= 3:
        return Image.fromarray(array[..., :3]).convert("RGB")
    return None


def find_image_keys(sample: dict[str, Any]) -> list[str]:
    """从一个样本中找出可以被当作图像解码的字段。

    优先找 observation 下面名字里带 image / camera / rgb 的 key。
    如果命名不标准，也会尝试扫描其他字段，保证脚本对新数据集更宽容。
    """

    preferred = [
        key
        for key in sample
        if key.startswith("observation") and ("image" in key or "camera" in key or "rgb" in key)
    ]
    candidates = preferred + [key for key in sample if key not in preferred]
    return [key for key in candidates if as_image(sample[key]) is not None]


def frame_label(sample: dict[str, Any], local_frame: int) -> str:
    """给 GIF 每一帧生成左上角文字标签。

    标签包含 episode、frame_index 和 timestamp。这样看回放时能知道当前播放到
    录制轨迹的哪一帧。
    """

    episode = to_builtin(sample.get("episode_index", "?"))
    frame = to_builtin(sample.get("frame_index", local_frame))
    timestamp = to_builtin(sample.get("timestamp", None))
    if isinstance(timestamp, (int, float)):
        return f"episode {episode} | frame {frame} | t={timestamp:.2f}s"
    return f"episode {episode} | frame {frame}"


def save_replay_gif(
    dataset: Any,
    image_key: str,
    output_path: Path,
    max_frames: int,
    stride: int,
) -> int:
    """把一个 episode 的图像 observation 保存成 GIF 回放。

    注意：这里没有控制机器人，也没有跑 policy。它只是把已经记录好的演示数据
    逐帧解码出来，方便我们用肉眼检查这个 episode 里发生了什么。

    参数：
        dataset: 已经选定 episode 的 LeRobotDataset。
        image_key: 用哪一路图像作为回放画面，例如 observation.image。
        output_path: GIF 输出路径。
        max_frames: 最多写多少帧，避免 GIF 太大。
        stride: 每隔多少帧取一帧，stride=2 表示隔帧采样。

    返回：
        实际写入 GIF 的帧数。
    """

    frames: list[Image.Image] = []
    total = len(dataset)
    indices = range(0, total, max(1, stride))

    for local_i, dataset_i in enumerate(indices):
        # 控制 GIF 的最大长度。数据集可能很长，初学时没必要全部导出。
        if len(frames) >= max_frames:
            break
        sample = dataset[dataset_i]
        image = as_image(sample[image_key])
        if image is None:
            continue
        image = image.copy()

        # 给每帧画一个小标签，不影响数据本身，只影响导出的可视化 GIF。
        draw = ImageDraw.Draw(image)
        draw.text(
            (8, 8),
            frame_label(sample, local_i),
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        frames.append(image)

    if not frames:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return len(frames)


def print_action_trace(dataset: Any, sample: dict[str, Any]) -> None:
    """打印 episode 开头、中间、结尾的 action / state 快照。

    单帧只能告诉我们 shape，连续几帧才能帮助理解动作随时间怎么变化。
    这里先取 3 个代表点，不做完整统计，保持输出简洁。
    """

    action_keys = [key for key in sample if key == "action" or key.startswith("action.")]
    state_keys = [key for key in sample if key.startswith("observation.state")]
    if not action_keys:
        return

    print_section("Action trace snapshots")
    indices = sorted({0, max(0, len(dataset) // 2), max(0, len(dataset) - 1)})
    for idx in indices:
        item = dataset[idx]
        print(f"[dataset index {idx}]")
        for key in action_keys:
            print(f"  {key}: {to_builtin(item[key])}")
        for key in state_keys:
            print(f"  {key}: {to_builtin(item[key])}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    默认参数面向第一课：

    - repo-id 默认是 lerobot/pusht，一个适合入门的小数据集。
    - episode 默认是 0，只看第一条轨迹。
    - max-frames 默认是 120，足够看回放，同时 GIF 不会太大。
    """

    parser = argparse.ArgumentParser(
        description="下载、读取并回放一个 LeRobot episode；只看数据，不训练。",
    )
    parser.add_argument("--repo-id", default="lerobot/pusht", help="Hugging Face 数据集仓库名。")
    parser.add_argument("--episode", type=int, default=0, help="要下载和检查的 episode 编号。")
    parser.add_argument("--root", type=Path, default=None, help="本地数据目录，默认是 ./data/<repo-id>。")
    parser.add_argument("--max-frames", type=int, default=120, help="导出 GIF 时最多写入多少帧。")
    parser.add_argument("--stride", type=int, default=1, help="GIF 每隔多少帧采样一次。")
    parser.add_argument("--skip-gif", action="store_true", help="只检查 tensor，不导出 GIF。")
    parser.add_argument("--no-videos", action="store_true", help="只下载 parquet/metadata，不下载视频，也不导出 GIF。")
    parser.add_argument("--video-backend", default=None, help="可选视频解码后端，例如 pyav。")
    parser.add_argument(
        "--return-uint8",
        action="store_true",
        help="如果当前 LeRobot 版本支持，则让图像以 uint8 tensor 返回。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Windows 下 Hugging Face Hub 可能提示“符号链接不可用”。
    # 这不影响缓存和下载；设置这个变量只是为了让终端输出更干净。
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # 默认把下载的数据放到本课目录下的 data/。
    # data/ 已经写进 .gitignore，因为数据集文件可能很大，而且可以重新下载。
    root = args.root
    if root is None:
        root = SCRIPT_DIR / "data" / sanitize_repo_id(args.repo_id)
    else:
        root = root.expanduser().resolve()

    LeRobotDataset = import_lerobot_dataset()

    print_section("Dataset")
    print(f"repo_id: {args.repo_id}")
    print(f"local root: {root}")
    print(f"episode: {args.episode}")
    print("mode: read-only dataset exploration, no policy/train/eval")

    # 不同 LeRobot 版本的 LeRobotDataset 构造函数参数可能不同。
    # 这里先读取当前安装版本的函数签名，再决定传哪些可选参数。
    # 这样脚本在小版本升级后更不容易因为“多传了不存在的参数”而崩。
    dataset_kwargs = {
        "repo_id": args.repo_id,
        "root": root,
        "episodes": [args.episode],
    }
    init_params = inspect.signature(LeRobotDataset.__init__).parameters
    if "download_videos" in init_params:
        dataset_kwargs["download_videos"] = not args.no_videos
    if "video_backend" in init_params:
        dataset_kwargs["video_backend"] = args.video_backend
    if "return_uint8" in init_params:
        dataset_kwargs["return_uint8"] = args.return_uint8
    elif args.return_uint8:
        print("This LeRobot version does not expose return_uint8; using its default image dtype.")

    try:
        # 创建 LeRobotDataset 时，如果本地 root 里没有数据，它会自动从 Hugging Face 下载。
        # 创建成功后，dataset[i] 就能取出第 i 帧，格式是一个字典。
        dataset = LeRobotDataset(**dataset_kwargs)
    except Exception as exc:  # noqa: BLE001 - 给初学阶段保留原始异常，方便定位。
        print("\nFailed to load the LeRobot dataset.", file=sys.stderr)
        print("Common causes:", file=sys.stderr)
        print("- dependencies are not installed", file=sys.stderr)
        print("- network cannot reach Hugging Face", file=sys.stderr)
        print("- video decoder dependency is missing", file=sys.stderr)
        print(f"\nOriginal error: {exc}", file=sys.stderr)
        raise

    meta = getattr(dataset, "meta", None)
    print(f"selected frames: {len(dataset)}")
    print(f"selected episodes: {getattr(dataset, 'num_episodes', 'unknown')}")
    if meta is not None:
        print(f"total frames in dataset metadata: {getattr(meta, 'total_frames', 'unknown')}")
        print(f"total episodes in dataset metadata: {getattr(meta, 'total_episodes', 'unknown')}")
        print(f"fps: {getattr(meta, 'fps', 'unknown')}")
        print(f"video keys: {getattr(meta, 'video_keys', [])}")
        print(f"camera keys: {getattr(meta, 'camera_keys', [])}")

    print_section("Features")
    print_features(dataset)

    if len(dataset) == 0:
        raise SystemExit("Selected episode has no frames.")

    # 先取第 0 帧看结构。
    # 这一步对应后续 policy 的输入输出契约：
    #   policy 的输入  -> observation.*
    #   policy 的输出  -> action
    sample = dataset[0]

    print_section("One sample")
    print_sample(sample)

    action_keys = [key for key in sample if key == "action" or key.startswith("action.")]
    observation_keys = [key for key in sample if key.startswith("observation")]
    image_keys = find_image_keys(sample)

    # 分别打印 action、observation 和图像候选字段。
    # 这一组输出就是第一课最重要的观察结果。
    print_key_group("Action", sample, action_keys)
    print_key_group("Observation", sample, observation_keys)
    print_key_group("Image observation candidates", sample, image_keys)
    print_action_trace(dataset, sample)

    print_section("Replay")
    if args.skip_gif:
        print("Skipped GIF export because --skip-gif was set.")
        return
    if args.no_videos:
        print("Skipped GIF export because --no-videos was set.")
        return
    if not image_keys:
        print("No image-like observation was found, so replay GIF was not created.")
        return

    # 如果有多路相机，这里先使用第一路能解码的图像作为回放视角。
    # 以后遇到双目相机、腕部相机时，可以扩展成多路 GIF 或拼图。
    image_key = image_keys[0]
    output_name = f"{sanitize_repo_id(args.repo_id)}_episode_{args.episode:06d}.gif"
    output_path = SCRIPT_DIR / "outputs" / output_name
    count = save_replay_gif(dataset, image_key, output_path, args.max_frames, args.stride)
    if count == 0:
        print(f"No GIF frames could be decoded from image key: {image_key}")
        return

    print(f"image key: {image_key}")
    print(f"frames written: {count}")
    print(f"gif: {output_path}")


if __name__ == "__main__":
    main()
