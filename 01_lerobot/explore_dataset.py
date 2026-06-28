from __future__ import annotations

"""Explore a LeRobot dataset before writing any policy or training code.

This script is intentionally read-only: it downloads one episode, prints the
action/observation structure, and optionally exports a replay GIF. The point is
to understand the data contract first.
"""

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
    """Import LeRobotDataset across a few LeRobot package layouts.

    LeRobot has moved modules between releases. Keeping the import logic here
    makes the rest of the script version-agnostic and easier to read.
    """

    candidates = (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
        "lerobot.datasets",
    )

    errors: list[str] = []
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["LeRobotDataset"])
            return module.LeRobotDataset
        except Exception as exc:  # noqa: BLE001 - report all import paths to the learner.
            errors.append(f"{module_name}: {exc}")

    print("Cannot import LeRobotDataset.", file=sys.stderr)
    print("Install dependencies first:", file=sys.stderr)
    print(r"  .\.venv\Scripts\python.exe -m pip install -r requirements.txt", file=sys.stderr)
    print("\nImport attempts:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(2)


def sanitize_repo_id(repo_id: str) -> str:
    """Turn a Hugging Face repo id into a safe local folder/file stem."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", repo_id).strip("_")


def to_builtin(value: Any) -> Any:
    """Convert tensors/arrays/scalars into small printable Python values."""

    if hasattr(value, "detach"):
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
    """Pretty-print feature metadata without flooding the terminal."""

    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def tensor_stats(value: Any) -> str:
    """Return shape, dtype, range, and a short value preview for one field."""

    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        if tensor.numel() == 0:
            return f"Tensor shape={shape} dtype={dtype} empty"

        numeric = tensor
        if not getattr(tensor, "is_floating_point", lambda: False)():
            try:
                numeric = tensor.float()
            except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
    """Print a terminal section divider."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_features(dataset: Any) -> None:
    """Show LeRobot feature metadata: keys, dtypes, shapes, fps, cameras."""

    features = getattr(dataset, "features", None)
    if not features:
        print("No dataset.features found.")
        return

    for key in sorted(features):
        print(f"- {key}: {compact_json(features[key])}")


def print_sample(sample: dict[str, Any]) -> None:
    """Print every key in one frame sample."""

    for key in sorted(sample):
        print(f"- {key}: {tensor_stats(sample[key])}")


def print_key_group(title: str, sample: dict[str, Any], keys: list[str]) -> None:
    """Print a named subset of sample keys, such as action or observation."""

    print_section(title)
    if not keys:
        print("No matching keys.")
        return
    for key in keys:
        print(f"- {key}: {tensor_stats(sample[key])}")


def as_image(value: Any) -> Image.Image | None:
    """Convert an image-like LeRobot field into a PIL RGB image.

    LeRobot image observations can arrive as PIL images, float tensors in
    [0, 1], uint8 tensors in [0, 255], channel-first tensors, or channel-last
    tensors. Replay code only needs a single normalized PIL representation.
    """

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if not hasattr(value, "detach"):
        return None

    tensor = value.detach().cpu()

    # Some datasets return a batch-like image field. For replay we only need
    # one frame, so take the first image.
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 2:
        array = tensor.numpy()
    elif tensor.ndim == 3:
        # Convert channel-first tensors [C, H, W] into PIL's [H, W, C].
        if tensor.shape[0] in (1, 3, 4) and tensor.shape[-1] not in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
    else:
        return None

    if np.issubdtype(array.dtype, np.floating):
        # Most LeRobot image tensors are floats in [0, 1].
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
    """Find observation fields that can be decoded as images."""

    preferred = [
        key
        for key in sample
        if key.startswith("observation") and ("image" in key or "camera" in key or "rgb" in key)
    ]
    candidates = preferred + [key for key in sample if key not in preferred]
    return [key for key in candidates if as_image(sample[key]) is not None]


def frame_label(sample: dict[str, Any], local_frame: int) -> str:
    """Build a small overlay label for the replay GIF."""

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
    """Decode image observations and save them as an animated GIF.

    The function returns the number of frames written. It does not control a
    robot or run a policy; it only visualizes the recorded demonstration.
    """

    frames: list[Image.Image] = []
    total = len(dataset)
    indices = range(0, total, max(1, stride))

    for local_i, dataset_i in enumerate(indices):
        if len(frames) >= max_frames:
            break
        sample = dataset[dataset_i]
        image = as_image(sample[image_key])
        if image is None:
            continue
        image = image.copy()
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
    """Print action/state snapshots from the start, middle, and end."""

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
    """Parse command-line options for the dataset exploration run."""

    parser = argparse.ArgumentParser(
        description="Download/read/replay a LeRobot dataset episode without training.",
    )
    parser.add_argument("--repo-id", default="lerobot/pusht", help="Hugging Face dataset repo id.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to download and inspect.")
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root. Defaults to ./data/<repo-id>.")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum GIF frames to export.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame in the GIF.")
    parser.add_argument("--skip-gif", action="store_true", help="Only inspect tensors; do not write a replay GIF.")
    parser.add_argument("--no-videos", action="store_true", help="Download parquet/metadata only. GIF replay will be skipped.")
    parser.add_argument("--video-backend", default=None, help="Optional LeRobot video backend, for example pyav.")
    parser.add_argument(
        "--return-uint8",
        action="store_true",
        help="Ask LeRobot to return uint8 image tensors instead of normalized float tensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # On Windows, Hugging Face may warn about symlinks. The cache still works;
    # this environment variable keeps the learning output focused.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # Store downloaded data inside this lesson by default. The folder is ignored
    # by Git because dataset files can be large and are reproducible.
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

    # Different LeRobot releases expose slightly different constructor
    # arguments. Check the installed signature before passing optional fields.
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
        # Constructing LeRobotDataset downloads missing files, then exposes each
        # frame as a dictionary of tensors/metadata.
        dataset = LeRobotDataset(**dataset_kwargs)
    except Exception as exc:  # noqa: BLE001
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

    # One frame is enough to inspect the policy contract:
    #   policy input  -> observation.*
    #   policy output -> action
    sample = dataset[0]

    print_section("One sample")
    print_sample(sample)

    action_keys = [key for key in sample if key == "action" or key.startswith("action.")]
    observation_keys = [key for key in sample if key.startswith("observation")]
    image_keys = find_image_keys(sample)

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

    # Use the first decodable image observation as the replay camera.
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
