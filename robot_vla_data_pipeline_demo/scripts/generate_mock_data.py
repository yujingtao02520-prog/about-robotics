"""Generate a mock LIBERO/RoboMimic-style HDF5 dataset for the demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np

from src.train_utils import config_path, load_config, resolve_path


ANOMALY_PLAN = {
    2: ["missing_language"],
    5: ["action_spike"],
    7: ["empty_rgb"],
    9: ["short_episode"],
    11: ["wrong_action_dim"],
    13: ["nan_action"],
    16: ["missing_language", "empty_rgb"],
}


def create_mock_hdf5(output_path: str | Path, config: Dict[str, Any]) -> Path:
    """Create a small HDF5 file with normal and intentionally flawed episodes."""
    mock_cfg = config["mock"]
    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(mock_cfg["seed"]))
    tasks: List[str] = list(mock_cfg["tasks"])

    with h5py.File(output, "w") as f:
        f.attrs["source_format"] = "mock"
        f.attrs["layout"] = "robomimic_like"
        data_group = f.create_group("data")

        for episode_index in range(int(mock_cfg["num_episodes"])):
            anomalies = ANOMALY_PLAN.get(episode_index, [])
            episode_id = f"episode_{episode_index:03d}"
            language = str(rng.choice(tasks))
            if "missing_language" in anomalies:
                language = ""

            num_steps = int(
                rng.integers(int(mock_cfg["min_steps"]), int(mock_cfg["max_steps"]) + 1)
            )
            if "short_episode" in anomalies:
                num_steps = int(mock_cfg["short_episode_steps"])

            action_dim = int(mock_cfg["action_dim"])
            if "wrong_action_dim" in anomalies:
                action_dim = max(1, action_dim - 1)

            rgb, state, actions = _make_episode_arrays(
                rng=rng,
                num_steps=num_steps,
                image_height=int(mock_cfg["image_height"]),
                image_width=int(mock_cfg["image_width"]),
                state_dim=int(mock_cfg["state_dim"]),
                action_dim=action_dim,
                episode_index=episode_index,
            )

            if "action_spike" in anomalies and num_steps > 12:
                actions[num_steps // 2] += 8.0
            if "empty_rgb" in anomalies and num_steps > 4:
                rgb[min(4, num_steps - 1)] = 0
            if "nan_action" in anomalies and num_steps > 3:
                actions[3, min(2, action_dim - 1)] = np.nan

            group = data_group.create_group(f"demo_{episode_index:03d}")
            group.attrs["episode_id"] = episode_id
            group.attrs["task_name"] = language.replace(" ", "_") if language else "unknown_task"
            group.attrs["language_instruction"] = language
            group.attrs["robot"] = str(mock_cfg["robot"])
            group.attrs["camera_names"] = ",".join(mock_cfg["camera_names"])
            group.attrs["fps"] = int(mock_cfg["fps"])
            group.attrs["anomalies"] = ",".join(anomalies)

            obs = group.create_group("obs")
            obs.create_dataset("rgb", data=rgb, compression="gzip")
            obs.create_dataset("state", data=state)
            group.create_dataset("actions", data=actions)

    return output


def _make_episode_arrays(
    rng: np.random.Generator,
    num_steps: int,
    image_height: int,
    image_width: int,
    state_dim: int,
    action_dim: int,
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create RGB/state/action arrays with a simple learnable relationship."""
    state = np.cumsum(
        rng.normal(loc=0.0, scale=0.05, size=(num_steps, state_dim)), axis=0
    ).astype(np.float32)
    action_base = np.tanh(state[:, :action_dim])
    actions = (action_base + rng.normal(0.0, 0.03, size=action_base.shape)).astype(np.float32)
    rgb = _draw_mock_rgb_sequence(
        num_steps=num_steps,
        height=image_height,
        width=image_width,
        episode_index=episode_index,
    )
    return rgb, state, actions


def _draw_mock_rgb_sequence(
    num_steps: int, height: int, width: int, episode_index: int
) -> np.ndarray:
    """Draw a moving colored block sequence as mock robot camera frames."""
    frames = np.zeros((num_steps, height, width, 3), dtype=np.uint8)
    y_grid = np.linspace(20, 90, height, dtype=np.float32)[:, None]
    x_grid = np.linspace(10, 70, width, dtype=np.float32)[None, :]
    base = np.clip(y_grid + x_grid, 0, 255).astype(np.uint8)
    color_bank = np.array(
        [
            [220, 70, 70],
            [70, 150, 230],
            [80, 200, 120],
            [230, 180, 60],
        ],
        dtype=np.uint8,
    )
    color = color_bank[episode_index % len(color_bank)]

    for t in range(num_steps):
        frames[t, :, :, 0] = base
        frames[t, :, :, 1] = np.roll(base, shift=t % max(width, 1), axis=1)
        frames[t, :, :, 2] = 35 + (episode_index * 9) % 80
        size = 12
        x0 = int((t * 2 + episode_index * 3) % max(width - size, 1))
        y0 = int((height // 2) + 10 * np.sin(t / 6.0 + episode_index))
        y0 = int(np.clip(y0, 0, max(height - size, 0)))
        frames[t, y0 : y0 + size, x0 : x0 + size] = color
    return frames


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output HDF5 path. Defaults to paths.mock_hdf5 in config.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    output_path = resolve_path(args.output) if args.output else config_path(config, "paths", "mock_hdf5")
    output = create_mock_hdf5(output_path, config)
    print(f"Mock HDF5 dataset written to: {output}")
    print(f"Episode count: {config['mock']['num_episodes']}")
    print("Injected anomalies: missing language, action spike, empty RGB, short episode, wrong action dim, NaN action")


if __name__ == "__main__":
    main()
