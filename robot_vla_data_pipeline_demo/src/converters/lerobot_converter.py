"""Simplified LeRobot-like JSON converter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data_schema import Episode, to_jsonable


def convert_episode_to_lerobot(episode: Episode, output_dir: str | Path) -> Path:
    """Write one episode as a LeRobot-like JSON file."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{episode['episode_id']}.json"

    state = np.asarray(episode["observations"]["state"])
    actions = np.asarray(episode["actions"])
    fps = int(episode.get("metadata", {}).get("fps", 10))
    frames = []
    for t in range(int(episode["num_steps"])):
        frames.append(
            {
                "observation.images.front": f"{episode['episode_id']}/frame_{t:06d}.png",
                "observation.state": to_jsonable(state[t]),
                "action": to_jsonable(actions[t]),
                "language_instruction": episode["language_instruction"],
                "timestamp": t / max(fps, 1),
            }
        )

    payload = {
        "episode_id": episode["episode_id"],
        "task_name": episode["task_name"],
        "num_steps": int(episode["num_steps"]),
        "metadata": to_jsonable(episode.get("metadata", {})),
        "frames": frames,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path
