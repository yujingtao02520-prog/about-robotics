"""Simplified OpenVLA-like JSONL converter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data_schema import Episode, to_jsonable


def convert_episode_to_openvla(episode: Episode, output_dir: str | Path) -> Path:
    """Write one episode as OpenVLA-like frame-level JSONL."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{episode['episode_id']}.jsonl"

    state = np.asarray(episode["observations"]["state"])
    actions = np.asarray(episode["actions"])
    with output_path.open("w", encoding="utf-8") as f:
        for t in range(int(episode["num_steps"])):
            row = {
                "image": f"{episode['episode_id']}/frame_{t:06d}.png",
                "instruction": episode["language_instruction"],
                "state": to_jsonable(state[t]),
                "action": to_jsonable(actions[t]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path
