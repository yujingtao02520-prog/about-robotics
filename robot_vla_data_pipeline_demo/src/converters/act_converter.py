"""Simplified ACT-like NPZ converter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data_schema import Episode, to_jsonable


def convert_episode_to_act(episode: Episode, output_dir: str | Path) -> Path:
    """Write one episode as an ACT-like compressed NPZ file."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{episode['episode_id']}.npz"
    np.savez_compressed(
        output_path,
        images=np.asarray(episode["observations"]["rgb"]),
        qpos=np.asarray(episode["observations"]["state"]),
        actions=np.asarray(episode["actions"]),
        language_instruction=np.array(episode["language_instruction"]),
        metadata=np.array(json.dumps(to_jsonable(episode.get("metadata", {})))),
    )
    return output_path
