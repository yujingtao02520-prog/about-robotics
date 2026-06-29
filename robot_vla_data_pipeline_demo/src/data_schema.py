"""Unified observation-action-language episode schema helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

Episode = Dict[str, Any]


def make_episode(
    episode_id: str,
    task_name: str,
    language_instruction: str,
    rgb: np.ndarray,
    state: np.ndarray,
    actions: np.ndarray,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Episode:
    """Create a dictionary following the unified VLA episode schema."""
    rgb_array = np.asarray(rgb)
    state_array = np.asarray(state, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    num_steps = int(min(len(rgb_array), len(state_array), len(action_array)))

    return {
        "episode_id": str(episode_id),
        "task_name": str(task_name),
        "language_instruction": str(language_instruction or ""),
        "num_steps": num_steps,
        "observations": {
            "rgb": rgb_array[:num_steps],
            "state": state_array[:num_steps],
        },
        "actions": action_array[:num_steps],
        "metadata": dict(metadata or {}),
    }


def validate_minimal_episode(episode: Episode) -> None:
    """Raise a clear error when a required schema field is missing."""
    required_top = [
        "episode_id",
        "task_name",
        "language_instruction",
        "num_steps",
        "observations",
        "actions",
        "metadata",
    ]
    for key in required_top:
        if key not in episode:
            raise ValueError(f"Episode is missing required key: {key}")
    for key in ["rgb", "state"]:
        if key not in episode["observations"]:
            raise ValueError(f"Episode observations missing required key: {key}")


def save_episodes_pickle(episodes: Iterable[Episode], output_path: Path | str) -> None:
    """Save parsed episodes to a pickle file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(list(episodes), f)


def load_episodes_pickle(input_path: Path | str) -> List[Episode]:
    """Load episodes from a pickle file and validate the minimal schema."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed episode file not found: {path}")
    with path.open("rb") as f:
        episodes = pickle.load(f)
    if not isinstance(episodes, list):
        raise ValueError(f"Expected a list of episodes in {path}")
    for episode in episodes:
        validate_minimal_episode(episode)
    return episodes


def summarize_episodes(episodes: List[Episode]) -> Dict[str, Any]:
    """Compute compact dataset-level statistics for console logging."""
    if not episodes:
        return {
            "num_episodes": 0,
            "avg_length": 0.0,
            "state_dim": None,
            "action_dim": None,
            "language_coverage": 0.0,
        }

    lengths = [int(ep["num_steps"]) for ep in episodes]
    state_dims = [
        int(np.asarray(ep["observations"]["state"]).shape[-1])
        for ep in episodes
        if np.asarray(ep["observations"]["state"]).ndim == 2
    ]
    action_dims = [
        int(np.asarray(ep["actions"]).shape[-1])
        for ep in episodes
        if np.asarray(ep["actions"]).ndim == 2
    ]
    language_count = sum(
        1 for ep in episodes if str(ep.get("language_instruction", "")).strip()
    )

    return {
        "num_episodes": len(episodes),
        "avg_length": float(np.mean(lengths)),
        "state_dim": _most_common(state_dims),
        "action_dim": _most_common(action_dims),
        "language_coverage": float(language_count / len(episodes)),
    }


def save_episodes_jsonl(episodes: Iterable[Episode], output_path: Path | str) -> None:
    """Save a lightweight JSONL view for inspection without image arrays."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for episode in episodes:
            item = {
                "episode_id": episode["episode_id"],
                "task_name": episode["task_name"],
                "language_instruction": episode["language_instruction"],
                "num_steps": episode["num_steps"],
                "state_shape": list(np.asarray(episode["observations"]["state"]).shape),
                "rgb_shape": list(np.asarray(episode["observations"]["rgb"]).shape),
                "action_shape": list(np.asarray(episode["actions"]).shape),
                "metadata": to_jsonable(episode.get("metadata", {})),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def to_jsonable(value: Any) -> Any:
    """Convert numpy values and non-finite floats into JSON-safe objects."""
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        value_float = float(value)
        return value_float if np.isfinite(value_float) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def _most_common(values: List[int]) -> Optional[int]:
    """Return the most common integer value, or None for an empty list."""
    if not values:
        return None
    return max(set(values), key=values.count)
