"""Convert raw HDF5 episode groups into the unified VLA schema."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .data_schema import Episode, make_episode, summarize_episodes
from .hdf5_loader import detect_hdf5_format, list_episode_groups, read_episode_group


def parse_hdf5_dataset(file_path: str | Path) -> List[Episode]:
    """Parse a LIBERO/RoboMimic/mock HDF5 file into unified episodes."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {path}")

    source_format = detect_hdf5_format(path)
    group_paths = list_episode_groups(path)
    if not group_paths:
        raise ValueError(f"No episode groups found in {path}")

    episodes: List[Episode] = []
    for index, group_path in enumerate(group_paths):
        raw = read_episode_group(path, group_path)
        rgb = _normalize_rgb(raw["rgb"])
        state = _normalize_2d(raw["state"], name=f"{group_path}/state")
        actions = _normalize_2d(raw["actions"], name=f"{group_path}/actions")
        num_steps = min(len(rgb), len(state), len(actions))

        episode_id = raw.get("episode_id") or f"episode_{index:03d}"
        task_name = raw.get("task_name") or _task_from_language(
            raw.get("language_instruction", "")
        )
        metadata = {
            "source_format": source_format,
            "robot": raw.get("robot") or "unknown_robot",
            "camera_names": raw.get("camera_names") or ["front"],
            "fps": int(raw.get("fps") or 10),
            "raw_group_path": group_path,
        }

        episodes.append(
            make_episode(
                episode_id=episode_id,
                task_name=task_name,
                language_instruction=raw.get("language_instruction", ""),
                rgb=rgb[:num_steps],
                state=state[:num_steps],
                actions=actions[:num_steps],
                metadata=metadata,
            )
        )
    return episodes


def print_parse_summary(episodes: List[Episode]) -> None:
    """Print compact parsing statistics for demo output."""
    summary = summarize_episodes(episodes)
    print("Parsed episode summary")
    print("----------------------")
    print(f"episode count          : {summary['num_episodes']}")
    print(f"average length         : {summary['avg_length']:.2f}")
    print(f"state_dim              : {summary['state_dim']}")
    print(f"action_dim             : {summary['action_dim']}")
    print(f"language coverage      : {summary['language_coverage']:.1%}")


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Normalize RGB arrays to uint8 shape [T, H, W, 3]."""
    arr = np.asarray(rgb)
    if arr.ndim != 4:
        raise ValueError(f"RGB must have 4 dimensions, got shape {arr.shape}")
    if arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] != 3:
        raise ValueError(f"RGB last dimension must be 3, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _normalize_2d(array: np.ndarray, name: str) -> np.ndarray:
    """Normalize state/action arrays to float32 shape [T, D]."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D after normalization, got {arr.shape}")
    return arr


def _task_from_language(language: str) -> str:
    """Create a simple task name from a language instruction."""
    text = str(language or "").strip()
    return text.replace(" ", "_") if text else "unknown_task"
