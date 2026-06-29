"""HDF5 discovery and reading helpers for LIBERO/RoboMimic-like datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np


def detect_hdf5_format(file_path: str | Path) -> str:
    """Infer a coarse source format from groups and file attributes."""
    with h5py.File(file_path, "r") as f:
        source_attr = _decode_attr(f.attrs.get("source_format"))
        if source_attr:
            return str(source_attr)
        if "data" in f and any(name.startswith("demo") for name in f["data"].keys()):
            return "robomimic"
        if "episodes" in f:
            return "libero"
    return "unknown"


def list_episode_groups(file_path: str | Path) -> List[str]:
    """Return HDF5 group paths that look like robot episodes."""
    groups: List[str] = []
    with h5py.File(file_path, "r") as f:
        if "data" in f:
            groups.extend(
                f"data/{name}"
                for name, value in f["data"].items()
                if isinstance(value, h5py.Group)
            )
        if "episodes" in f:
            groups.extend(
                f"episodes/{name}"
                for name, value in f["episodes"].items()
                if isinstance(value, h5py.Group)
            )
        if not groups:
            groups.extend(
                name for name, value in f.items() if isinstance(value, h5py.Group)
            )
    return sorted(groups)


def read_episode_group(file_path: str | Path, group_path: str) -> Dict[str, Any]:
    """Read one episode group into raw numpy arrays and metadata."""
    with h5py.File(file_path, "r") as f:
        group = f[group_path]
        rgb = _find_dataset(
            group,
            [
                "obs/rgb",
                "obs/agentview_rgb",
                "obs/front_rgb",
                "observations/rgb",
                "observations/images/front",
                "rgb",
            ],
        )
        state = _find_dataset(
            group,
            [
                "obs/state",
                "obs/robot_state",
                "obs/proprio",
                "observations/state",
                "states",
                "state",
            ],
        )
        actions = _find_dataset(group, ["actions", "action", "actions_abs"])

        if rgb is None:
            raise ValueError(f"No RGB dataset found in group {group_path}")
        if state is None:
            raise ValueError(f"No state dataset found in group {group_path}")
        if actions is None:
            raise ValueError(f"No action dataset found in group {group_path}")

        raw = {
            "group_path": group_path,
            "episode_id": _decode_attr(group.attrs.get("episode_id")),
            "task_name": _decode_attr(group.attrs.get("task_name")),
            "language_instruction": _read_language(group),
            "robot": _decode_attr(group.attrs.get("robot")),
            "camera_names": _read_camera_names(group.attrs.get("camera_names")),
            "fps": _decode_attr(group.attrs.get("fps")),
            "rgb": np.asarray(rgb),
            "state": np.asarray(state),
            "actions": np.asarray(actions),
            "attrs": {key: _decode_attr(value) for key, value in group.attrs.items()},
        }
    return raw


def _find_dataset(group: h5py.Group, candidates: List[str]) -> Optional[h5py.Dataset]:
    """Return the first matching dataset under a group."""
    for name in candidates:
        if name in group and isinstance(group[name], h5py.Dataset):
            return group[name]
    return None


def _read_language(group: h5py.Group) -> str:
    """Read a language instruction from attrs or known dataset names."""
    for key in ["language_instruction", "instruction", "lang", "task_description"]:
        if key in group.attrs:
            return str(_decode_attr(group.attrs.get(key)) or "")
        if key in group and isinstance(group[key], h5py.Dataset):
            value = np.asarray(group[key])
            if value.shape == ():
                return str(_decode_attr(value.item()) or "")
    return ""


def _read_camera_names(value: Any) -> List[str]:
    """Decode camera names stored as a list, bytes, or comma-separated string."""
    decoded = _decode_attr(value)
    if decoded is None:
        return ["front"]
    if isinstance(decoded, str):
        return [item.strip() for item in decoded.split(",") if item.strip()]
    if isinstance(decoded, (list, tuple, np.ndarray)):
        return [str(_decode_attr(item)) for item in decoded]
    return [str(decoded)]


def _decode_attr(value: Any) -> Any:
    """Decode HDF5 attributes into regular Python values."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_decode_attr(item) for item in value.tolist()]
    return value
