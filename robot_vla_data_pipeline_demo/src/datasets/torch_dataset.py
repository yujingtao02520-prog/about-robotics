"""PyTorch datasets built from unified VLA episode dictionaries."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data_schema import Episode


class FrameActionDataset(Dataset):
    """Flatten valid episode frames into state/action or image/state/action samples."""

    def __init__(
        self,
        episodes: List[Episode],
        use_images: bool = False,
        expected_state_dim: Optional[int] = None,
        expected_action_dim: Optional[int] = None,
    ) -> None:
        self.use_images = use_images
        self.state_dim, self.action_dim = infer_common_dims(
            episodes, expected_state_dim, expected_action_dim
        )
        self.samples = []
        self.skipped_frames = 0
        self.skipped_episodes = 0

        for episode in episodes:
            state = np.asarray(episode["observations"]["state"], dtype=np.float32)
            actions = np.asarray(episode["actions"], dtype=np.float32)
            rgb = np.asarray(episode["observations"]["rgb"])
            if state.ndim != 2 or actions.ndim != 2:
                self.skipped_episodes += 1
                continue
            if state.shape[1] != self.state_dim or actions.shape[1] != self.action_dim:
                self.skipped_episodes += 1
                continue
            if use_images and (rgb.ndim != 4 or rgb.shape[0] < len(state)):
                self.skipped_episodes += 1
                continue

            steps = min(len(state), len(actions), len(rgb) if use_images else len(state))
            for t in range(steps):
                if not np.all(np.isfinite(state[t])) or not np.all(np.isfinite(actions[t])):
                    self.skipped_frames += 1
                    continue
                sample = {
                    "state": state[t].astype(np.float32),
                    "action": actions[t].astype(np.float32),
                }
                if use_images:
                    image = rgb[t].astype(np.float32) / 255.0
                    sample["image"] = np.transpose(image, (2, 0, 1))
                self.samples.append(sample)

        if not self.samples:
            raise ValueError("No valid training samples after filtering episodes.")

    def __len__(self) -> int:
        """Return the number of valid frame samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one training sample as tensors."""
        sample = self.samples[index]
        item = {
            "state": torch.from_numpy(sample["state"]),
            "action": torch.from_numpy(sample["action"]),
        }
        if self.use_images:
            item["image"] = torch.from_numpy(sample["image"])
        return item

    def summary(self) -> Dict[str, int]:
        """Return filtering statistics for logging."""
        return {
            "num_samples": len(self.samples),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "skipped_frames": self.skipped_frames,
            "skipped_episodes": self.skipped_episodes,
        }


def infer_common_dims(
    episodes: List[Episode],
    expected_state_dim: Optional[int] = None,
    expected_action_dim: Optional[int] = None,
) -> Tuple[int, int]:
    """Infer common state/action dimensions, unless explicit values are provided."""
    state_dims = []
    action_dims = []
    for episode in episodes:
        state = np.asarray(episode["observations"]["state"])
        actions = np.asarray(episode["actions"])
        if state.ndim == 2:
            state_dims.append(int(state.shape[1]))
        if actions.ndim == 2:
            action_dims.append(int(actions.shape[1]))
    state_dim = expected_state_dim or _counter_mode(state_dims, "state")
    action_dim = expected_action_dim or _counter_mode(action_dims, "action")
    return int(state_dim), int(action_dim)


def _counter_mode(values: List[int], label: str) -> int:
    """Return the most frequent dimension value."""
    if not values:
        raise ValueError(f"Could not infer {label} dimension from episodes.")
    return int(Counter(values).most_common(1)[0][0])
