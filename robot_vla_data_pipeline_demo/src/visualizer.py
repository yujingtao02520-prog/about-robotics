"""Frame-by-frame visualization for unified VLA episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from .data_schema import Episode


def find_episode(episodes: list[Episode], episode_id: str) -> Episode:
    """Return an episode by id or raise a helpful error."""
    for episode in episodes:
        if episode["episode_id"] == episode_id:
            return episode
    available = ", ".join(ep["episode_id"] for ep in episodes[:5])
    raise KeyError(f"Episode {episode_id!r} not found. First available ids: {available}")


def save_episode_visualization(
    episode: Episode,
    output_dir: str | Path,
    fps: int = 8,
    max_frames: Optional[int] = None,
) -> Path:
    """Save an episode visualization as GIF, falling back to PNG frames."""
    rgb = np.asarray(episode["observations"]["rgb"])
    state = np.asarray(episode["observations"]["state"])
    actions = np.asarray(episode["actions"])
    if rgb.ndim != 4:
        raise ValueError(f"RGB must have shape [T,H,W,3], got {rgb.shape}")

    total_frames = int(episode["num_steps"])
    if max_frames is not None:
        total_frames = min(total_frames, int(max_frames))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    gif_path = output_root / f"{episode['episode_id']}.gif"

    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(8.5, 4.0))
    fig.tight_layout()

    def draw_frame(frame_index: int) -> None:
        ax_img.clear()
        ax_text.clear()
        ax_img.imshow(rgb[frame_index])
        ax_img.set_title(f"frame {frame_index:03d}")
        ax_img.axis("off")
        text = _format_side_text(episode, state, actions, frame_index)
        ax_text.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=8)
        ax_text.axis("off")

    def update(frame_index: int):
        draw_frame(frame_index)
        return []

    try:
        anim = animation.FuncAnimation(
            fig, update, frames=total_frames, interval=1000 / max(fps, 1), blit=False
        )
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer)
        plt.close(fig)
        return gif_path
    except Exception as exc:
        png_dir = output_root / episode["episode_id"]
        png_dir.mkdir(parents=True, exist_ok=True)
        for frame_index in range(total_frames):
            draw_frame(frame_index)
            fig.savefig(png_dir / f"frame_{frame_index:04d}.png", dpi=120)
        plt.close(fig)
        print(f"GIF export failed ({exc}); saved PNG frames instead.")
        return png_dir


def _format_side_text(
    episode: Episode, state: np.ndarray, actions: np.ndarray, frame_index: int
) -> str:
    """Format language, state, and action values for the visualization panel."""
    state_values = _slice_to_string(state, frame_index, limit=5)
    action_values = _slice_to_string(actions, frame_index, limit=7)
    return "\n".join(
        [
            f"episode_id: {episode['episode_id']}",
            f"task: {episode['task_name']}",
            f"instruction: {episode['language_instruction'] or '<missing>'}",
            f"num_steps: {episode['num_steps']}",
            "",
            f"state[:5]: {state_values}",
            f"action: {action_values}",
        ]
    )


def _slice_to_string(array: np.ndarray, frame_index: int, limit: int) -> str:
    """Render one vector as a compact string."""
    if array.ndim != 2 or frame_index >= len(array):
        return "<invalid>"
    values = array[frame_index, :limit]
    return np.array2string(values, precision=3, suppress_small=True)
