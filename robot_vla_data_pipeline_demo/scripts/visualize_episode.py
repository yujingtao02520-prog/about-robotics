"""Visualize one processed episode frame by frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_schema import load_episodes_pickle
from src.train_utils import config_path, load_config, resolve_path
from src.visualizer import find_episode, save_episode_visualization


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path.")
    parser.add_argument("--input", default=None, help="Processed episodes pickle path.")
    parser.add_argument("--episode_id", default=None, help="Episode id to visualize.")
    parser.add_argument("--output-dir", default=None, help="Visualization output directory.")
    parser.add_argument("--fps", type=int, default=None, help="GIF frames per second.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to render.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    viz_cfg = config["visualization"]
    input_path = resolve_path(args.input) if args.input else config_path(config, "paths", "processed_episodes")
    output_dir = resolve_path(args.output_dir) if args.output_dir else config_path(config, "paths", "visualization_dir")
    episode_id = args.episode_id or str(viz_cfg["default_episode_id"])
    fps = args.fps if args.fps is not None else int(viz_cfg["fps"])
    max_frames = args.max_frames if args.max_frames is not None else int(viz_cfg["max_frames"])

    episodes = load_episodes_pickle(input_path)
    episode = find_episode(episodes, episode_id)
    output = save_episode_visualization(episode, output_dir, fps=fps, max_frames=max_frames)
    print(f"Visualization saved to: {output}")


if __name__ == "__main__":
    main()
