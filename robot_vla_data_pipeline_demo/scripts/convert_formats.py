"""Convert unified episodes into simplified LeRobot/ACT/OpenVLA formats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.converters.act_converter import convert_episode_to_act
from src.converters.lerobot_converter import convert_episode_to_lerobot
from src.converters.openvla_converter import convert_episode_to_openvla
from src.data_schema import load_episodes_pickle
from src.train_utils import config_path, load_config, resolve_path


TARGETS = ("lerobot", "act", "openvla")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path.")
    parser.add_argument("--input", default=None, help="Processed episodes pickle path.")
    parser.add_argument(
        "--target",
        default="all",
        choices=("all",) + TARGETS,
        help="Format target to export.",
    )
    parser.add_argument("--output-dir", default=None, help="Converted output root directory.")
    return parser.parse_args()


def selected_targets(target: str) -> Iterable[str]:
    """Expand the all target into concrete converter names."""
    return TARGETS if target == "all" else (target,)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    input_path = resolve_path(args.input) if args.input else config_path(config, "paths", "processed_episodes")
    output_root = resolve_path(args.output_dir) if args.output_dir else config_path(config, "paths", "converted_dir")
    episodes = load_episodes_pickle(input_path)

    counters = {}
    for target in selected_targets(args.target):
        target_dir = output_root / target
        for episode in episodes:
            if target == "lerobot":
                convert_episode_to_lerobot(episode, target_dir)
            elif target == "act":
                convert_episode_to_act(episode, target_dir)
            elif target == "openvla":
                convert_episode_to_openvla(episode, target_dir)
        counters[target] = len(episodes)
        print(f"Exported {len(episodes)} episodes to {target_dir}")

    print(f"Conversion summary: {counters}")


if __name__ == "__main__":
    main()
