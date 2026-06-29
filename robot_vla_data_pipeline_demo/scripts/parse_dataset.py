"""Parse raw HDF5 data into the unified episode schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generate_mock_data import create_mock_hdf5

from src.data_schema import save_episodes_jsonl, save_episodes_pickle
from src.episode_parser import parse_hdf5_dataset, print_parse_summary
from src.train_utils import config_path, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path.")
    parser.add_argument(
        "--input",
        default=None,
        help="Input HDF5 path. Defaults to paths.mock_hdf5.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output pickle path. Defaults to paths.processed_episodes.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    input_path = resolve_path(args.input) if args.input else config_path(config, "paths", "mock_hdf5")
    output_path = resolve_path(args.output) if args.output else config_path(config, "paths", "processed_episodes")

    if not input_path.exists():
        print(f"Input HDF5 not found: {input_path}")
        print("Generating mock data so the demo can run locally.")
        create_mock_hdf5(input_path, config)

    episodes = parse_hdf5_dataset(input_path)
    save_episodes_pickle(episodes, output_path)
    save_episodes_jsonl(episodes, output_path.with_suffix(".jsonl"))
    print_parse_summary(episodes)
    print(f"Saved processed episodes to: {output_path}")
    print(f"Saved JSONL inspection file to: {output_path.with_suffix('.jsonl')}")


if __name__ == "__main__":
    main()
