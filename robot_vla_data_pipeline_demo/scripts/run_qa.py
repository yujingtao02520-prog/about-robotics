"""Run data quality checks for processed VLA episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_schema import load_episodes_pickle
from src.qa_rules import run_quality_checks, save_qa_reports
from src.train_utils import config_path, load_config, resolve_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path.")
    parser.add_argument("--input", default=None, help="Processed episodes pickle path.")
    parser.add_argument("--csv-output", default=None, help="QA CSV report path.")
    parser.add_argument("--json-output", default=None, help="QA JSON summary path.")
    parser.add_argument("--action-spike-threshold", type=float, default=None)
    parser.add_argument("--min-len", type=int, default=None)
    parser.add_argument("--expected-action-dim", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    qa_cfg = config["qa"]
    input_path = resolve_path(args.input) if args.input else config_path(config, "paths", "processed_episodes")
    csv_path = resolve_path(args.csv_output) if args.csv_output else config_path(config, "paths", "qa_report_csv")
    json_path = resolve_path(args.json_output) if args.json_output else config_path(config, "paths", "qa_summary_json")

    episodes = load_episodes_pickle(input_path)
    report, summary = run_quality_checks(
        episodes,
        action_spike_threshold=args.action_spike_threshold
        if args.action_spike_threshold is not None
        else float(qa_cfg["action_spike_threshold"]),
        min_episode_len=args.min_len if args.min_len is not None else int(qa_cfg["min_episode_len"]),
        expected_action_dim=args.expected_action_dim
        if args.expected_action_dim is not None
        else int(qa_cfg["expected_action_dim"]),
        empty_rgb_pixel_sum_threshold=float(qa_cfg["empty_rgb_pixel_sum_threshold"]),
    )
    save_qa_reports(report, summary, csv_path, json_path)
    print(report[["episode_id", "status", "issue_types"]].to_string(index=False))
    print(f"QA CSV report saved to: {csv_path}")
    print(f"QA JSON summary saved to: {json_path}")
    print(f"Failed episodes: {summary['failed_episodes']} / {summary['total_episodes']}")
    print(f"Issue counts: {summary['issue_counts']}")


if __name__ == "__main__":
    main()
