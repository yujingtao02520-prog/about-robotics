"""Evaluate saved baseline checkpoints on the processed dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_schema import load_episodes_pickle
from src.datasets.torch_dataset import FrameActionDataset
from src.metrics import failed_episode_ratio_from_qa, summarize_prediction_metrics
from src.models.cnn_policy import CNNPolicy
from src.models.mlp_policy import MLPPolicy
from src.train_utils import config_path, load_checkpoint, load_config, resolve_path, evaluate_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path.")
    parser.add_argument("--input", default=None, help="Processed episodes pickle path.")
    parser.add_argument("--mlp-checkpoint", default=None, help="MLP checkpoint path.")
    parser.add_argument("--cnn-checkpoint", default=None, help="CNN checkpoint path.")
    parser.add_argument("--qa-summary", default=None, help="QA summary JSON path.")
    parser.add_argument("--output", default=None, help="Evaluation CSV output path.")
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint_path: Path,
    episodes,
    batch_size: int,
    failed_ratio: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate one saved checkpoint and return a CSV-ready row."""
    checkpoint = load_checkpoint(checkpoint_path, device)
    metadata = checkpoint.get("metadata", {})
    model_type = metadata.get("model_type", "mlp")
    use_images = bool(metadata.get("use_images", model_type == "cnn"))
    state_dim = int(metadata["state_dim"])
    action_dim = int(metadata["action_dim"])

    dataset = FrameActionDataset(
        episodes,
        use_images=use_images,
        expected_state_dim=state_dim,
        expected_action_dim=action_dim,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if model_type == "cnn":
        model = CNNPolicy(state_dim=state_dim, action_dim=action_dim)
    else:
        model = MLPPolicy(state_dim=state_dim, action_dim=action_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    eval_result = evaluate_model(
        model, loader, device=device, use_images=use_images, collect_predictions=True
    )
    metrics = summarize_prediction_metrics(
        eval_result["predictions"], eval_result["targets"]
    )
    return {
        "baseline": model_type,
        "checkpoint": str(checkpoint_path),
        "num_samples": len(dataset),
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "action_smoothness_pred": metrics["action_smoothness_pred"],
        "action_smoothness_target": metrics["action_smoothness_target"],
        "per_dimension_mse": json.dumps(metrics["per_dimension_mse"]),
        "failed_episode_ratio_from_qa": failed_ratio,
    }


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    input_path = resolve_path(args.input) if args.input else config_path(config, "paths", "processed_episodes")
    output_path = resolve_path(args.output) if args.output else config_path(config, "paths", "baseline_eval_csv")
    qa_summary = resolve_path(args.qa_summary) if args.qa_summary else config_path(config, "paths", "qa_summary_json")
    mlp_checkpoint = resolve_path(args.mlp_checkpoint) if args.mlp_checkpoint else config_path(config, "paths", "mlp_checkpoint")
    cnn_checkpoint = resolve_path(args.cnn_checkpoint) if args.cnn_checkpoint else config_path(config, "paths", "cnn_checkpoint")

    episodes = load_episodes_pickle(input_path)
    failed_ratio = failed_episode_ratio_from_qa(qa_summary)
    device = torch.device(str(config["training"]["device"]))
    rows: List[Dict[str, Any]] = []
    for checkpoint_path in [mlp_checkpoint, cnn_checkpoint]:
        if not checkpoint_path.exists():
            print(f"Skipping missing checkpoint: {checkpoint_path}")
            continue
        rows.append(
            evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                episodes=episodes,
                batch_size=int(config["evaluation"]["batch_size"]),
                failed_ratio=failed_ratio,
                device=device,
            )
        )

    if not rows:
        raise FileNotFoundError("No baseline checkpoints found to evaluate.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(df[["baseline", "num_samples", "mse", "mae", "failed_episode_ratio_from_qa"]].to_string(index=False))
    print(f"Baseline evaluation saved to: {output_path}")


if __name__ == "__main__":
    main()
