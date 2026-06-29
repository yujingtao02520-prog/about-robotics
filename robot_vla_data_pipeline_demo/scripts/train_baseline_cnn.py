"""Train the CNN + MLP RGB/state-to-action baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data_schema import load_episodes_pickle
from src.datasets.torch_dataset import FrameActionDataset
from src.models.cnn_policy import CNNPolicy
from src.train_utils import (
    config_path,
    load_config,
    make_data_loaders,
    print_kv_table,
    resolve_path,
    save_checkpoint,
    set_seed,
    train_regression_model,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path.")
    parser.add_argument("--input", default=None, help="Processed episodes pickle path.")
    parser.add_argument("--output", default=None, help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["training"]
    set_seed(int(train_cfg["seed"]))

    input_path = resolve_path(args.input) if args.input else config_path(config, "paths", "processed_episodes")
    output_path = resolve_path(args.output) if args.output else config_path(config, "paths", "cnn_checkpoint")
    episodes = load_episodes_pickle(input_path)
    dataset = FrameActionDataset(
        episodes,
        use_images=True,
        expected_action_dim=int(config["qa"]["expected_action_dim"]),
    )
    summary = dataset.summary()
    print_kv_table((key, value) for key, value in summary.items())

    train_loader, val_loader = make_data_loaders(
        dataset=dataset,
        batch_size=args.batch_size or int(train_cfg["batch_size"]),
        val_ratio=float(train_cfg["val_ratio"]),
        seed=int(train_cfg["seed"]),
        num_workers=int(train_cfg["num_workers"]),
    )
    device = torch.device(str(train_cfg["device"]))
    model = CNNPolicy(state_dim=dataset.state_dim, action_dim=dataset.action_dim)
    result = train_regression_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs or int(train_cfg["epochs"]),
        learning_rate=args.lr or float(train_cfg["learning_rate"]),
        device=device,
        use_images=True,
    )
    checkpoint = save_checkpoint(
        model,
        output_path,
        metadata={
            "model_type": "cnn",
            "use_images": True,
            "state_dim": dataset.state_dim,
            "action_dim": dataset.action_dim,
            "num_samples": len(dataset),
            "final_metrics": result["final"],
        },
    )
    print(f"CNN checkpoint saved to: {checkpoint}")


if __name__ == "__main__":
    main()
