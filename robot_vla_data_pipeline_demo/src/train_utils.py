"""Shared configuration, training, and checkpoint utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "config/default.yaml") -> Dict[str, Any]:
    """Load the YAML configuration file relative to the project root."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a config path against the project root unless already absolute."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_path(config: Dict[str, Any], *keys: str) -> Path:
    """Read a nested path value from config and resolve it to a Path."""
    node: Any = config
    for key in keys:
        node = node[key]
    return resolve_path(node)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a file path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data_loaders(
    dataset: Dataset,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Split a dataset into train/validation loaders."""
    if len(dataset) < 2:
        raise ValueError("Need at least two samples to create train/val splits.")
    val_size = max(1, int(round(len(dataset) * val_ratio)))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def batch_forward(model: nn.Module, batch: Dict[str, torch.Tensor], use_images: bool) -> torch.Tensor:
    """Run the model with the expected input modality."""
    state = batch["state"]
    if use_images:
        return model(batch["image"], state)
    return model(state)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_images: bool,
    collect_predictions: bool = False,
) -> Dict[str, Any]:
    """Evaluate a policy on a loader and optionally return predictions."""
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_abs = 0.0
    total_values = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = batch_forward(model, batch, use_images)
            target = batch["action"]
            total_loss += float(criterion(pred, target).item())
            total_abs += float(torch.sum(torch.abs(pred - target)).item())
            total_values += int(target.numel())
            if collect_predictions:
                predictions.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())

    mse = total_loss / max(total_values, 1)
    mae = total_abs / max(total_values, 1)
    result: Dict[str, Any] = {"mse": mse, "mae": mae, "loss": mse}
    if collect_predictions:
        result["predictions"] = np.concatenate(predictions, axis=0) if predictions else np.empty((0,))
        result["targets"] = np.concatenate(targets, axis=0) if targets else np.empty((0,))
    return result


def train_regression_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    use_images: bool,
) -> Dict[str, Any]:
    """Train an action prediction model with MSE loss."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pred = batch_forward(model, batch, use_images)
            loss = criterion(pred, batch["action"])
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1

        train_loss = running / max(batches, 1)
        val_metrics = evaluate_model(model, val_loader, device, use_images)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "action_mse": val_metrics["mse"],
            "action_mae": val_metrics["mae"],
        }
        history.append(epoch_metrics)
        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_mse={val_metrics['mse']:.6f} | "
            f"val_mae={val_metrics['mae']:.6f}"
        )

    return {"history": history, "final": history[-1] if history else {}}


def save_checkpoint(
    model: nn.Module,
    output_path: str | Path,
    metadata: Dict[str, Any],
) -> Path:
    """Save model weights and metadata to a torch checkpoint."""
    path = ensure_parent(output_path)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)
    return path


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    """Load a torch checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def print_kv_table(rows: Iterable[Tuple[str, Any]]) -> None:
    """Print simple key-value rows for demo-friendly console output."""
    for key, value in rows:
        print(f"{key:24s}: {value}")
