"""Metrics used by baseline evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute mean squared error."""
    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    return float(np.mean((pred - tgt) ** 2))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute mean absolute error."""
    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    return float(np.mean(np.abs(pred - tgt)))


def per_dimension_mse(prediction: np.ndarray, target: np.ndarray) -> list[float]:
    """Compute MSE for each action dimension."""
    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if pred.ndim != 2 or tgt.ndim != 2:
        return []
    return np.mean((pred - tgt) ** 2, axis=0).astype(float).tolist()


def action_smoothness(actions: np.ndarray) -> float:
    """Compute mean L2 distance between adjacent actions."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or len(arr) < 2:
        return 0.0
    finite = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite]
    if len(arr) < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def summarize_prediction_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """Return all core action prediction metrics."""
    return {
        "mse": mse(prediction, target),
        "mae": mae(prediction, target),
        "action_smoothness_pred": action_smoothness(prediction),
        "action_smoothness_target": action_smoothness(target),
        "per_dimension_mse": per_dimension_mse(prediction, target),
    }


def failed_episode_ratio_from_qa(summary_path: str | Path) -> float:
    """Read failed episode ratio from a QA summary JSON file."""
    path = Path(summary_path)
    if not path.exists():
        return 0.0
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return float(summary.get("failed_episode_ratio", 0.0))
