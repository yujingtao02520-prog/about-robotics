"""Data quality rules for unified VLA episodes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_schema import Episode


def language_missing_check(episode: Episode) -> Tuple[bool, str]:
    """Check whether the language instruction is non-empty."""
    text = str(episode.get("language_instruction", "")).strip()
    return bool(text), "language_instruction is empty"


def action_spike_check(episode: Episode, threshold: float = 3.0) -> Tuple[bool, str]:
    """Check adjacent action L2 jumps against a threshold."""
    actions = np.asarray(episode.get("actions"))
    if actions.ndim != 2 or len(actions) < 2 or not np.all(np.isfinite(actions)):
        return True, ""
    diffs = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    max_diff = float(np.max(diffs)) if len(diffs) else 0.0
    return max_diff <= threshold, f"action spike detected: max_l2_diff={max_diff:.3f}"


def empty_rgb_check(
    episode: Episode, pixel_sum_threshold: float = 0.0
) -> Tuple[bool, str]:
    """Check whether RGB frames are missing or contain all-zero frames."""
    rgb = np.asarray(episode.get("observations", {}).get("rgb"))
    if rgb.size == 0 or rgb.ndim != 4:
        return False, f"RGB is missing or has invalid shape {rgb.shape}"
    frame_sums = rgb.reshape(rgb.shape[0], -1).sum(axis=1)
    empty_indices = np.where(frame_sums <= pixel_sum_threshold)[0]
    if len(empty_indices) > 0:
        first = int(empty_indices[0])
        return False, f"empty/all-zero RGB frame detected at index {first}"
    return True, ""


def short_episode_check(episode: Episode, min_len: int = 10) -> Tuple[bool, str]:
    """Check whether an episode has enough time steps."""
    length = int(episode.get("num_steps", 0))
    return length >= min_len, f"episode too short: num_steps={length}, min_len={min_len}"


def nan_check(episode: Episode) -> Tuple[bool, str]:
    """Check state and action arrays for NaN or Inf values."""
    state = np.asarray(episode.get("observations", {}).get("state"))
    actions = np.asarray(episode.get("actions"))
    bad_parts = []
    if state.size and not np.all(np.isfinite(state)):
        bad_parts.append("state")
    if actions.size and not np.all(np.isfinite(actions)):
        bad_parts.append("actions")
    if bad_parts:
        return False, "non-finite values found in " + ", ".join(bad_parts)
    return True, ""


def action_dim_check(episode: Episode, expected_dim: int | None = None) -> Tuple[bool, str]:
    """Check that action vectors have the expected dimensionality."""
    actions = np.asarray(episode.get("actions"))
    if actions.ndim != 2:
        return False, f"actions must be 2D, got shape {actions.shape}"
    if expected_dim is not None and actions.shape[1] != expected_dim:
        return False, f"action_dim={actions.shape[1]}, expected={expected_dim}"
    return True, ""


def run_quality_checks(
    episodes: List[Episode],
    action_spike_threshold: float = 3.0,
    min_episode_len: int = 10,
    expected_action_dim: int | None = 7,
    empty_rgb_pixel_sum_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run all QA checks and return per-episode rows plus a summary."""
    rows = []
    issue_counter: Counter[str] = Counter()

    for episode in episodes:
        checks = [
            ("language_missing", language_missing_check(episode)),
            (
                "action_spike",
                action_spike_check(episode, threshold=action_spike_threshold),
            ),
            (
                "empty_rgb",
                empty_rgb_check(
                    episode, pixel_sum_threshold=empty_rgb_pixel_sum_threshold
                ),
            ),
            ("short_episode", short_episode_check(episode, min_len=min_episode_len)),
            ("nan_or_inf", nan_check(episode)),
            ("action_dim", action_dim_check(episode, expected_dim=expected_action_dim)),
        ]
        failed = [(name, message) for name, (ok, message) in checks if not ok]
        for name, _ in failed:
            issue_counter[name] += 1

        rows.append(
            {
                "episode_id": episode["episode_id"],
                "status": "pass" if not failed else "fail",
                "fail_reasons": "; ".join(message for _, message in failed),
                "issue_types": ",".join(name for name, _ in failed),
                "num_steps": int(episode["num_steps"]),
                "action_dim": _action_dim(episode),
                "source_format": episode.get("metadata", {}).get("source_format", ""),
            }
        )

    report = pd.DataFrame(rows)
    total = len(report)
    failed_count = int((report["status"] == "fail").sum()) if total else 0
    summary = {
        "total_episodes": total,
        "passed_episodes": total - failed_count,
        "failed_episodes": failed_count,
        "failed_episode_ratio": failed_count / total if total else 0.0,
        "issue_counts": dict(issue_counter),
    }
    return report, summary


def save_qa_reports(
    report: pd.DataFrame,
    summary: Dict[str, Any],
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    """Write QA reports to CSV and JSON."""
    csv_output = Path(csv_path)
    json_output = Path(json_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(csv_output, index=False)
    with json_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _action_dim(episode: Episode) -> int | None:
    """Return action dimensionality when available."""
    actions = np.asarray(episode.get("actions"))
    return int(actions.shape[1]) if actions.ndim == 2 else None
