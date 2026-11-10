"""Shared utilities for reproducible training experiments."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_project_relative_path(path):
    """Return a portable POSIX-style path relative to the project root."""
    resolved_path = Path(path).resolve()
    try:
        relative_path = resolved_path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(
            f"Path is outside the project directory: {resolved_path}"
        ) from error
    return relative_path.as_posix()


def load_config(config_path, required_fields):
    """Load a JSON config and check its common required fields."""
    with open(config_path, "r", encoding="utf-8-sig") as file:
        config = json.load(file)

    missing = set(required_fields) - set(config)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    if "seeds" in required_fields and not config["seeds"]:
        raise ValueError("Config field 'seeds' must contain at least one seed.")
    return config


def save_json(data, path):
    """Write JSON using UTF-8 and a human-readable layout."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_csv(rows, path, fieldnames=None):
    """Write a list of dictionaries to a UTF-8 CSV file."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    path = Path(path)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def set_global_seed(seed, deterministic):
    """Seed Python, NumPy, and PyTorch and configure deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def select_device():
    """Select CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_few_shot_indices(dataset, shots_per_class, seed, num_classes=10):
    """Select exactly K reproducible examples from every class."""
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for class_id in range(num_classes):
        candidates = torch.where(dataset.labels == class_id)[0]
        if len(candidates) < shots_per_class:
            raise ValueError(
                f"Class {class_id} has fewer than {shots_per_class} samples."
            )
        order = torch.randperm(len(candidates), generator=generator)
        selected.extend(candidates[order[:shots_per_class]].tolist())
    return selected


def save_confusion_matrix(matrix, path):
    """Save a confusion matrix with explicit true/predicted class labels."""
    rows = []
    for true_label, values in enumerate(matrix):
        row = {"true_label": true_label}
        row.update({f"pred_{index}": value for index, value in enumerate(values)})
        rows.append(row)
    save_csv(rows, path)


def aggregate_results(
    run_metrics,
    experiment_dir,
    metadata,
    run_fields,
    summary_metrics,
):
    """Save per-seed results and mean/std/min/max summary statistics."""
    experiment_dir = Path(experiment_dir)
    save_csv(run_metrics, experiment_dir / "runs.csv", fieldnames=run_fields)

    summary_rows = []
    summary_json = {
        "num_runs": len(run_metrics),
        "metadata": metadata,
        "metrics": {},
    }
    for metric_name in summary_metrics:
        values = np.asarray(
            [run[metric_name] for run in run_metrics], dtype=float
        )
        stats = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
        summary_rows.append({"metric": metric_name, **stats})
        summary_json["metrics"][metric_name] = stats

    save_csv(summary_rows, experiment_dir / "summary.csv")
    save_json(summary_json, experiment_dir / "summary.json")
    return summary_json
