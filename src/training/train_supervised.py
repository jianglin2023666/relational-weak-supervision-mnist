"""All-in-one fully supervised MNIST baseline experiment."""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SupervisedMNISTDataset
from src.models.supervised_model import SupervisedModel
from src.utils.constants import SUPERVISED_TRAINING_DIR
from src.utils.experiment_utils import (
    aggregate_results,
    load_config,
    save_confusion_matrix,
    save_csv,
    save_json,
    select_device,
    set_global_seed,
    to_project_relative_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised.json"

REQUIRED_CONFIG_FIELDS = (
    'experiment_name',
    'epochs',
    'seeds',
    'batch_size',
    'learning_rate',
    'weight_decay',
    'num_workers',
    'deterministic',
)
RUN_FIELDS = (
    'seed',
    'best_epoch',
    'best_val_digit_accuracy',
    'digit_test_loss',
    'digit_test_accuracy',
    'training_time_seconds',
    'checkpoint_path',
)
SUMMARY_METRICS = (
    'best_val_digit_accuracy',
    'digit_test_loss',
    'digit_test_accuracy',
    'training_time_seconds',
)


def make_loaders(datasets, config, seed, use_cuda):
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": use_cuda,
    }
    return {
        "train": DataLoader(
            datasets["train"], shuffle=True, generator=generator, **common
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = total = 0
    context = torch.enable_grad() if is_training else torch.no_grad()
    description = "Supervised training" if is_training else "Digit eval"

    with context:
        for images, labels in tqdm(loader, desc=description, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }


def evaluate_with_confusion(model, loader, criterion, device, num_classes=10):
    model.eval()
    total_loss = 0.0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Digit test", leave=False):
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels_device)
            predictions = logits.argmax(dim=1).cpu()
            labels = labels.long().cpu()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            indices = labels * num_classes + predictions
            confusion += torch.bincount(
                indices, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    accuracy = 100.0 * confusion.diag().sum().item() / total
    per_class_accuracy = {}
    for class_id in range(num_classes):
        class_total = confusion[class_id].sum().item()
        per_class_accuracy[str(class_id)] = (
            100.0 * confusion[class_id, class_id].item() / class_total
            if class_total
            else 0.0
        )
    return {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def train_single_seed(config, datasets, device, experiment_dir, seed):
    print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
    set_global_seed(seed, config["deterministic"])
    loaders = make_loaders(datasets, config, seed, device.type == "cuda")
    seed_dir = experiment_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "best_supervised_model.pth"
    history_path = seed_dir / "history.csv"

    model = SupervisedModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    best_val_accuracy = -1.0
    best_epoch = 0
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        train_metrics = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_digit_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_digit_accuracy": val_metrics["accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - start_time,
            }
        )
        save_csv(history, history_path)
        print(
            f"Train digit loss/acc: {train_metrics['loss']:.4f}/"
            f"{train_metrics['accuracy']:.2f}%"
        )
        print(
            f"Val digit loss/acc: {val_metrics['loss']:.4f}/"
            f"{val_metrics['accuracy']:.2f}%"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_with_confusion(
        model, loaders["test"], criterion, device
    )
    training_time = time.perf_counter() - start_time
    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_digit_accuracy": best_val_accuracy,
        "digit_test_loss": test_metrics["loss"],
        "digit_test_accuracy": test_metrics["accuracy"],
        "digit_per_class_accuracy": test_metrics["per_class_accuracy"],
        "training_time_seconds": training_time,
        "checkpoint_path": to_project_relative_path(checkpoint_path),
    }
    save_json(metrics, seed_dir / "metrics.json")
    save_confusion_matrix(
        test_metrics["confusion_matrix"], seed_dir / "digit_confusion_matrix.csv"
    )
    torch.save(
        model.feature_extractor.state_dict(), seed_dir / "best_base_model.pth"
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best digit val acc: {best_val_accuracy:.2f}%")
    print(f"Digit test acc: {test_metrics['accuracy']:.2f}%")

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def run_experiment(config_path=DEFAULT_CONFIG_PATH):
    config_path = Path(config_path).resolve()
    config = load_config(config_path, REQUIRED_CONFIG_FIELDS)
    device = select_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = (
        Path(SUPERVISED_TRAINING_DIR) / f"{config['experiment_name']}_{timestamp}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)
    save_json(config, experiment_dir / "config.json")
    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "config_path": str(config_path),
        "checkpoint_selection": "digit_validation_accuracy",
        "supervision": "full_digit_labels",
    }
    save_json(metadata, experiment_dir / "metadata.json")

    print(f"Using device: {device}")
    print(f"Experiment directory: {experiment_dir}")
    print("Loading datasets once for all seeds...")
    datasets = {
        "train": SupervisedMNISTDataset(split="train"),
        "val": SupervisedMNISTDataset(split="val"),
        "test": SupervisedMNISTDataset(split="test"),
    }
    print(
        f"Fully labelled samples: train={len(datasets['train'])}, "
        f"val={len(datasets['val'])}, test={len(datasets['test'])}"
    )

    run_metrics = []
    for seed in config["seeds"]:
        run_metrics.append(
            train_single_seed(config, datasets, device, experiment_dir, int(seed))
        )
    summary = aggregate_results(
        run_metrics,
        experiment_dir,
        metadata,
        RUN_FIELDS,
        SUMMARY_METRICS,
    )
    digit = summary["metrics"]["digit_test_accuracy"]
    print(f"\n{'=' * 60}\nFinal summary ({len(run_metrics)} runs)\n{'=' * 60}")
    print(f"Digit test acc: {digit['mean']:.2f} ± {digit['std']:.2f}%")
    print(f"Results saved to: {experiment_dir.resolve()}")
    return experiment_dir, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete fully supervised baseline experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config path (default: {DEFAULT_CONFIG_PATH})",
    )
    arguments = parser.parse_args()
    run_experiment(arguments.config)
