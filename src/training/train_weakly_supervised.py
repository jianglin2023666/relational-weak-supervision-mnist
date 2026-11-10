"""All-in-one weakly supervised training, validation, and testing."""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

# Required by CUDA matrix multiplication when deterministic mode is enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SupervisedMNISTDataset, WeaklySupervisedMNISTDataset
from src.models.weakly_supervised_model import WeaklySupervisedModel
from src.utils.constants import WEAKLY_SUPERVISED_TRAINING_DIR
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
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "weakly_supervised.json"

REQUIRED_CONFIG_FIELDS = (
    'experiment_name',
    'train_size',
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
    'best_val_relation_accuracy',
    'relation_test_loss',
    'relation_test_accuracy',
    'digit_test_accuracy',
    'training_time_seconds',
    'checkpoint_path',
)
SUMMARY_METRICS = (
    'best_val_relation_accuracy',
    'relation_test_loss',
    'relation_test_accuracy',
    'digit_test_accuracy',
    'training_time_seconds',
)


def make_loaders(datasets, config, seed, use_cuda):
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": use_cuda,
    }
    return {
        "train": DataLoader(
            datasets["relation_train"],
            shuffle=True,
            generator=generator,
            **common,
        ),
        "relation_val": DataLoader(
            datasets["relation_val"], shuffle=False, **common
        ),
        "relation_test": DataLoader(
            datasets["relation_test"], shuffle=False, **common
        ),
        "digit_test": DataLoader(datasets["digit_test"], shuffle=False, **common),
    }


def run_relation_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0
    description = "Training" if is_training else "Evaluating"

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch_index, (data, target) in enumerate(
            tqdm(loader, desc=description, leave=False)
        ):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(data)
            if not torch.isfinite(outputs).all():
                raise RuntimeError(
                    f"Non-finite relation probability at batch {batch_index}."
                )

            loss = criterion(outputs, target.float())
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            correct += ((outputs > 0.5) == target.bool()).sum().item()
            total += batch_size

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }


def evaluate_digit_classification(model, loader, device, num_classes=10):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Digit test", leave=False):
            images = images.to(device, non_blocking=True)
            predictions = model.predict(images).argmax(dim=1).cpu()
            labels = labels.long().cpu()
            indices = labels * num_classes + predictions
            confusion += torch.bincount(
                indices, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    class_totals = confusion.sum(dim=1)
    per_class_accuracy = {}
    for class_id in range(num_classes):
        class_total = class_totals[class_id].item()
        class_correct = confusion[class_id, class_id].item()
        per_class_accuracy[str(class_id)] = (
            100.0 * class_correct / class_total if class_total else 0.0
        )

    return {
        "accuracy": 100.0 * correct / total,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def train_single_seed(config, datasets, device, experiment_dir, seed):
    print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
    set_global_seed(seed, config["deterministic"])
    loaders = make_loaders(datasets, config, seed, device.type == "cuda")

    seed_dir = experiment_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "best_weakly_supervised_model.pth"
    history_path = seed_dir / "history.csv"

    model = WeaklySupervisedModel().to(device)
    criterion = nn.BCELoss()
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
        train_metrics = run_relation_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        val_metrics = run_relation_epoch(
            model, loaders["relation_val"], criterion, device
        )

        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_relation_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_relation_accuracy": val_metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        history.append(row)
        save_csv(history, history_path)

        print(
            f"Train loss {train_metrics['loss']:.4f}, "
            f"relation acc {train_metrics['accuracy']:.2f}%"
        )
        print(
            f"Val loss {val_metrics['loss']:.4f}, "
            f"relation acc {val_metrics['accuracy']:.2f}%"
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
    relation_test = run_relation_epoch(
        model, loaders["relation_test"], criterion, device
    )
    digit_test = evaluate_digit_classification(model, loaders["digit_test"], device)
    training_time = time.perf_counter() - start_time

    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_relation_accuracy": best_val_accuracy,
        "relation_test_loss": relation_test["loss"],
        "relation_test_accuracy": relation_test["accuracy"],
        "digit_test_accuracy": digit_test["accuracy"],
        "digit_per_class_accuracy": digit_test["per_class_accuracy"],
        "training_time_seconds": training_time,
        "checkpoint_path": to_project_relative_path(checkpoint_path),
    }
    save_json(metrics, seed_dir / "metrics.json")
    save_confusion_matrix(
        digit_test["confusion_matrix"], seed_dir / "digit_confusion_matrix.csv"
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best relation val acc: {best_val_accuracy:.2f}%")
    print(f"Relation test acc: {relation_test['accuracy']:.2f}%")
    print(f"Digit test acc: {digit_test['accuracy']:.2f}%")

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
        Path(WEAKLY_SUPERVISED_TRAINING_DIR)
        / f"{config['experiment_name']}_{timestamp}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)
    save_json(config, experiment_dir / "config.json")

    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "config_path": str(config_path),
    }
    save_json(metadata, experiment_dir / "metadata.json")

    print(f"Using device: {device}")
    print(f"Experiment directory: {experiment_dir}")
    print("Loading datasets once for all seeds...")
    datasets = {
        "relation_train": WeaklySupervisedMNISTDataset(
            split="train", size=config["train_size"]
        ),
        "relation_val": WeaklySupervisedMNISTDataset(split="val"),
        "relation_test": WeaklySupervisedMNISTDataset(split="test"),
        "digit_test": SupervisedMNISTDataset(split="test"),
    }

    run_metrics = []
    for seed in config["seeds"]:
        run_metrics.append(
            train_single_seed(
                config, datasets, device, experiment_dir, int(seed)
            )
        )

    summary = aggregate_results(
        run_metrics,
        experiment_dir,
        metadata,
        RUN_FIELDS,
        SUMMARY_METRICS,
    )
    relation = summary["metrics"]["relation_test_accuracy"]
    digit = summary["metrics"]["digit_test_accuracy"]
    print(f"\n{'=' * 60}\nFinal summary ({len(run_metrics)} runs)\n{'=' * 60}")
    print(f"Relation test acc: {relation['mean']:.2f} ± {relation['std']:.2f}%")
    print(f"Digit test acc: {digit['mean']:.2f} ± {digit['std']:.2f}%")
    print(f"Results saved to: {experiment_dir.resolve()}")
    return experiment_dir, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete weakly supervised experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config path (default: {DEFAULT_CONFIG_PATH})",
    )
    arguments = parser.parse_args()
    run_experiment(arguments.config)
