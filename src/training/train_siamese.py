"""All-in-one Siamese relation pretraining and few-shot classification."""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import SupervisedMNISTDataset, WeaklySupervisedMNISTDataset
from src.models.siamese_model import SiameseModel
from src.models.supervised_model import SupervisedModel
from src.utils.constants import SIAMESE_TRAINING_DIR
from src.utils.experiment_utils import (
    aggregate_results,
    load_config,
    save_confusion_matrix,
    save_csv,
    save_json,
    select_device,
    select_few_shot_indices,
    set_global_seed,
    to_project_relative_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "siamese.json"

REQUIRED_CONFIG_FIELDS = (
    'experiment_name',
    'train_size',
    'relation_epochs',
    'few_shot_epochs',
    'seeds',
    'shots_per_class',
    'relation_batch_size',
    'few_shot_batch_size',
    'relation_learning_rate',
    'few_shot_learning_rate',
    'weight_decay',
    'num_workers',
    'deterministic',
)
RUN_FIELDS = (
    'seed',
    'best_epoch',
    'shots_per_class',
    'few_shot_epochs',
    'classification_update_steps',
    'best_val_relation_accuracy',
    'relation_test_loss',
    'relation_test_accuracy',
    'digit_test_accuracy',
    'training_time_seconds',
    'relation_checkpoint_path',
    'checkpoint_path',
)
SUMMARY_METRICS = (
    'best_val_relation_accuracy',
    'relation_test_loss',
    'relation_test_accuracy',
    'digit_test_accuracy',
    'training_time_seconds',
)


def make_relation_loaders(datasets, config, seed, use_cuda):
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": config["relation_batch_size"],
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
        "val": DataLoader(datasets["relation_val"], shuffle=False, **common),
        "test": DataLoader(datasets["relation_test"], shuffle=False, **common),
    }


def run_relation_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = total = 0
    context = torch.enable_grad() if is_training else torch.no_grad()
    description = "Relation training" if is_training else "Relation eval"

    with context:
        for pairs, labels in tqdm(loader, desc=description, leave=False):
            pairs = pairs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            probabilities = model(pairs)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Non-finite Siamese relation probability.")
            loss = criterion(probabilities, labels)
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += ((probabilities > 0.5) == labels.bool()).sum().item()
            total += batch_size

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }


def train_few_shot_epoch(model, loader, criterion, optimizer, device):
    model.train()
    # The backbone is frozen, but its Dropout must also remain disabled.
    model.feature_extractor.eval()
    total_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
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


def evaluate_digits(model, loader, device, num_classes=10):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Digit test", leave=False):
            images = images.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1).cpu()
            labels = labels.long().cpu()
            indices = labels * num_classes + predictions
            confusion += torch.bincount(
                indices, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    total = confusion.sum().item()
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
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def train_single_seed(config, datasets, device, experiment_dir, seed):
    print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
    set_global_seed(seed, config["deterministic"])
    relation_loaders = make_relation_loaders(
        datasets, config, seed, device.type == "cuda"
    )
    selected_indices = select_few_shot_indices(
        datasets["digit_train"], config["shots_per_class"], seed
    )

    seed_dir = experiment_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    relation_checkpoint_path = seed_dir / "best_siamese_relation_model.pth"
    classifier_checkpoint_path = seed_dir / "final_few_shot_classifier.pth"
    start_time = time.perf_counter()

    relation_model = SiameseModel().to(device)
    relation_criterion = nn.BCELoss()
    relation_optimizer = optim.Adam(
        relation_model.parameters(),
        lr=config["relation_learning_rate"],
        weight_decay=config["weight_decay"],
    )
    best_val_accuracy = -1.0
    best_relation_epoch = 0
    relation_history = []

    for epoch in range(1, config["relation_epochs"] + 1):
        print(f"\nRelation epoch {epoch}/{config['relation_epochs']}")
        train_metrics = run_relation_epoch(
            relation_model,
            relation_loaders["train"],
            relation_criterion,
            device,
            relation_optimizer,
        )
        val_metrics = run_relation_epoch(
            relation_model, relation_loaders["val"], relation_criterion, device
        )
        relation_history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_relation_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_relation_accuracy": val_metrics["accuracy"],
                "learning_rate": relation_optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - start_time,
            }
        )
        save_csv(relation_history, seed_dir / "relation_history.csv")
        print(
            f"Train relation loss/acc: {train_metrics['loss']:.4f}/"
            f"{train_metrics['accuracy']:.2f}%"
        )
        print(
            f"Val relation loss/acc: {val_metrics['loss']:.4f}/"
            f"{val_metrics['accuracy']:.2f}%"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_relation_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "config": config,
                    "model_state_dict": relation_model.state_dict(),
                    "optimizer_state_dict": relation_optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                },
                relation_checkpoint_path,
            )

    relation_checkpoint = torch.load(
        relation_checkpoint_path, map_location=device, weights_only=True
    )
    relation_model.load_state_dict(relation_checkpoint["model_state_dict"])
    relation_test = run_relation_epoch(
        relation_model, relation_loaders["test"], relation_criterion, device
    )

    # Stage 2: transfer and freeze only the best relation backbone.
    classifier = SupervisedModel().to(device)
    classifier.feature_extractor.load_state_dict(
        relation_model.feature_extractor.state_dict()
    )
    for parameter in classifier.feature_extractor.parameters():
        parameter.requires_grad = False

    few_shot_generator = torch.Generator().manual_seed(seed + 1)
    few_shot_dataset = Subset(datasets["digit_train"], selected_indices)
    few_shot_loader = DataLoader(
        few_shot_dataset,
        batch_size=min(config["few_shot_batch_size"], len(few_shot_dataset)),
        shuffle=True,
        generator=few_shot_generator,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    digit_test_loader = DataLoader(
        datasets["digit_test"],
        batch_size=config["relation_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    classification_criterion = nn.CrossEntropyLoss()
    classifier_optimizer = optim.Adam(
        classifier.classifier.parameters(),
        lr=config["few_shot_learning_rate"],
        weight_decay=config["weight_decay"],
    )
    few_shot_history = []

    for epoch in range(1, config["few_shot_epochs"] + 1):
        metrics = train_few_shot_epoch(
            classifier,
            few_shot_loader,
            classification_criterion,
            classifier_optimizer,
            device,
        )
        few_shot_history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": metrics["loss"],
                "train_accuracy": metrics["accuracy"],
                "learning_rate": classifier_optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - start_time,
            }
        )
        save_csv(few_shot_history, seed_dir / "few_shot_history.csv")
        if epoch == 1 or epoch % 10 == 0 or epoch == config["few_shot_epochs"]:
            print(
                f"Few-shot epoch {epoch}/{config['few_shot_epochs']}: "
                f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.2f}%"
            )

    # No fully labelled digit validation set is used. The fixed final epoch is
    # evaluated once on the digit test set.
    torch.save(
        {
            "epoch": config["few_shot_epochs"],
            "seed": seed,
            "config": config,
            "selected_indices": selected_indices,
            "relation_checkpoint": to_project_relative_path(
                relation_checkpoint_path
            ),
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": classifier_optimizer.state_dict(),
        },
        classifier_checkpoint_path,
    )
    digit_test = evaluate_digits(classifier, digit_test_loader, device)
    training_time = time.perf_counter() - start_time

    metrics = {
        "seed": seed,
        "best_epoch": best_relation_epoch,
        "best_relation_epoch": best_relation_epoch,
        "few_shot_epochs": config["few_shot_epochs"],
        "best_val_relation_accuracy": best_val_accuracy,
        "relation_test_loss": relation_test["loss"],
        "relation_test_accuracy": relation_test["accuracy"],
        "digit_test_accuracy": digit_test["accuracy"],
        "digit_per_class_accuracy": digit_test["per_class_accuracy"],
        "training_time_seconds": training_time,
        "checkpoint_path": to_project_relative_path(
            classifier_checkpoint_path
        ),
        "relation_checkpoint_path": to_project_relative_path(
            relation_checkpoint_path
        ),
        "shots_per_class": config["shots_per_class"],
        "selected_indices": selected_indices,
        "classification_update_steps": (
            config["few_shot_epochs"] * len(few_shot_loader)
        ),
    }
    save_json(metrics, seed_dir / "metrics.json")
    save_confusion_matrix(
        digit_test["confusion_matrix"], seed_dir / "digit_confusion_matrix.csv"
    )

    print(f"Best relation epoch: {best_relation_epoch}")
    print(f"Best relation val acc: {best_val_accuracy:.2f}%")
    print(f"Relation test acc: {relation_test['accuracy']:.2f}%")
    print(f"Digit test acc: {digit_test['accuracy']:.2f}%")

    del relation_model, classifier, relation_optimizer, classifier_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def run_experiment(config_path=DEFAULT_CONFIG_PATH):
    config_path = Path(config_path).resolve()
    config = load_config(config_path, REQUIRED_CONFIG_FIELDS)
    if config["shots_per_class"] not in (1, 3):
        raise ValueError("shots_per_class must be 1 or 3.")
    device = select_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = (
        Path(SIAMESE_TRAINING_DIR) / f"{config['experiment_name']}_{timestamp}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)
    save_json(config, experiment_dir / "config.json")

    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "config_path": str(config_path),
        "relation_checkpoint_selection": "relation_validation_accuracy",
        "classifier_checkpoint_selection": "fixed_final_epoch",
        "backbone_frozen_during_few_shot": True,
    }
    save_json(metadata, experiment_dir / "metadata.json")

    print(f"Using device: {device}")
    print(f"Experiment directory: {experiment_dir}")
    print(
        f"Few-shot labels: {config['shots_per_class']} per class; "
        f"classifier epochs: {config['few_shot_epochs']}"
    )
    print("Loading datasets once for all seeds...")
    datasets = {
        "relation_train": WeaklySupervisedMNISTDataset(
            split="train", size=config["train_size"]
        ),
        "relation_val": WeaklySupervisedMNISTDataset(split="val"),
        "relation_test": WeaklySupervisedMNISTDataset(split="test"),
        "digit_train": SupervisedMNISTDataset(split="train"),
        "digit_test": SupervisedMNISTDataset(split="test"),
    }

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
    relation = summary["metrics"]["relation_test_accuracy"]
    digit = summary["metrics"]["digit_test_accuracy"]
    print(f"\n{'=' * 60}\nFinal summary ({len(run_metrics)} runs)\n{'=' * 60}")
    print(f"Relation test acc: {relation['mean']:.2f} ± {relation['std']:.2f}%")
    print(f"Digit test acc: {digit['mean']:.2f} ± {digit['std']:.2f}%")
    print(f"Results saved to: {experiment_dir.resolve()}")
    return experiment_dir, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete Siamese plus few-shot experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config path (default: {DEFAULT_CONFIG_PATH})",
    )
    arguments = parser.parse_args()
    run_experiment(arguments.config)
