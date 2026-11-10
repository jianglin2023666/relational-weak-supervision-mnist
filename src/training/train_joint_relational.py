"""All-in-one joint relation and sparse-label training experiment."""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import SupervisedMNISTDataset, WeaklySupervisedMNISTDataset
from src.models.joint_relational_classifier import JointRelationalClassifier
from src.utils.constants import JOINT_RELATIONAL_TRAINING_DIR
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
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "joint_relational.json"

REQUIRED_CONFIG_FIELDS = (
    'experiment_name',
    'train_size',
    'epochs',
    'seeds',
    'shots_per_class',
    'classification_weight',
    'relation_batch_size',
    'labelled_batch_size',
    'learning_rate',
    'weight_decay',
    'num_workers',
    'deterministic',
)
RUN_FIELDS = (
    'seed',
    'best_epoch',
    'shots_per_class',
    'classification_weight',
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


def make_loaders(datasets, config, seed, selected_indices, use_cuda):
    relation_generator = torch.Generator().manual_seed(seed)
    labelled_generator = torch.Generator().manual_seed(seed + 1)
    common = {
        "num_workers": config["num_workers"],
        "pin_memory": use_cuda,
    }
    labelled_dataset = Subset(datasets["digit_train"], selected_indices)

    return {
        "relation_train": DataLoader(
            datasets["relation_train"],
            batch_size=config["relation_batch_size"],
            shuffle=True,
            generator=relation_generator,
            **common,
        ),
        "labelled_train": DataLoader(
            labelled_dataset,
            batch_size=min(config["labelled_batch_size"], len(labelled_dataset)),
            shuffle=True,
            generator=labelled_generator,
            **common,
        ),
        "relation_val": DataLoader(
            datasets["relation_val"],
            batch_size=config["relation_batch_size"],
            shuffle=False,
            **common,
        ),
        "relation_test": DataLoader(
            datasets["relation_test"],
            batch_size=config["relation_batch_size"],
            shuffle=False,
            **common,
        ),
        "digit_test": DataLoader(
            datasets["digit_test"],
            batch_size=config["relation_batch_size"],
            shuffle=False,
            **common,
        ),
    }


def train_joint_epoch(
    model,
    relation_loader,
    labelled_loader,
    optimizer,
    classification_weight,
    device,
):
    model.train()
    labelled_iterator = iter(labelled_loader)
    total_loss = 0.0
    total_relation_loss = 0.0
    total_classification_loss = 0.0
    relation_correct = relation_total = 0
    class_correct = class_total = 0

    for pairs, relation_labels in tqdm(
        relation_loader, desc="Joint training", leave=False
    ):
        try:
            labelled_images, class_labels = next(labelled_iterator)
        except StopIteration:
            labelled_iterator = iter(labelled_loader)
            labelled_images, class_labels = next(labelled_iterator)

        pairs = pairs.to(device, non_blocking=True)
        relation_labels = relation_labels.to(device, non_blocking=True).float()
        labelled_images = labelled_images.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(image_pairs=pairs, labelled_images=labelled_images)
        relation_probabilities = outputs["relation_probability"]
        if not torch.isfinite(relation_probabilities).all():
            raise RuntimeError("Non-finite relation probability during training.")

        relation_loss = F.binary_cross_entropy(
            relation_probabilities, relation_labels
        )
        classification_loss = F.cross_entropy(
            outputs["class_logits"], class_labels
        )
        loss = relation_loss + classification_weight * classification_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_relation_loss += relation_loss.item()
        total_classification_loss += classification_loss.item()
        relation_correct += (
            (relation_probabilities > 0.5) == relation_labels.bool()
        ).sum().item()
        relation_total += relation_labels.numel()
        class_correct += (
            outputs["class_logits"].argmax(dim=1) == class_labels
        ).sum().item()
        class_total += class_labels.numel()

    batches = len(relation_loader)
    return {
        "loss": total_loss / batches,
        "relation_loss": total_relation_loss / batches,
        "classification_loss": total_classification_loss / batches,
        "relation_accuracy": 100.0 * relation_correct / relation_total,
        "labelled_accuracy": 100.0 * class_correct / class_total,
    }


def evaluate_relations(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = total = 0
    with torch.inference_mode():
        for pairs, labels in tqdm(loader, desc="Relation eval", leave=False):
            pairs = pairs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            probabilities = model.relation_probability(pairs)
            loss = F.binary_cross_entropy(probabilities, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += ((probabilities > 0.5) == labels.bool()).sum().item()
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
            predictions = model.class_logits(images).argmax(dim=1).cpu()
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
    selected_indices = select_few_shot_indices(
        datasets["digit_train"], config["shots_per_class"], seed
    )
    loaders = make_loaders(
        datasets, config, seed, selected_indices, device.type == "cuda"
    )

    seed_dir = experiment_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "best_joint_relational_model.pth"
    history_path = seed_dir / "history.csv"

    model = JointRelationalClassifier().to(device)
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
        train_metrics = train_joint_epoch(
            model,
            loaders["relation_train"],
            loaders["labelled_train"],
            optimizer,
            config["classification_weight"],
            device,
        )
        val_metrics = evaluate_relations(
            model, loaders["relation_val"], device
        )

        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_relation_loss": train_metrics["relation_loss"],
            "train_classification_loss": train_metrics["classification_loss"],
            "train_relation_accuracy": train_metrics["relation_accuracy"],
            "train_labelled_accuracy": train_metrics["labelled_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_relation_accuracy": val_metrics["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - start_time,
        }
        history.append(row)
        save_csv(history, history_path)

        print(
            f"Train total/relation/class loss: {train_metrics['loss']:.4f}/"
            f"{train_metrics['relation_loss']:.4f}/"
            f"{train_metrics['classification_loss']:.4f}"
        )
        print(
            f"Train relation/labelled acc: "
            f"{train_metrics['relation_accuracy']:.2f}%/"
            f"{train_metrics['labelled_accuracy']:.2f}%"
        )
        print(
            f"Val relation loss/acc: {val_metrics['loss']:.4f}/"
            f"{val_metrics['accuracy']:.2f}%"
        )

        # Use only relation validation labels for checkpoint selection. This
        # avoids silently introducing a fully labelled digit validation set.
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "seed": seed,
                    "config": config,
                    "selected_indices": selected_indices,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    relation_test = evaluate_relations(model, loaders["relation_test"], device)
    digit_test = evaluate_digits(model, loaders["digit_test"], device)
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
        "shots_per_class": config["shots_per_class"],
        "classification_weight": config["classification_weight"],
        "selected_indices": selected_indices,
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
    if config["shots_per_class"] not in (1, 3):
        raise ValueError("shots_per_class must be 1 or 3.")
    if config["classification_weight"] < 0:
        raise ValueError("classification_weight must be nonnegative.")
    device = select_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = (
        Path(JOINT_RELATIONAL_TRAINING_DIR)
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
        "checkpoint_selection": "relation_validation_accuracy",
    }
    save_json(metadata, experiment_dir / "metadata.json")

    print(f"Using device: {device}")
    print(f"Experiment directory: {experiment_dir}")
    print(
        f"Sparse labels: {config['shots_per_class']} per class; "
        f"classification weight: {config['classification_weight']}"
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
        description="Run the complete joint relational experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config path (default: {DEFAULT_CONFIG_PATH})",
    )
    arguments = parser.parse_args()
    run_experiment(arguments.config)
