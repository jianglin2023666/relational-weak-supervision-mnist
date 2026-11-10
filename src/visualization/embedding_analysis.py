"""Generate reproducible embedding visualizations for one trained checkpoint."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from data.dataset import SupervisedMNISTDataset
from src.models.base_model import BaseModel
from src.utils.experiment_utils import to_project_relative_path


MODEL_NAMES = {
    "supervised": "Supervised CNN",
    "weakly_supervised": "Weakly Supervised",
    "siamese": "Siamese Network",
    "joint_relational": "Joint Relational Classifier",
}


def select_device(requested_device):
    """Resolve an explicit device or automatically select the best backend."""
    if requested_device != "auto":
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_backbone(checkpoint_path, device):
    """Load a BaseModel from either a full-model or backbone-only checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    backbone = BaseModel()
    expected_keys = set(backbone.state_dict())

    if expected_keys.issubset(state_dict):
        backbone_state = {key: state_dict[key] for key in expected_keys}
    else:
        prefixed_keys = {f"feature_extractor.{key}" for key in expected_keys}
        if not prefixed_keys.issubset(state_dict):
            missing = sorted(prefixed_keys - set(state_dict))[:5]
            raise ValueError(
                "Cannot locate BaseModel weights in checkpoint. "
                f"Example missing keys: {missing}"
            )
        backbone_state = {
            key: state_dict[f"feature_extractor.{key}"]
            for key in expected_keys
        }

    backbone.load_state_dict(backbone_state, strict=True)
    backbone.to(device)
    backbone.eval()
    return backbone, checkpoint


def select_balanced_indices(labels, samples_per_class, seed, num_classes=10):
    """Select the same number of test examples from every digit class."""
    generator = torch.Generator().manual_seed(seed)
    selected = []

    for class_id in range(num_classes):
        candidates = torch.where(labels == class_id)[0]
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(candidates)} samples; "
                f"requested {samples_per_class}."
            )
        order = torch.randperm(len(candidates), generator=generator)
        selected.extend(candidates[order[:samples_per_class]].tolist())

    shuffle_order = torch.randperm(len(selected), generator=generator).tolist()
    return [selected[index] for index in shuffle_order]


def extract_embeddings(backbone, loader, device):
    """Extract BaseModel embeddings and labels from a data loader."""
    embeddings = []
    labels = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            embeddings.append(backbone(images).cpu().numpy())
            labels.append(targets.cpu().numpy())

    return np.concatenate(embeddings), np.concatenate(labels).astype(int)


def create_tsne_plot(
    embeddings,
    labels,
    model_title,
    output_path,
    seed,
    perplexity,
):
    """Create and save a two-dimensional t-SNE embedding plot."""
    if perplexity >= len(embeddings):
        raise ValueError("t-SNE perplexity must be smaller than sample count.")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1_000,
        random_state=seed,
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    figure, axis = plt.subplots(figsize=(9, 7))
    scatter = axis.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=14,
        alpha=0.65,
        linewidths=0,
    )

    for digit in range(10):
        mask = labels == digit
        centroid = embeddings_2d[mask].mean(axis=0)
        axis.text(
            centroid[0],
            centroid[1],
            str(digit),
            fontsize=13,
            weight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8},
        )

    colorbar = figure.colorbar(scatter, ax=axis, ticks=range(10))
    colorbar.set_label("Digit class")
    axis.set_title(f"{model_title}: t-SNE of 128D embeddings")
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return embeddings_2d


def analyze_ordinal_structure(embeddings, labels):
    """Project embeddings to one PCA axis and quantify digit ordering."""
    pca = PCA(n_components=1)
    projection = pca.fit_transform(embeddings).ravel()
    raw_rho, p_value = spearmanr(projection, labels)
    if not np.isfinite(raw_rho):
        raise RuntimeError("Spearman correlation is not finite.")

    orientation_flipped = bool(raw_rho < 0)
    if orientation_flipped:
        projection = -projection

    rho, p_value = spearmanr(projection, labels)
    class_statistics = []
    for digit in range(10):
        values = projection[labels == digit]
        class_statistics.append(
            {
                "digit": digit,
                "count": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    discovered_order = [
        row["digit"]
        for row in sorted(class_statistics, key=lambda row: row["mean"])
    ]
    return {
        "projection": projection,
        "raw_spearman_rho": float(raw_rho),
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_value),
        "orientation_flipped": orientation_flipped,
        "pca_explained_variance_ratio": float(
            pca.explained_variance_ratio_[0]
        ),
        "discovered_order": discovered_order,
        "class_statistics": class_statistics,
    }


def create_violin_plot(labels, ordinal_results, model_title, output_path):
    """Create and save the PCA projection violin plot."""
    projection = ordinal_results["projection"]
    values_by_digit = [projection[labels == digit] for digit in range(10)]

    figure, axis = plt.subplots(figsize=(11, 7))
    parts = axis.violinplot(
        values_by_digit,
        positions=range(10),
        widths=0.7,
        showmeans=True,
        showextrema=True,
    )
    for body, digit in zip(parts["bodies"], range(10)):
        body.set_facecolor(plt.colormaps["tab10"](digit))
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    order_text = " → ".join(
        str(digit) for digit in ordinal_results["discovered_order"]
    )
    axis.text(
        0.5,
        0.97,
        f"Discovered order: {order_text}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff5a8", "alpha": 0.8},
    )
    axis.set_xticks(range(10))
    axis.set_xlabel("Digit class")
    axis.set_ylabel("First principal component value")
    axis.set_title(
        f"{model_title}: ordinal structure "
        f"(Spearman ρ = {ordinal_results['spearman_rho']:.3f})"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_csv(rows, path):
    """Save dictionary rows as UTF-8 CSV."""
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    model_name,
    checkpoint_path,
    output_dir,
    samples_per_class=200,
    batch_size=256,
    seed=42,
    num_workers=0,
    requested_device="auto",
    perplexity=30.0,
):
    """Run the complete embedding analysis for one checkpoint."""
    device = select_device(requested_device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path).resolve()

    print(f"Model: {MODEL_NAMES[model_name]}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    backbone, checkpoint = load_backbone(checkpoint_path, device)
    dataset = SupervisedMNISTDataset(split="test")
    selected_indices = select_balanced_indices(
        dataset.labels,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    embeddings, labels = extract_embeddings(backbone, loader, device)

    embeddings_2d = create_tsne_plot(
        embeddings,
        labels,
        MODEL_NAMES[model_name],
        output_dir / "tsne.png",
        seed,
        perplexity,
    )
    ordinal_results = analyze_ordinal_structure(embeddings, labels)
    create_violin_plot(
        labels,
        ordinal_results,
        MODEL_NAMES[model_name],
        output_dir / "pca_violin.png",
    )
    save_csv(
        ordinal_results["class_statistics"],
        output_dir / "class_projection.csv",
    )
    sample_rows = [
        {
            "dataset_index": selected_indices[index],
            "digit": int(labels[index]),
            "tsne_x": float(embeddings_2d[index, 0]),
            "tsne_y": float(embeddings_2d[index, 1]),
            "pca_projection": float(ordinal_results["projection"][index]),
        }
        for index in range(len(labels))
    ]
    save_csv(sample_rows, output_dir / "embedding_samples.csv")
    np.savez_compressed(
        output_dir / "embeddings.npz",
        embeddings=embeddings,
        labels=labels,
        selected_indices=np.asarray(selected_indices),
        tsne_coordinates=embeddings_2d,
        pca_projection=ordinal_results["projection"],
    )

    checkpoint_seed = checkpoint.get("seed") if isinstance(checkpoint, dict) else None
    metrics = {
        "model": model_name,
        "model_title": MODEL_NAMES[model_name],
        "checkpoint": to_project_relative_path(checkpoint_path),
        "checkpoint_seed": checkpoint_seed,
        "analysis_seed": seed,
        "device": str(device),
        "num_samples": int(len(labels)),
        "samples_per_class": samples_per_class,
        "embedding_dimension": int(embeddings.shape[1]),
        "tsne_perplexity": perplexity,
        "raw_spearman_rho": ordinal_results["raw_spearman_rho"],
        "spearman_rho": ordinal_results["spearman_rho"],
        "spearman_p_value": ordinal_results["spearman_p_value"],
        "orientation_flipped": ordinal_results["orientation_flipped"],
        "pca_explained_variance_ratio": ordinal_results[
            "pca_explained_variance_ratio"
        ],
        "discovered_order": ordinal_results["discovered_order"],
        "class_statistics": ordinal_results["class_statistics"],
    }
    with (output_dir / "embedding_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"Spearman rho: {metrics['spearman_rho']:.4f}")
    print(
        "Discovered order: "
        + " -> ".join(str(value) for value in metrics["discovered_order"])
    )
    print(f"Outputs saved to: {output_dir.resolve()}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate t-SNE, PCA violin, Spearman rho, and discovered-order "
            "analysis for one trained checkpoint."
        )
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    arguments = parser.parse_args()

    if arguments.samples_per_class <= 0:
        parser.error("--samples-per-class must be positive.")
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if arguments.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if arguments.perplexity <= 0:
        parser.error("--perplexity must be positive.")

    run_analysis(
        model_name=arguments.model,
        checkpoint_path=arguments.checkpoint,
        output_dir=arguments.output_dir,
        samples_per_class=arguments.samples_per_class,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        num_workers=arguments.num_workers,
        requested_device=arguments.device,
        perplexity=arguments.perplexity,
    )


if __name__ == "__main__":
    main()
