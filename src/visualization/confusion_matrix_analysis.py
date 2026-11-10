"""Plot count and row-normalized confusion matrices from a training result CSV."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_confusion_matrix(csv_path):
    """Load the CSV format produced by the training scripts."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Confusion matrix CSV does not exist: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        expected_columns = ["true_label"] + [f"pred_{i}" for i in range(10)]
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"Unexpected CSV columns: {reader.fieldnames}. "
                f"Expected: {expected_columns}"
            )

        rows = list(reader)

    if len(rows) != 10:
        raise ValueError(f"Expected 10 true-label rows, found {len(rows)}.")

    rows.sort(key=lambda row: int(row["true_label"]))
    labels = [int(row["true_label"]) for row in rows]
    if labels != list(range(10)):
        raise ValueError(f"Expected true labels 0-9, found: {labels}")

    matrix = np.array(
        [[int(row[f"pred_{i}"]) for i in range(10)] for row in rows],
        dtype=np.int64,
    )
    return matrix


def plot_matrix(matrix, output_path, title, normalized=False):
    """Save one confusion matrix heatmap with cell annotations."""
    if normalized:
        row_totals = matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        vmax = 1.0
    else:
        display_matrix = matrix
        vmax = None

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(display_matrix, cmap="Blues", vmin=0, vmax=vmax)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion" if normalized else "Sample count")

    ax.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xlabel="Predicted digit",
        ylabel="True digit",
        title=title,
    )

    threshold = display_matrix.max() / 2 if display_matrix.size else 0
    for row in range(10):
        for column in range(10):
            value = display_matrix[row, column]
            label = f"{value:.1%}" if normalized else str(int(value))
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot confusion matrices from digit_confusion_matrix.csv."
    )
    parser.add_argument("--input", required=True, help="Path to the confusion matrix CSV.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for PNG figures and metrics JSON.",
    )
    parser.add_argument(
        "--title",
        default="Digit Classification Confusion Matrix",
        help="Base title shown above the figures.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_confusion_matrix(args.input)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = correct / total if total else 0.0

    plot_matrix(
        matrix,
        output_dir / "confusion_matrix_counts.png",
        f"{args.title} (counts)",
        normalized=False,
    )
    plot_matrix(
        matrix,
        output_dir / "confusion_matrix_normalized.png",
        f"{args.title} (row-normalized)",
        normalized=True,
    )

    metrics = {
        "source_csv": Path(args.input).as_posix(),
        "num_samples": total,
        "num_correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
    }
    with (output_dir / "confusion_matrix_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(f"Saved confusion matrix figures to: {output_dir}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")


if __name__ == "__main__":
    main()
