"""Download MNIST and build reproducible relational image-pair datasets."""

import argparse
from pathlib import Path

import torch
from torchvision import datasets

from src.utils.constants import DATA_DIR, SUPERVISED_DIR, WEAKLY_SUPERVISED_DIR


SUPPORTED_TRAIN_SIZES = {
    "50k": 50_000,
    "500k": 500_000,
}
VALIDATION_PAIRS = 20_000
TEST_PAIRS = 20_000


def _supervised_paths():
    directory = Path(SUPERVISED_DIR)
    return {
        "x_train": directory / "x_train.pt",
        "y_train": directory / "y_train.pt",
        "x_val": directory / "x_val.pt",
        "y_val": directory / "y_val.pt",
        "x_test": directory / "x_test.pt",
        "y_test": directory / "y_test.pt",
    }


def download_mnist(force=False):
    """Download MNIST and save the 50k/10k/10k supervised splits."""
    paths = _supervised_paths()
    Path(SUPERVISED_DIR).mkdir(parents=True, exist_ok=True)
    Path(WEAKLY_SUPERVISED_DIR).mkdir(parents=True, exist_ok=True)

    if not force and all(path.exists() for path in paths.values()):
        print("Supervised MNIST tensors already exist; skipping preparation.")
        return

    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True)

    # Clone slices so each .pt file stores only its own tensor storage.
    train_data = train_dataset.data[:50_000].clone()
    train_labels = train_dataset.targets[:50_000].clone()
    val_data = train_dataset.data[50_000:].clone()
    val_labels = train_dataset.targets[50_000:].clone()
    test_data = test_dataset.data.clone()
    test_labels = test_dataset.targets.clone()

    tensors = {
        "x_train": train_data,
        "y_train": train_labels,
        "x_val": val_data,
        "y_val": val_labels,
        "x_test": test_data,
        "y_test": test_labels,
    }
    for name, tensor in tensors.items():
        torch.save(tensor, paths[name])

    print("Saved supervised splits: train=50000, val=10000, test=10000.")


def create_pairs(x_data, y_data, num_pairs, seed, chunk_size=65_536):
    """Create unequal-digit pairs using batched random sampling."""
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")
    if len(x_data) != len(y_data):
        raise ValueError("Image and label tensors must have the same length.")

    generator = torch.Generator().manual_seed(seed)
    x_pairs = torch.empty(
        (num_pairs, 2, *x_data.shape[1:]), dtype=x_data.dtype
    )
    y_pairs = torch.empty((num_pairs, 1), dtype=torch.float32)
    filled = 0

    while filled < num_pairs:
        remaining = num_pairs - filled
        # About 90% of uniformly sampled MNIST pairs have different digits.
        candidates = min(chunk_size, max(1_024, (remaining * 10) // 9 + 32))
        first = torch.randint(len(x_data), (candidates,), generator=generator)
        second = torch.randint(len(x_data), (candidates,), generator=generator)
        valid = y_data[first] != y_data[second]
        first = first[valid]
        second = second[valid]
        take = min(remaining, len(first))
        if take == 0:
            continue

        destination = slice(filled, filled + take)
        x_pairs[destination, 0] = x_data[first[:take]]
        x_pairs[destination, 1] = x_data[second[:take]]
        y_pairs[destination, 0] = (
            y_data[first[:take]] > y_data[second[:take]]
        ).float()
        filled += take

    return x_pairs, y_pairs


def _save_pair_split(name, x_data, y_data, num_pairs, seed, force):
    directory = Path(WEAKLY_SUPERVISED_DIR)
    x_path = directory / f"x_{name}.pt"
    y_path = directory / f"y_{name}.pt"

    if not force and x_path.exists() and y_path.exists():
        print(f"Pair split '{name}' already exists; skipping.")
        return

    print(f"Creating pair split '{name}' with {num_pairs:,} pairs...")
    x_pairs, y_pairs = create_pairs(x_data, y_data, num_pairs, seed)
    torch.save(x_pairs, x_path)
    torch.save(y_pairs, y_path)
    positive_rate = 100.0 * y_pairs.mean().item()
    print(f"Saved '{name}'; positive relation rate={positive_rate:.2f}%.")


def construct_weakly_supervised_dataset(
    train_sizes=("50k",), seed=42, force=False
):
    """Build selected training pair sets and fixed validation/test pair sets."""
    unknown = set(train_sizes) - set(SUPPORTED_TRAIN_SIZES)
    if unknown:
        raise ValueError(
            f"Unsupported train sizes: {sorted(unknown)}. "
            f"Choose from {sorted(SUPPORTED_TRAIN_SIZES)}."
        )

    paths = _supervised_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Supervised MNIST tensors are missing. Run download_mnist() first: "
            + ", ".join(missing)
        )

    tensors = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in paths.items()
    }

    for size_name in train_sizes:
        _save_pair_split(
            f"train_{size_name}",
            tensors["x_train"],
            tensors["y_train"],
            SUPPORTED_TRAIN_SIZES[size_name],
            seed + SUPPORTED_TRAIN_SIZES[size_name],
            force,
        )

    _save_pair_split(
        "val",
        tensors["x_val"],
        tensors["y_val"],
        VALIDATION_PAIRS,
        seed + 10_000,
        force,
    )
    _save_pair_split(
        "test",
        tensors["x_test"],
        tensors["y_test"],
        TEST_PAIRS,
        seed + 20_000,
        force,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download MNIST and construct relational pair datasets."
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        choices=sorted(SUPPORTED_TRAIN_SIZES),
        default=["50k"],
        help="Training pair-set sizes to generate (default: 50k).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing supervised and relational tensor files.",
    )
    args = parser.parse_args()

    download_mnist(force=args.force)
    construct_weakly_supervised_dataset(
        train_sizes=args.train_sizes,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
