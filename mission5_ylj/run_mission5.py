
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

CONFIGS = {
    "50k_1shot": CONFIG_DIR / "joint_50k_1shot_lambda0.5.json",
    "50k_3shot": CONFIG_DIR / "joint_50k_3shot_lambda0.5.json",
    "500k_1shot": CONFIG_DIR / "joint_500k_1shot_lambda0.5.json",
    "500k_3shot": CONFIG_DIR / "joint_500k_3shot_lambda0.5.json",
    "lambda0.1": CONFIG_DIR / "joint_50k_1shot_lambda0.1.json",
    "lambda1.0": CONFIG_DIR / "joint_50k_1shot_lambda1.0.json",
}

CORE_CONFIGS = ("50k_1shot", "50k_3shot", "500k_1shot", "500k_3shot")


def run(command: list[str]) -> None:
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def show_environment() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA is unavailable; training will run on CPU.")


def prepare_data() -> None:
    run(
        [
            sys.executable,
            "-m",
            "data.mnist.download_MNIST",
            "--train-sizes",
            "50k",
            "500k",
        ]
    )


def run_config(name: str) -> None:
    config_path = CONFIGS[name]
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    run(
        [
            sys.executable,
            "-m",
            "src.training.train_joint_relational",
            "--config",
            str(config_path),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=sorted(CONFIGS),
        default="50k_1shot",
        help="Experiment to run (default: 50k_1shot).",
    )
    parser.add_argument(
        "--all-core",
        action="store_true",
        help="Run the four core 50k/500k and 1-shot/3-shot experiments.",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Download MNIST and construct both 50k and 500k pair datasets first.",
    )
    parser.add_argument(
        "--environment-only",
        action="store_true",
        help="Only print the Python/PyTorch/GPU environment and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_environment()
    if args.environment_only:
        return
    if args.prepare_data:
        prepare_data()
    if args.all_core:
        for config_name in CORE_CONFIGS:
            run_config(config_name)
    else:
        run_config(args.config)


if __name__ == "__main__":
    main()

