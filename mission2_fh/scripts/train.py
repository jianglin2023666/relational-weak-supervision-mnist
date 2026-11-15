"""Train a new five-seed 100-epoch experiment with the upstream training code."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')


def main():
    delivery = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=delivery.parent)
    args = parser.parse_args()
    project = args.project_root.resolve()
    if not (project / 'src/training/train_supervised.py').is_file():
        parser.error('--project-root must be the original project checkout')
    sys.path.insert(0, str(project))
    os.chdir(project)
    import torch
    torch.set_num_threads(4)
    from data.mnist.download_MNIST import download_mnist
    from src.training.train_supervised import run_experiment
    download_mnist()
    run_experiment(delivery / 'configs/supervised_paper100.json')


if __name__ == '__main__':
    main()
