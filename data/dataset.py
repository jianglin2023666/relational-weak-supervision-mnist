from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.constants import SUPERVISED_DIR, WEAKLY_SUPERVISED_DIR


SUPPORTED_RELATION_TRAIN_SIZES = {"50k", "500k"}


class SupervisedMNISTDataset(Dataset):
    def __init__(self, split='train', transform=None):
        """
        Initialize the MNIST dataset.
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.split = split
        self.transform = transform
        
        # Load the appropriate data split
        if split == 'train':
            self.data = torch.load(Path(SUPERVISED_DIR) / "x_train.pt", weights_only=True)
            self.labels = torch.load(Path(SUPERVISED_DIR) / "y_train.pt", weights_only=True)
        elif split == 'val':
            self.data = torch.load(Path(SUPERVISED_DIR) / "x_val.pt", weights_only=True)
            self.labels = torch.load(Path(SUPERVISED_DIR) / "y_val.pt", weights_only=True)
        elif split == 'test':
            self.data = torch.load(Path(SUPERVISED_DIR) / "x_test.pt", weights_only=True)
            self.labels = torch.load(Path(SUPERVISED_DIR) / "y_test.pt", weights_only=True)
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'")
        
        # Add channel dimension if not present
        if len(self.data.shape) == 3:
            self.data = self.data.unsqueeze(1)
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        img = self.data[idx].float() / 255.0
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    

class WeaklySupervisedMNISTDataset(Dataset):
    def __init__(self, split='train', size='50k', transform=None):
        """
        Initialize the weakly supervised MNIST dataset.
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            size (str): For training split, one of '50k' or '500k'. Ignored for val/test.
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.split = split
        self.size = size
        self.transform = transform
        
        # Load the appropriate data split
        if split == 'train':
            if size not in SUPPORTED_RELATION_TRAIN_SIZES:
                choices = ", ".join(sorted(SUPPORTED_RELATION_TRAIN_SIZES))
                raise ValueError(f"Invalid size: {size}. Choose from: {choices}")
            x_path = Path(WEAKLY_SUPERVISED_DIR) / f"x_train_{size}.pt"
            y_path = Path(WEAKLY_SUPERVISED_DIR) / f"y_train_{size}.pt"
        elif split == 'val':
            x_path = Path(WEAKLY_SUPERVISED_DIR) / "x_val.pt"
            y_path = Path(WEAKLY_SUPERVISED_DIR) / "y_val.pt"
        elif split == 'test':
            x_path = Path(WEAKLY_SUPERVISED_DIR) / "x_test.pt"
            y_path = Path(WEAKLY_SUPERVISED_DIR) / "y_test.pt"
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'")
        
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing relational dataset files: {x_path}, {y_path}. "
                "Generate them with data.mnist.download_MNIST."
            )
        self.data = torch.load(x_path, weights_only=True)
        self.labels = torch.load(y_path, weights_only=True)

        # Add channel dimension if not present
        if len(self.data.shape) == 4:  # (N, 2, H, W)
            self.data = self.data.unsqueeze(2)  # (N, 2, 1, H, W)
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        img_pair = self.data[idx].float() / 255.0  # Shape: (2, 1, H, W)
        label = self.labels[idx]    # Shape: (1,)
        
        if self.transform:
            img_pair = self.transform(img_pair)
            
        return img_pair, label
