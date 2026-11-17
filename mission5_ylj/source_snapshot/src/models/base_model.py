import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base CNN model that is used as the backbone for various models.
    Used to keep the same architecture for various tasks.
    """
    
    def __init__(self, embedding_dim=128, hidden_dim=64, dropout=0.2):
        """
        Args:
            embedding_dim (int, optional): Dimension of the embedding space.
            hidden_dim (int, optional): Dimension of the hidden layers.
            dropout (float, optional): Dropout rate.
        """
        super(BaseModel, self).__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Embedding layer
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
            
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim).
        """
        features = self.feature_extractor(x)
        embedding = self.embedding_layer(features)
        return embedding