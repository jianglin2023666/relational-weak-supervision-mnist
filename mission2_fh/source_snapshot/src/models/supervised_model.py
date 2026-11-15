import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from src.models.base_model import BaseModel


class SupervisedModel(nn.Module):
    """
    Model for digit classification.
    Uses the base model as the backbone, and adds a classification head.
    """
    
    def __init__(self, num_classes=10, embedding_dim=128, hidden_dim=64, dropout=0.2):
        """
        Args:
            num_classes (int, optional): Number of classes.
            embedding_dim (int, optional): Dimension of the embedding space.
            hidden_dim (int, optional): Dimension of the hidden layers.
            dropout (float, optional): Dropout rate.
        """
        super(SupervisedModel, self).__init__()

        self.feature_extractor = BaseModel(embedding_dim, hidden_dim, dropout)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
            
        Returns:
            tuple: (logits, embedding)
                - logits: Tensor of shape (batch_size, num_classes)
                - embedding: Tensor of shape (batch_size, embedding_dim)
        """
        embedding = self.feature_extractor(x)
        logits = self.classifier(embedding)
        return logits
    
    def predict(self, x):
        """
        Predict class probabilities.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
            
        Returns:
            torch.Tensor: Class probabilities of shape (batch_size, num_classes).
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def get_embedding(self, x):
        """
        Get embeddings for visualization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
            
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim).
        """
        return self.feature_extractor(x)
    
    def save_base_model(self, save_dir):
        """
        Save the base model.
        """
        torch.save(self.feature_extractor.state_dict(), os.path.join(save_dir, 'base_model.pth'))

    def load_base_model(self, save_dir):
        """
        Load the base model.
        """
        self.feature_extractor.load_state_dict(torch.load(os.path.join(save_dir, 'base_model.pth')))