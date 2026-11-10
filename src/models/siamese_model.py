import torch
import torch.nn as nn
import os

from src.models.base_model import BaseModel


class SiameseModel(nn.Module):
    """
    Siamese model for learning from pairwise comparisons.
    """
    
    def __init__(self, embedding_dim=128, hidden_dim=64, dropout=0.2):
        """
        Args:
            embedding_dim (int, optional): Dimension of the embedding space.
            hidden_dim (int, optional): Dimension of the hidden layers.
            dropout (float, optional): Dropout rate.
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Feature extractors
        self.feature_extractor = BaseModel(embedding_dim, hidden_dim, dropout)
        
        # Relation head
        self.relation_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: tensor of shape (batch_size, 2, 1, 28, 28). The 2 is for the two images in the pair.
            
        Returns:
            torch.Tensor: P(first image > second image), shape (batch_size, 1).
        """
        if x.ndim != 5 or x.shape[1] != 2:
            raise ValueError(
                "Expected pair input with shape [batch, 2, 1, 28, 28], "
                f"but received {tuple(x.shape)}"
            )

        x1, x2 = x[:, 0, :, :, :], x[:, 1, :, :, :]

        # Both images pass through the same feature extractor, so the two
        # Siamese branches always share weights.
        embedding1 = self.feature_extractor(x1)
        embedding2 = self.feature_extractor(x2)
        
        # Concatenate embeddings for relation prediction
        concat_embedding = torch.cat([embedding1, embedding2], dim=1)
        relation_score = self.relation_head(concat_embedding)
        
        return relation_score
    
    def predict_relation(self, x1, x2):
        """
        Predict relation between two images.
        
        Args:
            x1 (torch.Tensor): First input tensor of shape (batch_size, 1, 28, 28).
            x2 (torch.Tensor): Second input tensor of shape (batch_size, 1, 28, 28).
            
        Returns:
            torch.Tensor: Relation scores of shape (batch_size, 1).
        """
        if x1.shape != x2.shape:
            raise ValueError(
                f"x1 and x2 must have the same shape, got {tuple(x1.shape)} "
                f"and {tuple(x2.shape)}"
            )
        image_pairs = torch.stack((x1, x2), dim=1)
        return self.forward(image_pairs)
    
    def extract_embedding(self, x):
        """
        Extract embeddings from the input.
        """
        return self.feature_extractor(x)
    
    def save_base_model(self, save_dir):
        """
        Save the base model.
        """
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.feature_extractor.state_dict(), os.path.join(save_dir, 'base_model.pth'))

    def load_base_model(self, save_dir, map_location=None):
        """
        Load the base model.
        """
        state_dict = torch.load(
            os.path.join(save_dir, 'base_model.pth'),
            map_location=map_location,
            weights_only=True,
        )
        self.feature_extractor.load_state_dict(state_dict)
