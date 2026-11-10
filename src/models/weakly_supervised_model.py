import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from src.models.base_model import BaseModel


class WeaklySupervisedModel(nn.Module):
    """
    Model for digit classification that takes in pairs of images and a relational label (weakly supervised setting).
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
        super(WeaklySupervisedModel, self).__init__()
        
        self.feature_extractor = BaseModel(embedding_dim, hidden_dim, dropout)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2, 1, 28, 28). The 2 is for the two images in the pair.
            
        Returns:
            torch.Tensor: Probability of the first image being smaller than the second image of shape (batch_size, 2).
        """
        x1, x2 = x[:, 0, :, :, :], x[:, 1, :, :, :]
        # Squeeze dimension to get (batch_size, 1, 28, 28)
        x1 = x1.squeeze(2)
        x2 = x2.squeeze(2)

        # Get embeddings for the first input
        embedding1 = self.feature_extractor(x1)
        embedding2 = self.feature_extractor(x2)

        logits1 = self.classifier(embedding1)
        logits2 = self.classifier(embedding2)

        class_probs1 = F.softmax(logits1, dim=1)
        class_probs2 = F.softmax(logits2, dim=1)

        # Vectorized calculation of probabilities
        # Create masks for pairs where first digit is greater than second
        mask_greater = torch.tril(torch.ones(10, 10, device=x.device), diagonal=-1).bool()
        
        # Expand probabilities to match the mask shape
        probs1_expanded = class_probs1.unsqueeze(2)  # [batch_size, 10, 1]
        probs2_expanded = class_probs2.unsqueeze(1)  # [batch_size, 1, 10]
        
        # Calculate joint probabilities
        joint_probs = probs1_expanded * probs2_expanded  # [batch_size, 10, 10]
        
        prob_first_greater = (joint_probs * mask_greater).sum(dim=(1, 2))

        # The sum is mathematically in [0, 1], but CUDA floating-point
        # accumulation can produce values such as 1.0000001. BCELoss rejects
        # those values, so keep the probability inside a stable open interval.
        eps = torch.finfo(prob_first_greater.dtype).eps
        prob_first_greater = prob_first_greater.clamp(min=eps, max=1.0 - eps)

        return prob_first_greater.unsqueeze(1)
    
    def predict(self, x):
        """
        Predict the class of the image.
        x is a tensor of shape (batch_size, 1, 28, 28)
        """
        return F.softmax(self.classifier(self.feature_extractor(x)), dim=1)
    
    def get_embedding(self, x):
        """
        Get embeddings for visualization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).
            
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim).
        """
        return super().forward(x)
    
    def save_base_model(self, save_dir):
        """
        Save the base model.
        """
        torch.save(self.base_model.state_dict(), os.path.join(save_dir, 'base_model.pth'))

    def load_base_model(self, save_dir):
        """
        Load the base model.
        """
        self.base_model.load_state_dict(torch.load(os.path.join(save_dir, 'base_model.pth')))
