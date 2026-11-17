r"""Joint relational and class-supervised MNIST model.
关系数据：
图片对 → 共享 BaseModel → 10类概率 → P(A>B) → 关系损失

少量标签数据：
单张图片 → 同一个 BaseModel → 10类输出 → 分类损失
计算联合损失 = \(L=L_{\text{relation}}+\lambda L_{\text{classification}}\)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel


class JointRelationalClassifier(nn.Module):
    """Train one classifier from pairwise relations and sparse class labels.

    The same CNN and classification head process every image. Pairwise relation
    probabilities are derived from the two predicted class distributions, while
    explicitly labelled images are trained with ordinary cross-entropy.
    """

    def __init__(self, num_classes=10, embedding_dim=128, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = BaseModel(embedding_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # mask[i, j] is true when class i is greater than class j.
        greater_mask = torch.tril(
            torch.ones(num_classes, num_classes, dtype=torch.bool),
            diagonal=-1,
        )
        self.register_buffer("greater_mask", greater_mask)

    def extract_embedding(self, images):
        """Return BaseModel embeddings for images shaped [B, 1, 28, 28]."""
        return self.feature_extractor(images)

    def class_logits(self, images):
        """Return unnormalized class scores shaped [B, num_classes]."""
        return self.classifier(self.extract_embedding(images))

    def predict_classes(self, images):
        """Return class probabilities for single images."""
        return F.softmax(self.class_logits(images), dim=1)

    def relation_probability(self, image_pairs):
        """Return P(first class > second class) for [B, 2, 1, 28, 28]."""
        if image_pairs.ndim != 5 or image_pairs.shape[1] != 2:
            raise ValueError(
                "Expected pair input with shape [batch, 2, 1, 28, 28], "
                f"but received {tuple(image_pairs.shape)}"
            )

        probabilities1 = self.predict_classes(image_pairs[:, 0])
        probabilities2 = self.predict_classes(image_pairs[:, 1])
        joint = probabilities1.unsqueeze(2) * probabilities2.unsqueeze(1)
        probability = joint.masked_select(self.greater_mask.unsqueeze(0)).view(
            image_pairs.shape[0], -1
        ).sum(dim=1, keepdim=True)
        eps = torch.finfo(probability.dtype).eps
        return probability.clamp(min=eps, max=1.0 - eps)

    def forward(self, image_pairs=None, labelled_images=None):
        """Compute either or both outputs needed for joint training."""
        if image_pairs is None and labelled_images is None:
            raise ValueError("Provide image_pairs, labelled_images, or both.")

        outputs = {}
        if image_pairs is not None:
            outputs["relation_probability"] = self.relation_probability(image_pairs)
        if labelled_images is not None:
            outputs["class_logits"] = self.class_logits(labelled_images)
        return outputs

    def joint_loss(
        self,
        image_pairs,
        relation_labels,
        labelled_images,
        class_labels,
        classification_weight=1.0,
    ):
        """Return total, relation, and classification losses.

        total_loss = relation_loss + classification_weight * classification_loss
        """
        if classification_weight < 0:
            raise ValueError("classification_weight must be nonnegative.")

        outputs = self(image_pairs=image_pairs, labelled_images=labelled_images)
        relation_labels = relation_labels.float().view_as(
            outputs["relation_probability"]
        )
        relation_loss = F.binary_cross_entropy(
            outputs["relation_probability"], relation_labels
        )
        classification_loss = F.cross_entropy(
            outputs["class_logits"], class_labels.long()
        )
        total_loss = relation_loss + classification_weight * classification_loss
        return {
            "loss": total_loss,
            "relation_loss": relation_loss,
            "classification_loss": classification_loss,
        }

    def save_base_model(self, save_dir):
        """Save only the shared BaseModel feature extractor."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.feature_extractor.state_dict(),
            os.path.join(save_dir, "base_model.pth"),
        )

    def load_base_model(self, save_dir, map_location=None):
        """Load BaseModel weights previously saved by ``save_base_model``."""
        state_dict = torch.load(
            os.path.join(save_dir, "base_model.pth"),
            map_location=map_location,
            weights_only=True,
        )
        self.feature_extractor.load_state_dict(state_dict)
