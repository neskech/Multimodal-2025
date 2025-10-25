"""
Simple CLIP contrastive loss implementation for single-GPU training.
"""

import torch
import torch.nn.functional as F
from torch import nn

# https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py#L68C1-L156C1
# https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py#L34


class ClipLoss(nn.Module):
    """
    Simple CLIP contrastive loss for single-GPU training.
    Computes contrastive loss between image and text features.
    """

    def __init__(self, temperature: float = 0.07, label_smoothing: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.register_buffer("t", torch.tensor(self.temperature))
        self.register_buffer("smoothing", torch.tensor(self.label_smoothing))

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        """
        Compute CLIP contrastive loss.

        Args:
            image_features: Image embeddings, shape [batch_size, embedding_dim]
            text_features: Text embeddings, shape [batch_size, embedding_dim]
            logit_scale: Temperature parameter for scaling logits

        Returns:
            Contrastive loss value
        """
        # Normalize features
        image_features = torch.nn.functional.normalize(image_features, dim=-1, p=2)
        text_features = torch.nn.functional.normalize(text_features, dim=-1, p=2)

        # Clamp to avoid numerical issues
        image_features = torch.clamp(image_features, -1.0, 1.0)
        text_features = torch.clamp(text_features, -1.0, 1.0)

        batch_size = image_features.shape[0]

        # Compute similarity matrices
        logits_per_image = image_features @ text_features.T / self.temperature
        logits_per_text = text_features @ image_features.T / self.temperature

        # Clamp logits to avoid overflow in exp
        logits_per_image = torch.clamp(logits_per_image, -20, 20)
        logits_per_text = torch.clamp(logits_per_text, -20, 20)

        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=image_features.device)

        # Use stable cross entropy with label smoothing
        loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels, label_smoothing=self.label_smoothing)
        loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels, label_smoothing=self.label_smoothing)

        loss = (loss_img + loss_txt) / 2

        return loss
