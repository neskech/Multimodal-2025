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

    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
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
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute similarity matrix
        logits = image_features @ text_features.T
        logits = torch.exp(logit_scale) * logits

        # Create labels (diagonal should be positive pairs)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # Compute contrastive loss
        loss = (F.cross_entropy(logits, labels, axis=0) +
                F.cross_entropy(logits, labels, axis=1)) / 2

        return loss
