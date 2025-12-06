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

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer("smoothing", torch.tensor(self.label_smoothing))

    def forward_sampled(self, image_features: torch.Tensor, text_features: torch.Tensor, logits_scale: torch.Tensor):
        B1, B2, E = image_features.shape

        # 1. Normalize features (dim=-1 handles the last dimension correctly for 3D tensors)
        image_features = torch.nn.functional.normalize(image_features, dim=-1, p=2)
        text_features = torch.nn.functional.normalize(text_features, dim=-1, p=2)

        # 2. Batched Matrix Multiplication
        # We want [B1, B2, B2]. 
        # By permuting only the last two dims of text_features, torch.matmul 
        # automatically treats the first dim (B1) as a batch dimension.
        logit_scale = logits_scale.clamp(max=3.912).exp()
        
        # (B1, B2, E) @ (B1, E, B2) -> (B1, B2, B2)
        logits_per_image = logit_scale * image_features @ text_features.permute(0, 2, 1)
        logits_per_text = logits_per_image.permute(0, 2, 1)
      #  print("FUCKING LOGIT SCALE", logit_scale, "AND AVG LOGITS", logits_per_image.mean())

        # 3. Create Labels
        # Inside every B1 group, the targets are simply [0, 1, 2, ... B2-1]
        # We expand this to shape [B1, B2]
        labels = torch.arange(B2, device=image_features.device).expand(B1, B2)

        # 4. Flatten for Cross Entropy
        # PyTorch CrossEntropy expects 2D logits [N, C] and 1D labels [N].
        # Here N = B1*B2 total samples, and C = B2 classes.
        logits_img_flat = logits_per_image.reshape(B1 * B2, B2)
        logits_txt_flat = logits_per_text.reshape(B1 * B2, B2)
        labels_flat = labels.reshape(B1 * B2)

        # 5. Compute Loss with reduction='none'
        # This gives us a raw loss vector of shape [B1 * B2] 
        # instead of a single scalar average.
        loss_img_flat = torch.nn.functional.cross_entropy(
            logits_img_flat, 
            labels_flat, 
            label_smoothing=self.label_smoothing, 
            reduction='none' # <--- CRITICAL STEP
        )
        loss_txt_flat = torch.nn.functional.cross_entropy(
            logits_txt_flat, 
            labels_flat, 
            label_smoothing=self.label_smoothing, 
            reduction='none' # <--- CRITICAL STEP
        )

        # 6. Reshape back to [B1, B2] and average over B2
        # We calculate the mean loss for each group of B2 items
        loss_img = loss_img_flat.view(B1, B2).mean(dim=1)
        loss_txt = loss_txt_flat.view(B1, B2).mean(dim=1)

        # 7. Final combination
        # Result shape: [B1]
        loss = (loss_img + loss_txt) / 2

        return loss

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, logits_scale: torch.Tensor, normalize=True):
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
        if normalize:
            image_features = torch.nn.functional.normalize(image_features, dim=-1, p=2)
            text_features = torch.nn.functional.normalize(text_features, dim=-1, p=2)

        # Clamp to avoid numerical issues
        # image_features = torch.clamp(image_features, -1.0, 1.0)
        # text_features = torch.clamp(text_features, -1.0, 1.0)

        batch_size = image_features.shape[0]

        # cosine similarity as logits
        logit_scale = logits_scale.clamp(max=4.6052).exp()
        logits_per_image = logit_scale.float() * image_features.float() @ text_features.t().float()
        logits_per_text = logits_per_image.t()
        # Clamp logits to avoid overflow in exp
        # logits_per_image = torch.clamp(logits_per_image, -20, 20)
        # logits_per_text = torch.clamp(logits_per_text, -20, 20)

        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=image_features.device)

        # Use stable cross entropy with label smoothing
        loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels, label_smoothing=self.label_smoothing)
        loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels, label_smoothing=self.label_smoothing)

        loss = (loss_img + loss_txt) / 2

        return loss
