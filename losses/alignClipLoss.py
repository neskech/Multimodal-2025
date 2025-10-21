import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(
    image_embeds: torch.Tensor, text_embeds: torch.Tensor, logit_scale: float
):
    """
    Calculates the contrastive loss.

    Args:
        image_embeds: Tensor of shape (batch_size, embed_dim)
        text_embeds: Tensor of shape (batch_size, embed_dim)
        logit_scale: Learnable parameter for scaling logits
    """
    # Cosine similarity
    logits_per_image = logit_scale * image_embeds @ text_embeds.t()
    logits_per_text = logits_per_image.t()

    # Create labels (correct pairs are on the diagonal)
    batch_size = image_embeds.shape[0]
    labels = torch.arange(batch_size, device=image_embeds.device)

    # Cross-entropy loss for both directions, averaged
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)

    return (loss_img + loss_txt) / 2


def ims_loss(features, temperature=0.1):
    """
    Calculates the Intra-Modal Separation (IMS) loss.

    Args:
        features: Tensor of shape (batch_size, feature_dim)
        temperature: Scaling factor for logits
    """
    # Normalize features to lie on unit sphere (L2 norm, not by the dimension)
    features = F.normalize(features, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = features @ features.t()

    # Create labels (each sample is most similar to itself)
    batch_size = features.shape[0]
    labels = torch.arange(batch_size, device=features.device)

    # Get cross-entropy (between similarity matrix and labels)
    loss = F.cross_entropy(similarity_matrix / temperature, labels)

    return loss


class AlignCLIPLoss(nn.Module):
    def __init__(self, lambda_ims=1.0, temperature=0.1):
        """
        Initializes the combined AlignCLIP loss.

        Args:
            lambda_ims: The weight for the IMS loss component.
            temperature: The temperature for the IMS loss.
        """
        super().__init__()
        self.lambda_ims = lambda_ims
        self.temperature = temperature

    def forward(self, image_embeds, text_embeds, image_features, text_features, logit_scale):
        # Get contrastive loss
        loss_con = contrastive_loss(image_embeds, text_embeds, logit_scale)

        # Get IMS losses on both modalities
        loss_ims_img = ims_loss(image_features, self.temperature)
        loss_ims_txt = ims_loss(text_features, self.temperature)
        
        # Combine
        total_loss = loss_con + self.lambda_ims * (loss_ims_img + loss_ims_txt) / 2
        
        return total_loss