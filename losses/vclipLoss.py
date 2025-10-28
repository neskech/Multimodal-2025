import torch
from torch import nn
from power_spherical import PowerSpherical, HypersphericalUniform

from losses.clipLoss import ClipLoss

class VClipLoss(nn.Module):
    def __init__(self, temperature=0.07, kl_weight=1.0):
        super(VClipLoss, self).__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.clip_loss = ClipLoss(temperature=temperature)

    def forward(self, image_distribution: PowerSpherical, text_distribution: PowerSpherical):
        """
        Compute V-CLIP loss using Power Spherical distributions.

        Args:
            means: Mean directions of the distributions, shape [batch_size, embedding_dim]
            concentrations: Concentration parameters of the distributions, shape [batch_size]
        """
        image_samples = image_distribution.rsample()
        text_samples = text_distribution.rsample()

        clip_loss = self.clip_loss.forward(image_samples, text_samples)
        kl_image = torch.distributions.kl_divergence(
            image_distribution,
            HypersphericalUniform(dim=image_distribution.loc.shape[-1])
        ).mean()
        kl_text = torch.distributions.kl_divergence(
            text_distribution,
            HypersphericalUniform(dim=text_distribution.loc.shape[-1])
        ).mean()


        total_loss = clip_loss + 0.5 * self.kl_weight * (kl_image + kl_text)
        return {
            'total_loss': total_loss,
            'clip_loss': clip_loss,
            'image_kl_loss': kl_image,
            'text_kl_loss': kl_text
        }