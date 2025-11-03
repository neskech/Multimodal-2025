import torch
from torch import nn
from power_spherical import PowerSpherical, HypersphericalUniform

from losses.clipLoss import ClipLoss


def power_spherical_mean(distribution: PowerSpherical):
    return distribution.loc * distribution.base_dist.marginal_t.mean.unsqueeze(
        -1)

def power_spherical_mean_normalized(distribution: PowerSpherical):
    return distribution.loc


class VClipLoss(nn.Module):

    def __init__(self, kl_weight=1.0, num_samples=20):
        super(VClipLoss, self).__init__()
        self.kl_weight = kl_weight
        self.num_samples = num_samples
        self.clip_loss = ClipLoss()

    def forward(self,
                image_distribution: PowerSpherical,
                text_distribution: PowerSpherical,
                logits_scale: torch.Tensor,
                kl_weight_override: float | None = None):
        """
        Compute V-CLIP loss using Power Spherical distributions.

        Args:
            means: Mean directions of the distributions, shape [batch_size, embedding_dim]
            concentrations: Concentration parameters of the distributions, shape [batch_size]
        """
        B, D = image_distribution.loc.shape
        
        kl_weight = kl_weight_override if kl_weight_override is not None else self.kl_weight
        if kl_weight > 0 and False:
            image_samples = image_distribution.rsample(sample_shape=(self.num_samples, ))
            text_samples = text_distribution.rsample(sample_shape=(self.num_samples, ))
            # 3. Reshape for ClipLoss
            # Shape [K * B, D]
            #image_samples_flat = image_samples.mean(dim=0)
            #text_samples_flat = text_samples.mean(dim=0)
            image_samples_flat = image_samples.reshape(self.num_samples * B, D)
            text_samples_flat = text_samples.reshape(self.num_samples * B, D)
            #print(image_samples.shape, text_samples.shape, "FUCK", image_samples)
        else:
            # Use mean directions only
            image_samples_flat = power_spherical_mean_normalized(image_distribution)
            text_samples_flat = power_spherical_mean_normalized(text_distribution)

        
        clip_loss = self.clip_loss.forward(image_samples_flat, text_samples_flat, logits_scale)
        kl_image = torch.distributions.kl_divergence(
            image_distribution,
            HypersphericalUniform(
                dim=image_distribution.loc.shape[-1])).mean()
        kl_text = torch.distributions.kl_divergence(
            text_distribution,
            HypersphericalUniform(dim=text_distribution.loc.shape[-1])).mean()

        total_loss = clip_loss + 0.5 * kl_weight * (kl_image + kl_text)
        return {
            'total_loss': total_loss,
            'clip_loss': clip_loss,
            'image_kl_loss': kl_image,
            'text_kl_loss': kl_text
        }
