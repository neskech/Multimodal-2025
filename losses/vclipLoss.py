import torch
from torch import nn
from power_spherical import PowerSpherical, HypersphericalUniform
from torch.distributions import Distribution

from losses.clipLoss import ClipLoss
from losses.distributions.VonMisesFisher import VonMisesFisher


def power_spherical_mean(distribution: PowerSpherical):
    return distribution.loc * distribution.base_dist.marginal_t.mean.unsqueeze(
        -1)

def power_spherical_mean_normalized(distribution: PowerSpherical):
    return distribution.loc

def vmf_mean_normalized(distribution: VonMisesFisher):
    """Get the normalized mean (mode) of a VonMisesFisher distribution."""
    return distribution.mode

def get_distribution_mean(distribution):
    """Get the mean/mode of a distribution regardless of type."""
    if isinstance(distribution, PowerSpherical):
        return power_spherical_mean_normalized(distribution)
    elif isinstance(distribution, VonMisesFisher):
        return vmf_mean_normalized(distribution)
    else:
        # Generic fallback
        return distribution.mean


class VClipLoss(nn.Module):
    """
    Variational CLIP Loss with support for multiple spherical distributions.
    
    Supports:
    - PowerSpherical distribution (default)
    - VonMisesFisher distribution
    - Any distribution that implements rsample() and works on the unit sphere
    
    Args:
        kl_weight: Weight for the KL divergence term
        num_samples: Number of samples to draw from the distributions
        distribution_type: Type of distribution to use ('power_spherical' or 'vmf')
        use_mean_only: If True, use only the mean/mode instead of sampling
    """

    def __init__(self, kl_weight=1.0, num_samples=20, distribution_type='power_spherical', use_mean_only=False, label_smoothing=0.1):
        super(VClipLoss, self).__init__()
        self.kl_weight = kl_weight
        self.num_samples = num_samples
        self.distribution_type = distribution_type
        self.use_mean_only = use_mean_only
        self.clip_loss = ClipLoss(label_smoothing)

    def forward(self,
                image_distribution: Distribution,
                text_distribution: Distribution,
                logits_scale: torch.Tensor,
                kl_weight_override: float | None = None):
        """
        Compute V-CLIP loss using spherical distributions.

        Args:
            image_distribution: Distribution for image embeddings (PowerSpherical or VonMisesFisher)
            text_distribution: Distribution for text embeddings (PowerSpherical or VonMisesFisher)
            logits_scale: Temperature parameter for scaling logits
            kl_weight_override: Optional override for kl_weight
            
        Returns:
            Dictionary with total_loss, clip_loss, image_kl_loss, text_kl_loss
        """
        B, D = image_distribution.loc.shape
        
        kl_weight = kl_weight_override if kl_weight_override is not None else self.kl_weight
        
        # Determine whether to use sampling or just the mean
        use_sampling = kl_weight > 0 
        
        if use_sampling:
            # Sample from distributions: shape will be [num_samples, batch_size, embedding_dim]
            image_samples = image_distribution.rsample((self.num_samples,))
            text_samples = text_distribution.rsample((self.num_samples,))
            
            # Reshape to [num_samples * batch_size, embedding_dim]
            image_samples_flat = image_samples.reshape(self.num_samples * B, D)
            text_samples_flat = text_samples.reshape(self.num_samples * B, D)
        else:
            # Use mean/mode only
            image_samples_flat = get_distribution_mean(image_distribution)
            text_samples_flat = get_distribution_mean(text_distribution)

        # Compute CLIP loss
        clip_loss = self.clip_loss.forward(image_samples_flat, text_samples_flat, logits_scale)
        
        # Compute KL divergence
        kl_image = self._compute_kl_divergence(image_distribution)
        kl_text = self._compute_kl_divergence(text_distribution)

        total_loss = clip_loss + 0.5 * kl_weight * (kl_image + kl_text)
        
        return {
            'total_loss': total_loss,
            'clip_loss': clip_loss,
            'image_kl_loss': kl_image,
            'text_kl_loss': kl_text
        }
    
    def _compute_kl_divergence(self, distribution):
        """
        Compute KL divergence to uniform distribution on the sphere.
        
        Args:
            distribution: Either PowerSpherical or VonMisesFisher
            
        Returns:
            KL divergence value
        """
        if isinstance(distribution, PowerSpherical):
            # Use built-in KL divergence for PowerSpherical
            return torch.distributions.kl_divergence(
                distribution,
                HypersphericalUniform(dim=distribution.loc.shape[-1])
            ).mean()
        elif isinstance(distribution, VonMisesFisher):
            # For VonMisesFisher, use entropy-based approximation
            # KL(p || uniform) = -H(p) - log(1/surface_area)
            # For uniform on d-dim sphere: log(surface_area) = log(2π^(d/2) / Γ(d/2))
            
            d = distribution.dim
            entropy = distribution.entropy()
            
            # Log surface area of unit sphere in d dimensions
            # S_d = 2π^(d/2) / Γ(d/2)
            import math
            if d == 2:
                log_surface_area = math.log(2 * math.pi)
            else:
                # Approximate using Stirling's approximation for large d
                log_surface_area = (d/2) * math.log(2 * math.pi) - torch.lgamma(torch.tensor(d/2.0)).item()
            
            kl = -entropy - (-log_surface_area)
            return kl.mean()
        else:
            # Generic fallback: use entropy as proxy
            return -distribution.entropy().mean()
