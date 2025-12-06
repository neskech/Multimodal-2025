from pydoc import text
import torch
from torch import nn
from power_spherical import PowerSpherical, HypersphericalUniform
from torch.distributions import Distribution

from losses.clipLoss import ClipLoss
from losses.distributions.VonMisesFisher import VonMisesFisher


def power_spherical_mean(distribution: PowerSpherical):
    return distribution.loc * distribution.base_dist.marginal_t.mean.unsqueeze(
        -1)


def power_spherical_mean2(distribution: PowerSpherical):
    alpha = (512 - 1) / 2 + distribution.scale
    beta = (512 - 1) / 2
    mean = (alpha - beta) / (alpha + beta)
    return distribution.loc * mean.unsqueeze(-1)


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

    def __init__(self,
                 clip_weight=1.0,
                 kl_weight=1.0,
                 num_samples=20,
                 var_reg_weight=0.1,
                 distribution_type='power_spherical',
                 use_mean_only=False,
                 expected_value=False,
                 label_smoothing=0.1):
        super(VClipLoss, self).__init__()
        self.clip_weight = clip_weight
        self.kl_weight = kl_weight
        self.num_samples = num_samples
        self.distribution_type = distribution_type
        self.use_mean_only = use_mean_only
        self.var_reg_weight = var_reg_weight
        self.expected_value = expected_value
        self.clip_loss = ClipLoss(label_smoothing)

    def forward(self,
                image_distribution: Distribution,
                text_distribution: Distribution,
                image_vars: torch.Tensor,
                text_vars: torch.Tensor,
                logits_scale: torch.Tensor,
                is_train: bool,
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

        if use_sampling and is_train:
            # Sample from distributions: shape will be [num_samples, batch_size, embedding_dim]
            if self.expected_value:
                clip_loss = self.clip_loss.forward(
                    power_spherical_mean2(image_distribution),
                    power_spherical_mean2(text_distribution),
                    logits_scale,
                    normalize=False)
            else:
                image_samples = image_distribution.rsample(
                    (self.num_samples, ))
                text_samples = text_distribution.rsample((self.num_samples, ))
                # l = 0
                # for m in range(self.num_samples):
                #     l = l + self.clip_loss.forward(image_samples[m],text_samples[m], logits_scale)
                
                # l = l / self.num_samples
                # clip_loss = l
                
                # #                                text_samples[m], logits_scale)
                clip_loss = self.clip_loss.forward_sampled(
                    image_samples, text_samples, logits_scale)
                clip_loss = clip_loss.mean()

        else:
            # Use mean/mode only
            image_samples_flat = get_distribution_mean(image_distribution)
            text_samples_flat = get_distribution_mean(text_distribution)

            # Compute CLIP loss
            clip_loss = self.clip_loss.forward(image_samples_flat,
                                               text_samples_flat, logits_scale)

        # Compute KL divergence
        kl_image = self._compute_kl_divergence(image_distribution)
        kl_text = self._compute_kl_divergence(text_distribution)

        # In vclipLoss.py, replace line 114:
        # Current: var_reg = torch.mean((image_vars - text_vars) ** 2)
        # Better: penalize log ratio
        log_image_vars = torch.log(image_vars + 1e-8)
        log_text_vars = torch.log(text_vars + 1e-8)
        if isinstance(image_distribution, torch.distributions.Normal):
            # For normal distributions, use log variance directly
            var_reg = torch.mean(
                ((log_image_vars - log_text_vars)**2).sum(dim=-1))
        else:
            var_reg = torch.mean((log_image_vars - log_text_vars)**2)
        total_loss = self.clip_weight * clip_loss + 0.5 * kl_weight * (
            kl_image + kl_text) + self.var_reg_weight * var_reg

        return {
            'total_loss': total_loss,
            'clip_loss': clip_loss,
            'image_kl_loss': kl_image,
            'var_reg': var_reg,
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
        if isinstance(distribution, torch.distributions.Normal):
            # Create a standard normal prior matching the device and shape of the input
            prior = torch.distributions.Normal(
                torch.zeros_like(distribution.loc),
                torch.ones_like(distribution.scale))

            # Sum over the embedding dimensions (D), average over batch (B) if needed,
            # but usually kl_divergence returns shape (B, D).
            # We explicitly sum over D to get the KL per vector, then mean over batch.
            kl = torch.distributions.kl_divergence(distribution, prior)
            return kl.sum(dim=-1).mean()
        if isinstance(distribution, PowerSpherical):
            # Use built-in KL divergence for PowerSpherical
            return torch.distributions.kl_divergence(
                distribution,
                HypersphericalUniform(dim=distribution.loc.shape[-1])).mean()
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
                log_surface_area = (d / 2) * math.log(
                    2 * math.pi) - torch.lgamma(torch.tensor(d / 2.0)).item()

            kl = -entropy - (-log_surface_area)
            return kl.mean()
        else:
            # Generic fallback: use entropy as proxy
            return -distribution.entropy().mean()
