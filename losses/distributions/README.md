# Von Mises-Fisher Distribution Implementation

## Overview

This directory contains a PyTorch implementation of the **von Mises-Fisher (vMF) distribution** on the unit hypersphere. The implementation is based on the excellent [vmf-lib](https://github.com/minyoungkim21/vmf-lib) repository, which has been added as a git submodule.

## The von Mises-Fisher Distribution

The vMF distribution is a probability distribution on the (d-1)-dimensional unit sphere in R^d, parameterized by:

- **μ (loc)**: Mean direction (unit vector of dimension d)
- **κ (concentration)**: Concentration parameter (scalar > 0)

The probability density function is:

```
p(x | μ, κ) = C_d(κ) * exp(κ * μ^T x)
```

where:
- x is a point on the unit sphere (||x|| = 1)
- C_d(κ) is the normalization constant involving modified Bessel functions

### Properties

- **Higher κ** → samples concentrated around μ
- **Lower κ** → samples more uniformly distributed on sphere
- **Mode**: The mean direction μ
- **Support**: Unit sphere in R^d

## Usage

```python
import torch
from losses.distributions.VonMisesFisher import VonMisesFisher

# Create mean direction (will be normalized automatically)
loc = torch.randn(batch_size, dim)

# Concentration parameter
concentration = torch.ones(batch_size) * 10.0

# Create distribution
vmf = VonMisesFisher(loc, concentration)

# Sample from the distribution
samples = vmf.sample((num_samples,))

# Compute log probability
log_probs = vmf.log_prob(samples)

# Reparameterizable sampling (for VAE, etc.)
samples_grad = vmf.rsample((num_samples,))

# Get properties
mean = vmf.mean
mode = vmf.mode
entropy = vmf.entropy()
```

## Implementation Details

### Key Features

1. **Exact Log Partition Function**: Uses `mpmath` for high-precision computation of modified Bessel functions
2. **Efficient Sampling**: Implements the rejection sampling algorithm from Wood (1994)
3. **Autodiff Support**: Full gradient support for variational inference
4. **Batch Processing**: Supports arbitrary batch dimensions
5. **High-Dimensional**: Tested up to 512 dimensions

### Sampling Algorithm

The implementation uses rejection sampling with the following steps:

1. Sample uniform unit vectors in R^{d-1}
2. Use rejection sampling to generate v_0 component
3. Form sample as [v_0; √(1-v_0²) * v]
4. Apply Householder transformation to rotate to mean direction

### Gradient Flow

The distribution supports both:
- `sample()`: Standard sampling (no gradients)
- `rsample()`: Reparameterizable sampling (with gradients)

## Testing

Run the test suite:

```bash
uv run python test_vmf_distribution.py
```

This will:
- Test basic functionality (sampling, log_prob, etc.)
- Visualize samples for different concentration parameters
- Test high-dimensional cases (up to 512D)
- Verify gradient flow

## Visualization Example

The test creates a visualization showing how concentration affects the distribution:

![vMF Samples](vmf_distribution_samples.png)

## References

1. **Original Paper**: Mardia, K. V., & Jupp, P. E. (2009). Directional Statistics. John Wiley & Sons.
2. **Sampling Algorithm**: Wood, A. T. (1994). Simulation of the von Mises Fisher distribution. Communications in Statistics-Simulation and Computation.
3. **Implementation Reference**: [vmf-lib](https://github.com/minyoungkim21/vmf-lib) by Min-Young Kim

## Integration with VClip Loss

This distribution can be used as an alternative to PowerSpherical in the VClip loss:

```python
from losses.distributions.VonMisesFisher import VonMisesFisher
from losses.vclipLoss import VClipLoss

# Create vMF distributions
image_dist = VonMisesFisher(image_locs, image_concentrations)
text_dist = VonMisesFisher(text_locs, text_concentrations)

# Use with VClip (may need adapter)
# loss = vclip_loss(image_dist, text_dist, logits_scale)
```

## Notes

- The normalization constant computation uses `mpmath` for numerical stability
- Gradients flow through the log partition function via custom autograd
- Sampling is exact (not approximate)
- All samples are guaranteed to lie on the unit sphere (||x|| = 1)

## Future Improvements

- [ ] Implement mixture of vMF distributions
- [ ] Add KL divergence between vMF distributions
- [ ] Optimize sampling for very high dimensions (>1000)
- [ ] Add support for conditional vMF
- [ ] Implement variational vMF families
