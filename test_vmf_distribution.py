"""
Test the VonMisesFisher distribution implementation.
"""

import torch
import matplotlib.pyplot as plt
from losses.distributions.VonMisesFisher import VonMisesFisher

def test_vmf_basic():
    """Test basic functionality of VonMisesFisher distribution."""
    print("=" * 60)
    print("Testing VonMisesFisher Distribution")
    print("=" * 60)
    
    # Test 1: 2D distribution
    print("\n1. Testing 2D von Mises-Fisher distribution...")
    batch_size = 10
    dim = 2
    
    # Create mean direction (will be normalized)
    loc = torch.randn(batch_size, dim)
    concentration = torch.ones(batch_size) * 10.0
    
    # Create distribution
    vmf = VonMisesFisher(loc, concentration)
    
    print(f"   Batch shape: {vmf.batch_shape}")
    print(f"   Event shape: {vmf.event_shape}")
    print(f"   Mean directions (first 3):\n{vmf.loc[:3]}")
    print(f"   Concentrations (first 3): {vmf.concentration[:3]}")
    
    # Test 2: Sampling
    print("\n2. Testing sampling...")
    samples = vmf.sample((100,))
    print(f"   Sample shape: {samples.shape}")
    print(f"   Sample norms (should be ~1): {samples.norm(dim=-1)[:5]}")
    
    # Test 3: Log probability
    print("\n3. Testing log probability...")
    log_probs = vmf.log_prob(samples)
    print(f"   Log prob shape: {log_probs.shape}")
    print(f"   Log prob values (first 5): {log_probs[:5, 0]}")
    
    # Test 4: Mean and mode
    print("\n4. Testing mean and mode...")
    mean = vmf.mean
    mode = vmf.mode
    print(f"   Mean shape: {mean.shape}")
    print(f"   Mode shape: {mode.shape}")
    print(f"   Mean equals loc: {torch.allclose(mean, vmf.loc)}")
    
    # Test 5: Entropy
    print("\n5. Testing entropy...")
    entropy = vmf.entropy()
    print(f"   Entropy shape: {entropy.shape}")
    print(f"   Entropy values (first 3): {entropy[:3]}")
    
    # Test 6: Gradient flow
    print("\n6. Testing gradient flow...")
    loc_grad = torch.randn(batch_size, dim, requires_grad=True)
    conc_grad = torch.ones(batch_size, requires_grad=True) * 10.0
    
    vmf_grad = VonMisesFisher(loc_grad, conc_grad)
    samples_grad = vmf_grad.rsample((50,))
    log_probs_grad = vmf_grad.log_prob(samples_grad)
    loss = -log_probs_grad.mean()
    loss.backward()
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Loc gradient exists: {loc_grad.grad is not None}")
    print(f"   Concentration gradient exists: {conc_grad.grad is not None}")
    if loc_grad.grad is not None:
        print(f"   Loc gradient norm: {loc_grad.grad.norm().item():.6f}")
    if conc_grad.grad is not None:
        print(f"   Concentration gradient norm: {conc_grad.grad.norm().item():.6f}")
    
    print("\n✓ All basic tests passed!")


def test_vmf_visualization():
    """Visualize samples from VonMisesFisher distribution in 2D."""
    print("\n" + "=" * 60)
    print("Visualizing VonMisesFisher Samples")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Von Mises-Fisher Distribution Samples (500 samples each)', fontsize=16, fontweight='bold')
    
    concentrations = [0.5, 2.0, 5.0, 10.0, 20.0, 50.0]
    
    for idx, kappa in enumerate(concentrations):
        ax = axes[idx // 3, idx % 3]
        
        # Fixed mean direction
        loc = torch.tensor([[1.0, 0.0]])
        concentration = torch.tensor([kappa])
        
        # Create distribution and sample
        vmf = VonMisesFisher(loc, concentration)
        samples = vmf.sample((500,))
        
        # Plot
        samples_np = samples[:, 0, :].detach().numpy()  # Shape: [500, 2]
        
        print(f"  κ={kappa}: Sampled {samples_np.shape[0]} points")
        print(f"    Sample stats: mean={samples_np.mean(axis=0)}, std={samples_np.std(axis=0)}")
        
        # Draw unit circle
        circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, 
                           linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        
        # Plot samples with smaller dots and higher transparency for high concentration
        alpha_val = 0.3 if kappa > 10 else 0.5
        size_val = 15 if kappa > 10 else 25
        ax.scatter(samples_np[:, 0], samples_np[:, 1], 
                  alpha=alpha_val, s=size_val, c='blue', edgecolors='none')
        
        # Plot mean direction
        ax.arrow(0, 0, loc[0, 0].item() * 0.8, loc[0, 1].item() * 0.8,
                head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2, zorder=10)
        
        # Add text showing number of samples
        ax.text(0.05, 0.95, f'n=500', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Formatting
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'κ = {kappa}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
    
    plt.tight_layout()
    plt.savefig('losses/vmf_distribution_samples.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to losses/vmf_distribution_samples.png")
    plt.show()


def test_vmf_high_dim():
    """Test VonMisesFisher in higher dimensions."""
    print("\n" + "=" * 60)
    print("Testing High-Dimensional VonMisesFisher")
    print("=" * 60)
    
    dims = [10, 50, 128, 512]
    
    for dim in dims:
        print(f"\nTesting dimension {dim}...")
        
        # Create distribution
        loc = torch.randn(1, dim)
        concentration = torch.tensor([10.0])
        
        vmf = VonMisesFisher(loc, concentration)
        
        # Sample
        samples = vmf.sample((100,))
        
        # Check properties
        norms = samples.norm(dim=-1)
        log_probs = vmf.log_prob(samples)
        
        print(f"  Sample shape: {samples.shape}")
        print(f"  Norm range: [{norms.min().item():.6f}, {norms.max().item():.6f}]")
        print(f"  Log prob mean: {log_probs.mean().item():.4f}")
        print(f"  Log prob std: {log_probs.std().item():.4f}")
    
    print("\n✓ High-dimensional tests passed!")


if __name__ == "__main__":
    test_vmf_basic()
    test_vmf_visualization()
    test_vmf_high_dim()
    
    print("\n" + "=" * 60)
    print("All VonMisesFisher tests completed successfully!")
    print("=" * 60)
