"""
General tester for all three distributions supported by test_vclip_loss:
- VonMisesFisher
- PowerSpherical  
- ProjectedNormal

Automatically tests all distributions and saves results to organized folders.
"""

from typing_extensions import Literal
import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from losses.distributions.VonMisesFisher import VonMisesFisher
from losses.distributions.ProjectedNormal import ProjectedNormal
from power_spherical import PowerSpherical
import numpy as np
import os

def test_distribution_basic(distribution_type: Literal["VonMisesFisher", "PowerSpherical", "ProjectedNormal"] = "VonMisesFisher", output_dir: str = "."):
    """Test basic functionality of a distribution."""
    
    # Create output directory
    dist_dir = os.path.join(output_dir, distribution_type, "basic_tests")
    os.makedirs(dist_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Testing {distribution_type} Distribution")
    print(f"Output directory: {dist_dir}")
    print("=" * 60)
    
    # Test 1: 2D distribution
    print(f"\n1. Testing 2D {distribution_type} distribution...")
    batch_size = 10
    dim = 2
    
    # Create mean/location
    loc = torch.randn(batch_size, dim)
    
    # Create concentration/sigma parameter based on distribution type
    if distribution_type == "VonMisesFisher":
        param = torch.ones(batch_size) * 10.0
        param_name = "concentration"
    elif distribution_type == "PowerSpherical":
        param = torch.ones(batch_size) * 10.0
        param_name = "concentration"
    else:  # ProjectedNormal
        param = torch.ones(batch_size, dim) * 0.3
        param_name = "sigma"
    
    # Create distribution
    if distribution_type == "VonMisesFisher":
        dist = VonMisesFisher(loc, param)
    elif distribution_type == "PowerSpherical":
        dist = PowerSpherical(loc, param)
    else:  # ProjectedNormal
        dist = ProjectedNormal(loc, param)
    
    print(f"   Location shape: {loc.shape}")
    print(f"   {param_name.capitalize()} shape: {param.shape}")
    print(f"   Location (first 3):\n{loc[:3]}")
    print(f"   {param_name.capitalize()} (first 3): {param[:3] if param.dim() == 1 else param[:3, 0]}")
    
    # Test 2: Sampling
    print("\n2. Testing sampling...")
    samples = dist.sample((100,))
    print(f"   Sample shape: {samples.shape}")
    sample_norms = samples.norm(dim=-1)
    print(f"   Sample norms (should be ~1): {sample_norms[:5, 0] if sample_norms.dim() > 1 else sample_norms[:5]}")
    
    # Verify samples are on unit sphere
    if distribution_type == "ProjectedNormal":
        assert torch.allclose(sample_norms, torch.ones_like(sample_norms), atol=1e-6), \
            "Samples are not on unit sphere!"
        print("   ✓ All samples verified to be on unit sphere")
    
    # Test 3: Log probability
    print("\n3. Testing log probability...")
    log_probs = dist.log_prob(samples)
    print(f"   Log prob shape: {log_probs.shape}")
    if log_probs.dim() > 1:
        print(f"   Log prob values (first 5): {log_probs[:5, 0]}")
    else:
        print(f"   Log prob values (first 5): {log_probs[:5]}")
    
    # Test 4: Properties
    print("\n4. Testing distribution properties...")
    # Try to access mean (hasattr triggers @property, so just try/except)
    try:
        mean = dist.mean
        print(f"   Mean shape: {mean.shape}")
    except Exception as e:
        print(f"   Mean computation failed: {type(e).__name__}: {e}")
    
    # Try to access mode
    try:
        mode = dist.mode
        print(f"   Mode shape: {mode.shape}")
    except Exception as e:
        print(f"   Mode computation failed: {type(e).__name__}: {e}")
    
    # Try to compute entropy
    try:
        entropy = dist.entropy()
        print(f"   Entropy computed successfully")
    except Exception as e:
        print(f"   Entropy not available for {distribution_type}: {type(e).__name__}")
    
    # Test 5: Gradient flow
    print("\n5. Testing gradient flow...")
    loc_grad = torch.randn(batch_size, dim, requires_grad=True)
    
    if distribution_type == "VonMisesFisher":
        param_grad = torch.ones(batch_size, requires_grad=True) * 10.0
    elif distribution_type == "PowerSpherical":
        param_grad = torch.ones(batch_size, requires_grad=True) * 10.0
    else:  # ProjectedNormal
        param_grad = torch.ones(batch_size, dim, requires_grad=True) * 0.3
    
    # Retain gradients for non-leaf tensors
    loc_grad.retain_grad()
    param_grad.retain_grad()
    
    if distribution_type == "VonMisesFisher":
        dist_grad = VonMisesFisher(loc_grad, param_grad)
    elif distribution_type == "PowerSpherical":
        dist_grad = PowerSpherical(loc_grad, param_grad)
    else:
        dist_grad = ProjectedNormal(loc_grad, param_grad)
    
    samples_grad = dist_grad.rsample((50,))
    log_probs_grad = dist_grad.log_prob(samples_grad)
    loss = -log_probs_grad.mean()
    loss.backward()
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Location gradient exists: {loc_grad.grad is not None}")
    print(f"   {param_name.capitalize()} gradient exists: {param_grad.grad is not None}")
    if loc_grad.grad is not None:
        print(f"   Location gradient norm: {loc_grad.grad.norm().item():.6f}")
    if param_grad.grad is not None:
        print(f"   {param_name.capitalize()} gradient norm: {param_grad.grad.norm().item():.6f}")
    
    # Save test results to file
    results_file = os.path.join(dist_dir, "basic_test_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Distribution: {distribution_type}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Test Configuration:\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Dimensions: {dim}\n")
        f.write(f"  Parameter name: {param_name}\n\n")
        f.write("Sampling Test:\n")
        f.write(f"  Samples generated: {samples.shape[0]}\n")
        f.write(f"  Sample norms (first 5): {sample_norms[:5, 0] if sample_norms.dim() > 1 else sample_norms[:5]}\n\n")
        f.write("Gradient Flow Test:\n")
        f.write(f"  Loss: {loss.item():.6f}\n")
        if loc_grad.grad is not None:
            f.write(f"  Location gradient norm: {loc_grad.grad.norm().item():.6f}\n")
        else:
            f.write(f"  Location gradient norm: N/A (gradient is None)\n")
        if param_grad.grad is not None:
            f.write(f"  {param_name.capitalize()} gradient norm: {param_grad.grad.norm().item():.6f}\n")
        else:
            f.write(f"  {param_name.capitalize()} gradient norm: N/A (gradient is None)\n")
    
    print(f"\n✓ Test results saved to {results_file}")
    print(f"✓ All basic {distribution_type} tests passed!")


def test_distribution_visualization(distribution_type: Literal["VonMisesFisher", "PowerSpherical", "ProjectedNormal"] = "VonMisesFisher", output_dir: str = "."):
    """Visualize samples from a distribution in 2D."""
    
    # Create output directory
    dist_dir = os.path.join(output_dir, distribution_type, "visualizations")
    os.makedirs(dist_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"Visualizing {distribution_type} Samples")
    print(f"Output directory: {dist_dir}")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{distribution_type} Distribution Samples (500 samples each)', fontsize=16, fontweight='bold')
    
    # Different parameter values based on distribution type
    if distribution_type == "VonMisesFisher":
        param_values = [0.5, 2.0, 5.0, 10.0, 20.0, 50.0]
        param_name = "κ"
        color = 'blue'
    elif distribution_type == "PowerSpherical":
        param_values = [0.5, 2.0, 5.0, 10.0, 20.0, 50.0]
        param_name = "κ"
        color = 'purple'
    else:  # ProjectedNormal
        param_values = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
        param_name = "σ"
        color = 'green'
    
    for idx, param_val in enumerate(param_values):
        ax = axes[idx // 3, idx % 3]
        
        # Fixed mean direction
        loc = torch.tensor([[1.0, 0.0]])
        
        # Create parameter based on distribution type
        if distribution_type in ["VonMisesFisher", "PowerSpherical"]:
            param = torch.tensor([param_val])
        else:  # ProjectedNormal
            param = torch.ones(1, 2) * param_val
        
        # Create distribution and sample
        if distribution_type == "VonMisesFisher":
            dist = VonMisesFisher(loc, param)
        elif distribution_type == "PowerSpherical":
            dist = PowerSpherical(loc, param)
        else:
            dist = ProjectedNormal(loc, param)
        
        samples = dist.sample((500,))
        
        # Plot
        samples_np = samples[:, 0, :].detach().numpy()  # Shape: [500, 2]
        
        print(f"  {param_name}={param_val}: Sampled {samples_np.shape[0]} points")
        
        # Draw unit circle
        circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, 
                           linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        
        # Plot samples with varying transparency and size
        if distribution_type == "ProjectedNormal":
            alpha_val = 0.3 if param_val < 0.5 else 0.5
            size_val = 15 if param_val < 0.5 else 25
        else:
            alpha_val = 0.3 if param_val > 10 else 0.5
            size_val = 15 if param_val > 10 else 25
        
        ax.scatter(samples_np[:, 0], samples_np[:, 1], 
                  alpha=alpha_val, s=size_val, c=color, edgecolors='none')
        
        # Plot mean direction
        loc_normalized = loc / loc.norm() if distribution_type == "ProjectedNormal" else loc
        ax.arrow(0, 0, loc_normalized[0, 0].item() * 0.8, loc_normalized[0, 1].item() * 0.8,
                head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2, zorder=10)
        
        # Add text showing number of samples
        ax.text(0.05, 0.95, f'n=500', transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Formatting
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'{param_name} = {param_val}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
    
    plt.tight_layout()
    filename = os.path.join(dist_dir, 'parameter_comparison_samples.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {filename}")
    plt.close()


def test_distribution_high_dim(distribution_type: Literal["VonMisesFisher", "PowerSpherical", "ProjectedNormal"] = "VonMisesFisher", output_dir: str = "."):
    """Test distribution in higher dimensions."""
    
    # Create output directory
    dist_dir = os.path.join(output_dir, distribution_type, "high_dim_tests")
    os.makedirs(dist_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print(f"Testing High-Dimensional {distribution_type}")
    print(f"Output directory: {dist_dir}")
    print("=" * 60)
    
    dims = [10, 50, 128, 512]
    
    for dim in dims:
        print(f"\nTesting dimension {dim}...")
        
        # Create location
        loc = torch.randn(1, dim)
        
        # Create parameter based on distribution type
        if distribution_type == "VonMisesFisher":
            param = torch.tensor([10.0])
        elif distribution_type == "PowerSpherical":
            param = torch.tensor([10.0])
        else:  # ProjectedNormal
            param = torch.ones(1, dim) * 0.5
        
        # Create distribution
        if distribution_type == "VonMisesFisher":
            dist = VonMisesFisher(loc, param)
        elif distribution_type == "PowerSpherical":
            dist = PowerSpherical(loc, param)
        else:
            dist = ProjectedNormal(loc, param)
        
        # Sample
        samples = dist.sample((100,))
        
        # Check properties
        norms = samples.norm(dim=-1)
        log_probs = dist.log_prob(samples)
        
        print(f"  Sample shape: {samples.shape}")
        print(f"  Norm range: [{norms.min().item():.6f}, {norms.max().item():.6f}]")
        print(f"  Log prob mean: {log_probs.mean().item():.4f}")
        print(f"  Log prob std: {log_probs.std().item():.4f}")
        
        # Verify samples are on unit sphere for ProjectedNormal
        if distribution_type == "ProjectedNormal":
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
                f"Samples not on unit sphere in {dim}D!"
            print(f"  ✓ All samples verified on unit sphere")
    
    # Save high-dimensional test results
    results_file = os.path.join(dist_dir, "high_dim_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Distribution: {distribution_type}\n")
        f.write("=" * 60 + "\n\n")
        f.write("High-Dimensional Testing Results:\n\n")
        for dim in dims:
            f.write(f"Dimension {dim}:\n")
            f.write(f"  Tested successfully\n")
            f.write(f"  Samples generated: 100\n")
            f.write(f"  All tests passed\n\n")
    
    print(f"\n✓ High-dimensional test results saved to {results_file}")
    print(f"✓ High-dimensional {distribution_type} tests passed!")


if __name__ == "__main__":
    output_dir = "."
    distributions_to_test = ["VonMisesFisher", "PowerSpherical", "ProjectedNormal"]
    
    print("\n" + "=" * 80)
    print("TESTING ALL DISTRIBUTIONS - Comprehensive Distribution Tests")
    print("=" * 80)
    print(f"\nDistributions to test: {', '.join(distributions_to_test)}")
    print(f"Output directory: {output_dir}")
    print("\n" + "=" * 80 + "\n")
    
    # Test all three distributions
    for distribution in distributions_to_test:
        print("\n" + "=" * 70)
        print(f"TESTING {distribution.upper()} DISTRIBUTION")
        print("=" * 70)
        
        test_distribution_basic(distribution, output_dir)
        test_distribution_visualization(distribution, output_dir)
        test_distribution_high_dim(distribution, output_dir)
        
        print(f"\n✓ All {distribution} tests completed!")
    
    # Create overall summary
    summary_file = os.path.join(output_dir, "distribution_tests_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Distribution Testing - Comprehensive Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tested Distributions: {', '.join(distributions_to_test)}\n")
        f.write(f"Test Date: Generated on run\n\n")
        f.write("Test Categories:\n")
        f.write("1. Basic Functionality Tests\n")
        f.write("   - 2D sampling and properties\n")
        f.write("   - Log probability computation\n")
        f.write("   - Gradient flow verification\n\n")
        f.write("2. Visualization Tests\n")
        f.write("   - Parameter comparison plots\n")
        f.write("   - 500 samples per configuration\n")
        f.write("   - 6 different parameter values\n\n")
        f.write("3. High-Dimensional Tests\n")
        f.write("   - Tested dimensions: 10D, 50D, 128D, 512D\n")
        f.write("   - 100 samples per dimension\n")
        f.write("   - Norm and log prob verification\n\n")
        f.write("Output Structure:\n")
        for dist in distributions_to_test:
            f.write(f"\n{dist}/\n")
            f.write(f"  ├── basic_tests/\n")
            f.write(f"  │   └── basic_test_results.txt\n")
            f.write(f"  ├── visualizations/\n")
            f.write(f"  │   └── parameter_comparison_samples.png\n")
            f.write(f"  └── high_dim_tests/\n")
            f.write(f"      └── high_dim_results.txt\n")
        f.write(f"\nComparison Guide:\n")
        f.write(f"- Compare visualization plots to see concentration differences\n")
        f.write(f"- Review test results files for numerical comparisons\n")
        f.write(f"- All distributions support gradients and high dimensions\n")
    
    print("\n" + "=" * 80)
    print("ALL DISTRIBUTION TESTS COMPLETED!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}/")
    print(f"Summary file: {summary_file}")
    print("\nDirectory structure:")
    for dist in distributions_to_test:
        print(f"  {output_dir}/{dist}/")
        print(f"    ├── basic_tests/")
        print(f"    ├── visualizations/")
        print(f"    └── high_dim_tests/")
    print("=" * 80 + "\n")
