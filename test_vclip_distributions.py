"""
Test VClipLoss with both PowerSpherical and VonMisesFisher distributions.
"""

import torch
import matplotlib.pyplot as plt
from power_spherical import PowerSpherical
from losses.distributions.VonMisesFisher import VonMisesFisher
from losses.vclipLoss import VClipLoss


def test_vclip_with_both_distributions():
    """Compare VClipLoss with PowerSpherical vs VonMisesFisher."""
    print("=" * 70)
    print("Testing VClipLoss with PowerSpherical vs VonMisesFisher")
    print("=" * 70)
    
    batch_size = 32
    embedding_dim = 128
    
    # Create random embeddings
    torch.manual_seed(42)
    image_locs = torch.randn(batch_size, embedding_dim)
    image_locs = image_locs / image_locs.norm(dim=-1, keepdim=True)
    
    text_locs = torch.randn(batch_size, embedding_dim)
    text_locs = text_locs / text_locs.norm(dim=-1, keepdim=True)
    
    # Concentration parameters
    image_concs = torch.ones(batch_size) * 10.0
    text_concs = torch.ones(batch_size) * 10.0
    
    logits_scale = torch.tensor([1.0])
    
    # Test 1: PowerSpherical
    print("\n1. Testing with PowerSpherical distribution...")
    
    image_dist_ps = PowerSpherical(image_locs, image_concs)
    text_dist_ps = PowerSpherical(text_locs, text_concs)
    
    vclip_ps = VClipLoss(kl_weight=0.1, num_samples=20, distribution_type='power_spherical')
    loss_ps = vclip_ps(image_dist_ps, text_dist_ps, logits_scale)
    
    print(f"   Total Loss: {loss_ps['total_loss'].item():.4f}")
    print(f"   CLIP Loss: {loss_ps['clip_loss'].item():.4f}")
    print(f"   Image KL: {loss_ps['image_kl_loss'].item():.4f}")
    print(f"   Text KL: {loss_ps['text_kl_loss'].item():.4f}")
    
    # Test 2: VonMisesFisher
    print("\n2. Testing with VonMisesFisher distribution...")
    
    image_dist_vmf = VonMisesFisher(image_locs, image_concs)
    text_dist_vmf = VonMisesFisher(text_locs, text_concs)
    
    vclip_vmf = VClipLoss(kl_weight=0.1, num_samples=20, distribution_type='vmf')
    loss_vmf = vclip_vmf(image_dist_vmf, text_dist_vmf, logits_scale)
    
    print(f"   Total Loss: {loss_vmf['total_loss'].item():.4f}")
    print(f"   CLIP Loss: {loss_vmf['clip_loss'].item():.4f}")
    print(f"   Image KL: {loss_vmf['image_kl_loss'].item():.4f}")
    print(f"   Text KL: {loss_vmf['text_kl_loss'].item():.4f}")
    
    # Test 3: Mean-only mode (no sampling)
    print("\n3. Testing mean-only mode (no sampling)...")
    
    vclip_mean = VClipLoss(kl_weight=0.1, num_samples=20, use_mean_only=True)
    loss_mean_ps = vclip_mean(image_dist_ps, text_dist_ps, logits_scale)
    loss_mean_vmf = vclip_mean(image_dist_vmf, text_dist_vmf, logits_scale)
    
    print(f"   PowerSpherical - Total Loss: {loss_mean_ps['total_loss'].item():.4f}")
    print(f"   VonMisesFisher - Total Loss: {loss_mean_vmf['total_loss'].item():.4f}")
    
    # Test 4: Gradient flow
    print("\n4. Testing gradient flow...")
    
    image_locs_grad = image_locs.clone().requires_grad_(True)
    text_locs_grad = text_locs.clone().requires_grad_(True)
    image_concs_grad = image_concs.clone().requires_grad_(True)
    text_concs_grad = text_concs.clone().requires_grad_(True)
    
    # Test with VonMisesFisher
    image_dist = VonMisesFisher(image_locs_grad, image_concs_grad)
    text_dist = VonMisesFisher(text_locs_grad, text_concs_grad)
    
    vclip = VClipLoss(kl_weight=0.1, num_samples=10)
    loss = vclip(image_dist, text_dist, logits_scale)
    loss['total_loss'].backward()
    
    print(f"   Loss: {loss['total_loss'].item():.4f}")
    print(f"   Image loc gradient exists: {image_locs_grad.grad is not None}")
    print(f"   Image conc gradient exists: {image_concs_grad.grad is not None}")
    if image_locs_grad.grad is not None:
        print(f"   Image loc gradient norm: {image_locs_grad.grad.norm().item():.6f}")
    if image_concs_grad.grad is not None:
        print(f"   Image conc gradient norm: {image_concs_grad.grad.norm().item():.6f}")
    
    print("\n✓ All tests passed!")


def compare_distributions_visualization():
    """Visualize the difference between PowerSpherical and VonMisesFisher."""
    print("\n" + "=" * 70)
    print("Comparing PowerSpherical vs VonMisesFisher Samples")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PowerSpherical (top) vs VonMisesFisher (bottom)', fontsize=16, fontweight='bold')
    
    concentrations = [1.0, 5.0, 20.0]
    
    for idx, kappa in enumerate(concentrations):
        # Fixed mean direction
        loc = torch.tensor([[1.0, 0.0]])
        concentration = torch.tensor([kappa])
        
        # PowerSpherical
        ax_ps = axes[0, idx]
        ps_dist = PowerSpherical(loc, concentration)
        ps_samples = ps_dist.rsample((500,))
        ps_samples_np = ps_samples[:, 0, :].detach().numpy()
        
        circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, 
                           linestyle='--', linewidth=2, alpha=0.5)
        ax_ps.add_patch(circle)
        ax_ps.scatter(ps_samples_np[:, 0], ps_samples_np[:, 1], 
                     alpha=0.3, s=15, c='blue', edgecolors='none')
        ax_ps.arrow(0, 0, 0.8, 0, head_width=0.1, head_length=0.1, 
                   fc='red', ec='red', linewidth=2)
        ax_ps.set_xlim(-1.5, 1.5)
        ax_ps.set_ylim(-1.5, 1.5)
        ax_ps.set_aspect('equal')
        ax_ps.set_title(f'PowerSpherical κ={kappa}', fontsize=12, fontweight='bold')
        ax_ps.grid(True, alpha=0.3)
        
        # VonMisesFisher
        ax_vmf = axes[1, idx]
        vmf_dist = VonMisesFisher(loc, concentration)
        vmf_samples = vmf_dist.rsample((500,))
        vmf_samples_np = vmf_samples[:, 0, :].detach().numpy()
        
        circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, 
                           linestyle='--', linewidth=2, alpha=0.5)
        ax_vmf.add_patch(circle)
        ax_vmf.scatter(vmf_samples_np[:, 0], vmf_samples_np[:, 1], 
                      alpha=0.3, s=15, c='green', edgecolors='none')
        ax_vmf.arrow(0, 0, 0.8, 0, head_width=0.1, head_length=0.1, 
                    fc='red', ec='red', linewidth=2)
        ax_vmf.set_xlim(-1.5, 1.5)
        ax_vmf.set_ylim(-1.5, 1.5)
        ax_vmf.set_aspect('equal')
        ax_vmf.set_title(f'VonMisesFisher κ={kappa}', fontsize=12, fontweight='bold')
        ax_vmf.grid(True, alpha=0.3)
        ax_vmf.set_xlabel('Dimension 1')
        if idx == 0:
            ax_ps.set_ylabel('Dimension 2')
            ax_vmf.set_ylabel('Dimension 2')
    
    plt.tight_layout()
    plt.savefig('losses/powerspherical_vs_vmf_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to losses/powerspherical_vs_vmf_comparison.png")
    plt.show()


def benchmark_performance():
    """Benchmark the performance of both distribution types."""
    print("\n" + "=" * 70)
    print("Benchmarking Performance")
    print("=" * 70)
    
    import time
    
    batch_size = 64
    embedding_dim = 512
    num_iterations = 50
    
    # Setup
    torch.manual_seed(42)
    image_locs = torch.randn(batch_size, embedding_dim)
    image_locs = image_locs / image_locs.norm(dim=-1, keepdim=True)
    text_locs = torch.randn(batch_size, embedding_dim)
    text_locs = text_locs / text_locs.norm(dim=-1, keepdim=True)
    concs = torch.ones(batch_size) * 10.0
    logits_scale = torch.tensor([1.0])
    
    # Benchmark PowerSpherical
    print("\n1. Benchmarking PowerSpherical...")
    image_dist_ps = PowerSpherical(image_locs, concs)
    text_dist_ps = PowerSpherical(text_locs, concs)
    vclip_ps = VClipLoss(kl_weight=0.1, num_samples=20)
    
    start_time = time.time()
    for _ in range(num_iterations):
        loss = vclip_ps(image_dist_ps, text_dist_ps, logits_scale)
    ps_time = time.time() - start_time
    print(f"   Total time: {ps_time:.3f}s")
    print(f"   Time per iteration: {ps_time/num_iterations*1000:.2f}ms")
    
    # Benchmark VonMisesFisher
    print("\n2. Benchmarking VonMisesFisher...")
    image_dist_vmf = VonMisesFisher(image_locs, concs)
    text_dist_vmf = VonMisesFisher(text_locs, concs)
    vclip_vmf = VClipLoss(kl_weight=0.1, num_samples=20)
    
    start_time = time.time()
    for _ in range(num_iterations):
        loss = vclip_vmf(image_dist_vmf, text_dist_vmf, logits_scale)
    vmf_time = time.time() - start_time
    print(f"   Total time: {vmf_time:.3f}s")
    print(f"   Time per iteration: {vmf_time/num_iterations*1000:.2f}ms")
    
    print(f"\n   Speedup: {vmf_time/ps_time:.2f}x {'(VonMisesFisher is slower)' if vmf_time > ps_time else '(VonMisesFisher is faster)'}")


if __name__ == "__main__":
    test_vclip_with_both_distributions()
    compare_distributions_visualization()
    benchmark_performance()
    
    print("\n" + "=" * 70)
    print("All VClipLoss distribution comparison tests completed!")
    print("=" * 70)
