"""
A synthetic test for vclipLoss module.
It creates random tensors close together and backprops through the loss, 
plotting the gradients to ensure the KL term makes the variance increase
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
from losses.vclipLoss import VClipLoss
from power_spherical import PowerSpherical
import numpy as np
from PIL import Image
import io
import os


def riemannian_gradient(x, euclidean_grad):
    """
    Project Euclidean gradient onto the tangent space of the unit sphere.
    This ensures the gradient update stays on the manifold.
    
    For the unit sphere, the tangent space projection is:
    grad_tangent = grad - (grad · x) * x
    
    Args:
        x: Points on the unit sphere [batch_size, dim]
        euclidean_grad: Euclidean gradient [batch_size, dim]
    
    Returns:
        Riemannian gradient projected onto tangent space
    """
    # Compute dot product: (grad · x)
    dot_product = (euclidean_grad * x).sum(dim=-1, keepdim=True)
    # Project: grad - (grad · x) * x
    riemannian_grad = euclidean_grad - dot_product * x
    return riemannian_grad


def exponential_map(x, tangent_vector):
    """
    Exponential map: moves along the geodesic on the sphere.
    For the unit sphere, this is equivalent to:
    1. Move in the tangent direction
    2. Normalize back to unit sphere
    
    Args:
        x: Points on the unit sphere [batch_size, dim]
        tangent_vector: Tangent vector (Riemannian gradient * step_size) [batch_size, dim]
    
    Returns:
        New points on the unit sphere
    """
    # Move in tangent direction
    new_x = x + tangent_vector
    # Project back to sphere (normalize)
    new_x = new_x / new_x.norm(p=2, dim=-1, keepdim=True)
    return new_x


def compute_arc_length(x1, x2):
    """
    Compute the geodesic distance (arc length) between points on the unit sphere.
    
    For points on the unit sphere, the arc length is:
    d(x1, x2) = arccos(x1 · x2)
    
    Args:
        x1: Points on unit sphere [batch_size, dim]
        x2: Points on unit sphere [batch_size, dim]
    
    Returns:
        Arc lengths [batch_size]
    """
    # Compute dot product
    dot_product = (x1 * x2).sum(dim=-1)
    # Clamp to avoid numerical issues with arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    # Arc length is the angle between the vectors
    arc_length = torch.acos(dot_product)
    return arc_length


def create_snapshot_frame(og_features_a, og_features_b, features_a, features_b, 
                         epoch, total_epochs, batch_size, avg_arc_length):
    """
    Create a single frame for the animation showing current state of features.
    
    Args:
        og_features_a, og_features_b: Original features
        features_a, features_b: Current features
        epoch: Current epoch number
        total_epochs: Total number of epochs
        batch_size: Number of samples
        avg_arc_length: Average arc length between pairs
    
    Returns:
        PIL Image of the frame
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw unit circle to show the constraint
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', 
                        linewidth=2, alpha=0.5, label='Unit Sphere')
    ax.add_patch(circle)
    
    # Plot original positions (faded)
    ax.scatter(og_features_a[:,0], og_features_a[:,1], 
              label='Start A', color='blue', alpha=0.2, s=40, 
              edgecolors='darkblue', linewidths=0.5, marker='o')
    ax.scatter(og_features_b[:,0], og_features_b[:,1], 
              label='Start B', color='orange', alpha=0.2, s=40, 
              edgecolors='darkorange', linewidths=0.5, marker='o')
    
    # Plot current positions (bright)
    ax.scatter(features_a[:,0], features_a[:,1], 
              label='Current A', color='cyan', alpha=0.7, s=80, 
              edgecolors='darkturquoise', linewidths=1.5, marker='*')
    ax.scatter(features_b[:,0], features_b[:,1], 
              label='Current B', color='green', alpha=0.7, s=80, 
              edgecolors='darkgreen', linewidths=1.5, marker='*')
    
    # Draw lines connecting pairs
    for i in range(min(batch_size, len(features_a))):
        ax.plot([features_a[i,0], features_b[i,0]], 
               [features_a[i,1], features_b[i,1]], 
               color='purple', linestyle=':', linewidth=1, alpha=0.4)
    
    # Draw trajectory lines from start to current
    for i in range(min(batch_size, len(features_a))):
        ax.plot([og_features_a[i,0], features_a[i,0]],
               [og_features_a[i,1], features_a[i,1]], 
               color='gray', linestyle='--', linewidth=0.8, alpha=0.3)
        ax.plot([og_features_b[i,0], features_b[i,0]],
               [og_features_b[i,1], features_b[i,1]], 
               color='gray', linestyle='--', linewidth=0.8, alpha=0.3)
    
    # Plot origin
    ax.scatter(0, 0, color='red', marker='x', s=150, linewidths=3, zorder=10)
    
    # Formatting
    ax.axis('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_title(f'Epoch {epoch}/{total_epochs}\nAvg Arc Length: {avg_arc_length:.4f} rad ({avg_arc_length * 180 / np.pi:.2f}°)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add progress bar
    progress = epoch / total_epochs
    ax.axhline(y=-1.4, xmin=0.05, xmax=0.05 + 0.9 * progress, 
              color='green', linewidth=8, alpha=0.7)
    ax.text(0, -1.45, f'{progress*100:.1f}%', ha='center', va='top', 
           fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    
    return image

def test_vclip_loss(distribution: Literal["PowerSpherical", "VonMisesFisher", "ProjectedNormal"] = "PowerSpherical", output_dir: str = "."):
    torch.manual_seed(42)
    batch_size = 10
    feature_dim = 2
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    dist_dir = os.path.join(output_dir, distribution)
    os.makedirs(dist_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Testing with {distribution}")
    print(f"Output directory: {dist_dir}")
    print(f"{'='*60}\n")

    
    # create two sets of features all in a small range of the hypersphere
    base_features = torch.cat([(torch.randn(batch_size, feature_dim)/10.0+3), (torch.randn(batch_size, feature_dim)/10.0-3)], dim=0)
    features_a = base_features + torch.randn(batch_size*2, feature_dim)/50.0
    features_b = base_features + torch.randn(batch_size*2, feature_dim)/50.0
    features_a /= features_a.norm(p=2, dim=-1, keepdim=True)
    features_b /= features_b.norm(p=2, dim=-1, keepdim=True)

    print ("Feature A norms:", features_a.norm(p=2, dim=-1, keepdim=True).shape, features_a.shape)
    print(f"Using distribution: {distribution}")

    # PowerSpherical expects concentration shape [batch] (1D)
    # VonMisesFisher expects concentration shape [batch, 1] (2D)
    # ProjectedNormal expects sigma shape [batch, dim] (same as features)
    if distribution == "PowerSpherical":
        concentrations_a = torch.ones(batch_size*2) * 0.1
        concentrations_b = torch.ones(batch_size*2) * 0.1
    elif distribution == "VonMisesFisher":
        concentrations_a = torch.ones(batch_size*2, 1) * 0.1
        concentrations_b = torch.ones(batch_size*2, 1) * 0.1
    else:  # ProjectedNormal
        concentrations_a = torch.ones(batch_size*2, feature_dim) * 0.1
        concentrations_b = torch.ones(batch_size*2, feature_dim) * 0.1

    og_features_a = features_a.clone().detach()
    og_features_b = features_b.clone().detach()

    og_concentrations_a = concentrations_a.clone().detach()
    og_concentrations_b = concentrations_b.clone().detach()

    # Compute initial arc length between pairs
    initial_arc_lengths = compute_arc_length(og_features_a, og_features_b)
    print(f"\n=== Initial Statistics ===")
    print(f"Average arc length between pairs: {initial_arc_lengths.mean().item():.6f} radians ({initial_arc_lengths.mean().item() * 180 / 3.14159:.2f} degrees)")
    print(f"Min arc length: {initial_arc_lengths.min().item():.6f} radians")
    print(f"Max arc length: {initial_arc_lengths.max().item():.6f} radians")
    print(f"Std arc length: {initial_arc_lengths.std().item():.6f} radians\n")

    features_a.requires_grad_()
    features_b.requires_grad_()

    concentrations_a.requires_grad_()
    concentrations_b.requires_grad_()

    logits_scale = torch.tensor(1.0)
    logits_scale.requires_grad_()

    # Learning rate for features (on manifold) and concentrations (in Euclidean space)
    lr_features = 0.5
    lr_concentrations = 100.0

    # Initialize VClipLoss
    vclip_loss_fn = VClipLoss(kl_weight=0.1)

    # Storage for snapshots
    snapshot_frames = []
    snapshot_interval = 10  # Take snapshot every N epochs
    total_epochs = 1000
    
    # Storage for tracking metrics
    metrics_history = {
        'total_loss': [],
        'contrastive_loss': [],
        'kl_loss': [],
        'feature_a_grad_norm': [],
        'feature_b_grad_norm': [],
        'concentration_a_grad_norm': [],
        'concentration_b_grad_norm': []
    }

    for epoch in range(total_epochs):
        # Zero gradients manually
        if features_a.grad is not None:
            features_a.grad.zero_()
        if features_b.grad is not None:
            features_b.grad.zero_()
        if concentrations_a.grad is not None:
            concentrations_a.grad.zero_()
        if concentrations_b.grad is not None:
            concentrations_b.grad.zero_()
        if logits_scale.grad is not None:
            logits_scale.grad.zero_()
        
        # Create distributions based on the flag
        if distribution == "PowerSpherical":
            dist_a = PowerSpherical(features_a, concentrations_a)
            dist_b = PowerSpherical(features_b, concentrations_b)
        elif distribution == "VonMisesFisher":
            dist_a = VonMisesFisher(features_a, concentrations_a)
            dist_b = VonMisesFisher(features_b, concentrations_b)
        elif distribution == "ProjectedNormal":
            dist_a = ProjectedNormal(features_a, concentrations_a)
            dist_b = ProjectedNormal(features_b, concentrations_b)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution}")

        # Compute loss 
        with torch.enable_grad():
            loss = vclip_loss_fn(
                image_distribution=dist_a,
                text_distribution=dist_b,
                logits_scale=logits_scale
            )

        # Backpropagate
        loss['total_loss'].backward()
        
        # Store metrics for plotting
        metrics_history['total_loss'].append(loss['total_loss'].item())
        metrics_history['contrastive_loss'].append(loss['clip_loss'].item())
        print(loss.keys())
        metrics_history['kl_loss'].append(loss['image_kl_loss'].item() + loss['text_kl_loss'].item())
        metrics_history['feature_a_grad_norm'].append(features_a.grad.norm().item())
        metrics_history['feature_b_grad_norm'].append(features_b.grad.norm().item())
        metrics_history['concentration_a_grad_norm'].append(concentrations_a.grad.norm().item())
        metrics_history['concentration_b_grad_norm'].append(concentrations_b.grad.norm().item())
        
        # magnitude of gradient going to features and concentrations
        print(f"Epoch {epoch}: Feature A grad norm: {features_a.grad.norm().item():.6f}, Concentration A grad norm: {concentrations_a.grad.norm().item():.6f}")
        print(f"loss: {loss['total_loss'].item()}")
        
        # Update features using Riemannian gradient descent
        with torch.no_grad():
            # Project Euclidean gradient to tangent space
            riem_grad_a = riemannian_gradient(features_a, features_a.grad)
            riem_grad_b = riemannian_gradient(features_b, features_b.grad)
            
            # Move along geodesic using exponential map
            features_a.data = exponential_map(features_a, -lr_features * riem_grad_a)
            features_b.data = exponential_map(features_b, -lr_features * riem_grad_b)
            
            # Update concentrations (standard Euclidean gradient descent)
            concentrations_a.data -= lr_concentrations * concentrations_a.grad
            concentrations_b.data -= lr_concentrations * concentrations_b.grad
            
            # Clamp concentrations to be positive
            concentrations_a.data.clamp_(min=0.00001)
            concentrations_b.data.clamp_(min=0.00001)
        
        # Take snapshot every N epochs
        if epoch % snapshot_interval == 0 or epoch == total_epochs - 1:
            current_arc_lengths = compute_arc_length(features_a, features_b)
            avg_arc = current_arc_lengths.mean().item()
            
            frame = create_snapshot_frame(
                og_features_a.cpu().numpy(), 
                og_features_b.cpu().numpy(),
                features_a.detach().cpu().numpy(), 
                features_b.detach().cpu().numpy(),
                epoch, total_epochs, batch_size, avg_arc
            )
            snapshot_frames.append(frame)
            print(f"  Snapshot saved at epoch {epoch}")
        
        # Verify features are still on unit sphere
        if epoch % 20 == 0:
            print(f"  Feature A norms: min={features_a.norm(p=2, dim=-1).min():.6f}, max={features_a.norm(p=2, dim=-1).max():.6f}")
            print(f"  Feature B norms: min={features_b.norm(p=2, dim=-1).min():.6f}, max={features_b.norm(p=2, dim=-1).max():.6f}")
            
            # Print current arc length statistics
            current_arc_lengths = compute_arc_length(features_a, features_b)
            print(f"  Average arc length between pairs: {current_arc_lengths.mean().item():.6f} radians ({current_arc_lengths.mean().item() * 180 / 3.14159:.2f} degrees)")

    # Compute final arc length between pairs
    final_arc_lengths = compute_arc_length(features_a, features_b)
    print(f"\n=== Final Statistics ===")
    print(f"Average arc length between pairs: {final_arc_lengths.mean().item():.6f} radians ({final_arc_lengths.mean().item() * 180 / 3.14159:.2f} degrees)")
    print(f"Min arc length: {final_arc_lengths.min().item():.6f} radians")
    print(f"Max arc length: {final_arc_lengths.max().item():.6f} radians")
    print(f"Std arc length: {final_arc_lengths.std().item():.6f} radians")
    print(f"\nChange in average arc length: {(final_arc_lengths.mean() - initial_arc_lengths.mean()).item():.6f} radians")
    print(f"  ({(final_arc_lengths.mean() - initial_arc_lengths.mean()).item() * 180 / 3.14159:.2f} degrees)\n")

    # Create animated GIF from snapshots
    if len(snapshot_frames) > 0:
        print(f"\n=== Creating Animation ===")
        print(f"Total frames: {len(snapshot_frames)}")
        
        # Save as GIF
        gif_path = os.path.join(dist_dir, 'training_evolution.gif')
        snapshot_frames[0].save(
            gif_path,
            save_all=True,
            append_images=snapshot_frames[1:],
            duration=200,  # milliseconds per frame
            loop=0  # loop forever
        )
        print(f"✓ Animation saved to {gif_path}")
        
        # Also save a higher quality version with slower playback
        gif_path_slow = os.path.join(dist_dir, 'training_evolution_slow.gif')
        snapshot_frames[0].save(
            gif_path_slow,
            save_all=True,
            append_images=snapshot_frames[1:],
            duration=500,  # slower playback
            loop=0
        )
        print(f"✓ Slow animation saved to {gif_path_slow}\n")
    else:
        print("Warning: No snapshots were captured!")

    # Plot training metrics
    print(f"\n=== Creating Training Metrics Plots ===")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs_range = range(len(metrics_history['total_loss']))
    
    # Plot 1: Loss components over time
    ax1 = axes[0, 0]
    ax1.plot(epochs_range, metrics_history['total_loss'], label='Total Loss', linewidth=2, color='black')
    ax1.plot(epochs_range, metrics_history['contrastive_loss'], label='Contrastive Loss', linewidth=1.5, color='blue', alpha=0.7)
    ax1.plot(epochs_range, metrics_history['kl_loss'], label='KL Loss', linewidth=1.5, color='red', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Components Over Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Feature gradient norms
    ax2 = axes[0, 1]
    ax2.plot(epochs_range, metrics_history['feature_a_grad_norm'], label='Feature A Grad Norm', linewidth=1.5, color='cyan', alpha=0.8)
    ax2.plot(epochs_range, metrics_history['feature_b_grad_norm'], label='Feature B Grad Norm', linewidth=1.5, color='green', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Gradient Norm', fontsize=12)
    ax2.set_title('Feature Gradient Norms Over Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Concentration/Sigma gradient norms
    ax3 = axes[1, 0]
    ax3.plot(epochs_range, metrics_history['concentration_a_grad_norm'], label='Concentration A Grad Norm', linewidth=1.5, color='orange', alpha=0.8)
    ax3.plot(epochs_range, metrics_history['concentration_b_grad_norm'], label='Concentration B Grad Norm', linewidth=1.5, color='purple', alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Gradient Norm', fontsize=12)
    ax3.set_title('Concentration/Sigma Gradient Norms Over Training', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss ratio (contrastive vs KL)
    ax4 = axes[1, 1]
    loss_ratio = np.array(metrics_history['contrastive_loss']) / (np.array(metrics_history['kl_loss']) + 1e-8)
    ax4.plot(epochs_range, loss_ratio, label='Contrastive / KL Ratio', linewidth=2, color='darkviolet')
    ax4.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Equal Contribution')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss Ratio', fontsize=12)
    ax4.set_title('Contrastive to KL Loss Ratio', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(dist_dir, 'training_metrics.png')
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training metrics plot saved to {metrics_plot_path}")
    plt.close()


    # find the range of the angles and concentrations before and after the update
    angles_og_a = torch.arctan2(og_features_a[:,1], og_features_a[:,0])
    angles_og_b = torch.arctan2(og_features_b[:,1], og_features_b[:,0])
    angles_a = torch.arctan2(features_a[:,1], features_a[:,0])
    angles_b = torch.arctan2(features_b[:,1], features_b[:,0])

    # plot the angles before and after the update in a histogram (transparent bars)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(angles_og_a.detach().numpy().flatten(), bins=30, alpha=0.5, label='Original Angles A', color='blue')
    plt.hist(angles_a.detach().numpy().flatten(), bins=30, alpha=0.5, label='Updated Angles A', color='cyan')
    plt.legend()
    plt.title('Angles A Before and After Update')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Count')

    plt.subplot(1, 3, 2)
    plt.hist(angles_og_b.detach().numpy().flatten(), bins=30, alpha=0.5, label='Original Angles B', color='orange')
    plt.hist(angles_b.detach().numpy().flatten(), bins=30, alpha=0.5, label='Updated Angles B', color='green')
    plt.legend()
    plt.title('Angles B Before and After Update')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 3)
    plt.hist(initial_arc_lengths.detach().numpy().flatten(), bins=30, alpha=0.5, label='Initial Arc Lengths', color='purple')
    plt.hist(final_arc_lengths.detach().numpy().flatten(), bins=30, alpha=0.5, label='Final Arc Lengths', color='red')
    plt.axvline(initial_arc_lengths.mean().item(), color='purple', linestyle='--', linewidth=2, label=f'Initial Mean: {initial_arc_lengths.mean().item():.4f}')
    plt.axvline(final_arc_lengths.mean().item(), color='red', linestyle='--', linewidth=2, label=f'Final Mean: {final_arc_lengths.mean().item():.4f}')
    plt.legend()
    plt.title('Arc Length Between Pairs')
    plt.xlabel('Arc Length (radians)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    angles_plot_path = os.path.join(dist_dir, 'angle_histograms.png')
    plt.savefig(angles_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Angle histograms saved to {angles_plot_path}")
    plt.close()

    # plot the vectors before the update
    plt.figure(figsize=(10, 10))
    
    # Draw unit circle to show the constraint
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', linewidth=2, alpha=0.5, label='Unit Sphere')
    plt.gca().add_patch(circle)
    
    plt.scatter(og_features_a[:,0].detach().numpy(), og_features_a[:,1].detach().numpy(), label='Original Features A', color='blue', alpha=0.4, s=30, edgecolors='darkblue', linewidths=0.5)
    plt.scatter(og_features_b[:,0].detach().numpy(), og_features_b[:,1].detach().numpy(), label='Original Features B', color='orange', alpha=0.4, s=30, edgecolors='darkorange', linewidths=0.5)
    plt.scatter(features_a[:,0].detach().numpy(), features_a[:,1].detach().numpy(), label='Updated Features A', color='cyan', alpha=0.5, s=40, edgecolors='darkturquoise', linewidths=0.5)
    plt.scatter(features_b[:,0].detach().numpy(), features_b[:,1].detach().numpy(), label='Updated Features B', color='green', alpha=0.5, s=40, edgecolors='darkgreen', linewidths=0.5)

    # plot lines between original and updated features
    for i in range(batch_size):
        plt.plot([og_features_a[i,0].detach().numpy(), features_a[i,0].detach().numpy()],
                 [og_features_a[i,1].detach().numpy(), features_a[i,1].detach().numpy()], color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.plot([og_features_b[i,0].detach().numpy(), features_b[i,0].detach().numpy()],
                 [og_features_b[i,1].detach().numpy(), features_b[i,1].detach().numpy()], color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # plot 0,0 for reference
    plt.scatter(0, 0, label='Origin', color='red', marker='x', s=100, linewidths=2)
    plt.axis('equal')  # Ensure circle looks like a circle
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.title('Feature Vectors Before and After Update (Constrained to Unit Sphere)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    features_plot_path = os.path.join(dist_dir, 'feature_trajectories.png')
    plt.savefig(features_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature trajectories saved to {features_plot_path}")
    plt.close()

    # Save summary statistics to text file
    summary_path = os.path.join(dist_dir, 'summary_statistics.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Distribution: {distribution}\n")
        f.write(f"{'='*60}\n\n")
        f.write("Concentration/Sigma Statistics:\n")
        f.write(f"  Average Concentration A before: {og_concentrations_a.mean().item():.6f}\n")
        f.write(f"  Average Concentration A after: {concentrations_a.mean().item():.6f}\n")
        f.write(f"  Average Concentration B before: {og_concentrations_b.mean().item():.6f}\n")
        f.write(f"  Average Concentration B after: {concentrations_b.mean().item():.6f}\n")
        f.write(f"\nArc Length Statistics:\n")
        f.write(f"  Initial average: {initial_arc_lengths.mean().item():.6f} radians ({initial_arc_lengths.mean().item() * 180 / 3.14159:.2f}°)\n")
        f.write(f"  Final average: {final_arc_lengths.mean().item():.6f} radians ({final_arc_lengths.mean().item() * 180 / 3.14159:.2f}°)\n")
        f.write(f"  Change: {(final_arc_lengths.mean() - initial_arc_lengths.mean()).item():.6f} radians ({(final_arc_lengths.mean() - initial_arc_lengths.mean()).item() * 180 / 3.14159:.2f}°)\n")
        f.write(f"\nFinal Loss Values:\n")
        f.write(f"  Total Loss: {metrics_history['total_loss'][-1]:.6f}\n")
        f.write(f"  Contrastive Loss: {metrics_history['contrastive_loss'][-1]:.6f}\n")
        f.write(f"  KL Loss: {metrics_history['kl_loss'][-1]:.6f}\n")
    
    print(f"✓ Summary statistics saved to {summary_path}")
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {dist_dir}")
    print(f"{'='*60}\n")

    # plot the concentrations before and after the update in a bar chart
    print("Average Concentration A before:", og_concentrations_a.mean().item())
    print("Average Concentration A after:", concentrations_a.mean().item())
    print("Average Concentration B before:", og_concentrations_b.mean().item())
    print("Average Concentration B after:", concentrations_b.mean().item())


if __name__ == "__main__":
    output_dir = "."
    distributions_to_test = ["PowerSpherical", "VonMisesFisher", "ProjectedNormal"]
    
    print("\n" + "=" * 80)
    print("TESTING ALL DISTRIBUTIONS - VClip Loss Comparison")
    print("=" * 80)
    print(f"\nDistributions to test: {', '.join(distributions_to_test)}")
    print(f"Output directory: {output_dir}")
    print("\n" + "=" * 80 + "\n")
    
    # Test all three distributions
    for distribution in distributions_to_test:
        test_vclip_loss(distribution, output_dir)
    
    # Create a comparison summary
    summary_file = os.path.join(output_dir, "comparison_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("VClip Loss - Distribution Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tested Distributions: {', '.join(distributions_to_test)}\n")
        f.write(f"Total Epochs: 1000\n")
        f.write(f"Batch Size: 10 (doubled for pairs)\n")
        f.write(f"Feature Dimension: 2D\n")
        f.write(f"KL Weight: 0.1\n")
        f.write(f"\nOutput Structure:\n")
        for dist in distributions_to_test:
            f.write(f"\n{dist}/\n")
            f.write(f"  ├── training_evolution.gif (200ms per frame)\n")
            f.write(f"  ├── training_evolution_slow.gif (500ms per frame)\n")
            f.write(f"  ├── training_metrics.png (loss and gradient plots)\n")
            f.write(f"  ├── angle_histograms.png (before/after distributions)\n")
            f.write(f"  ├── feature_trajectories.png (movement on unit sphere)\n")
            f.write(f"  └── summary_statistics.txt (numerical results)\n")
        f.write(f"\nTo compare distributions:\n")
        f.write(f"1. Review training_metrics.png for each distribution\n")
        f.write(f"2. Compare convergence speed and stability\n")
        f.write(f"3. Check summary_statistics.txt for final values\n")
        f.write(f"4. Watch training animations to see dynamics\n")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED!")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}/")
    print(f"Comparison summary: {summary_file}")
    print("\nDirectory structure:")
    for dist in distributions_to_test:
        print(f"  {output_dir}/{dist}/")
    print("=" * 80 + "\n")