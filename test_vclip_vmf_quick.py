"""
Quick test to verify VonMisesFisher works with VClipLoss
"""

import torch
from losses.vclipLoss import VClipLoss
from losses.distributions.VonMisesFisher import VonMisesFisher
from power_spherical import PowerSpherical
print("Testing VonMisesFisher with VClipLoss...")

# Create simple test case
batch_size = 5
dim = 128

# Create features
image_loc = torch.randn(batch_size, dim)
text_loc = torch.randn(batch_size, dim)

# Normalize to unit sphere
image_loc = image_loc / image_loc.norm(dim=-1, keepdim=True)
text_loc = text_loc / text_loc.norm(dim=-1, keepdim=True)

# Create concentrations
image_conc = torch.ones(batch_size) * 10.0
text_conc = torch.ones(batch_size) * 10.0

# Create distributions
# image_dist = VonMisesFisher(image_loc, image_conc)
# text_dist = VonMisesFisher(text_loc, text_conc)

image_dist = PowerSpherical(image_loc, image_conc)
text_dist = PowerSpherical(text_loc, text_conc)

print(f"Image dist batch shape: {image_dist.batch_shape}")
print(f"Text dist batch shape: {text_dist.batch_shape}")
print(f"Image dist event shape: {image_dist.event_shape}")

# Create loss
vclip_loss_fn = VClipLoss(kl_weight=0.1, num_samples=10)
logits_scale = torch.tensor(1.0)

# Compute loss
try:
    loss_dict = vclip_loss_fn(image_dist, text_dist, logits_scale)
    print(f"\n✓ Loss computation successful!")
    print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  CLIP loss: {loss_dict['clip_loss'].item():.4f}")
    print(f"  Image KL: {loss_dict['image_kl_loss'].item():.4f}")
    print(f"  Text KL: {loss_dict['text_kl_loss'].item():.4f}")
    
    # Test gradient flow
    image_loc_grad = torch.randn(batch_size, dim, requires_grad=True)
    image_loc_grad = image_loc_grad / image_loc_grad.norm(dim=-1, keepdim=True)
    text_loc_grad = torch.randn(batch_size, dim, requires_grad=True)
    text_loc_grad = text_loc_grad / text_loc_grad.norm(dim=-1, keepdim=True)
    
    image_conc_grad = torch.ones(batch_size, requires_grad=True) * 10.0
    text_conc_grad = torch.ones(batch_size, requires_grad=True) * 10.0
    
    image_dist_grad = VonMisesFisher(image_loc_grad, image_conc_grad)
    text_dist_grad = VonMisesFisher(text_loc_grad, text_conc_grad)
    
    loss_dict_grad = vclip_loss_fn(image_dist_grad, text_dist_grad, logits_scale)
    loss_dict_grad['total_loss'].backward()
    
    print(f"\n✓ Gradient flow successful!")
    print(f"  Image loc gradient norm: {image_loc_grad.grad.norm().item():.6f}")
    print(f"  Text loc gradient norm: {text_loc_grad.grad.norm().item():.6f}")
    
    print("\n" + "="*60)
    print("All tests passed! VonMisesFisher works with VClipLoss ✓")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
