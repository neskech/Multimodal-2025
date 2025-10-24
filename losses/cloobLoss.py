# https://arxiv.org/pdf/2110.11316

from torch import nn
import torch
import sys
import os



def cloob_loss(image_features, text_features, inv_tau, scale_hopfield, device):
    """Computes the CLOOB loss (negative mean log odds assigned to positive pairs after
    Hopfield retrieval).
    
    Note: this loss has been rescaled from the original CLOOB loss for interpretability,
    to convert to the original, divide it by inv_tau / 2.
    """
    p_xx, p_yy, p_xy, p_yx = hopfield_retrieval(image_features, text_features, scale_hopfield)
    identity = torch.eye(p_xx.shape[1]).to(device) > 0.5
    loss_img = infoloob_loss(p_xx.T, p_xy.T, identity, inv_tau=inv_tau, device=device)
    loss_txt = infoloob_loss(p_yy.T, p_yx.T, identity, inv_tau=inv_tau, device=device)
    return (loss_img + loss_txt) / 2


def infoloob_loss(x, y, i, inv_tau, device):
    """Computes the InfoLOOB loss (negative mean log odds assigned to positive pairs)."""
    k = x @ y.T * inv_tau
    positives = -torch.mean(torch.sum(k * i, dim=1))
    # For logsumexp the zero entries must be equal to a very large negative number
    large_neg = -10000.
    arg_lse = k * torch.logical_not(i).to(device) + i * large_neg
    negatives = torch.mean(torch.logsumexp(arg_lse, dim=1))
    return positives + negatives


def hopfield_retrieval(image_features, text_features, scale_hopfield):
    patterns_xx = hopfield(state_patterns=image_features, stored_patterns=image_features, scale_hopfield=scale_hopfield)
    patterns_yy = hopfield(state_patterns=text_features, stored_patterns=text_features, scale_hopfield=scale_hopfield)
    patterns_xy = hopfield(state_patterns=text_features, stored_patterns=image_features, scale_hopfield=scale_hopfield)
    patterns_yx = hopfield(state_patterns=image_features, stored_patterns=text_features, scale_hopfield=scale_hopfield)
    return patterns_xx, patterns_yy, patterns_xy, patterns_yx


def hopfield(state_patterns, stored_patterns, scale_hopfield):
    retrieved_patterns = stored_patterns.T @ torch.softmax(scale_hopfield * stored_patterns @ state_patterns.T, dim=0)
    # Column vectors -> dim=0 to normalize the column vectors
    retrieved_patterns = retrieved_patterns / torch.linalg.norm(retrieved_patterns, axis=0, keepdims=True)
    return retrieved_patterns


class CLOOBLoss(nn.Module):

    def __init__(self, inv_tau, scale_hopfield, device):
        super().__init__()
        self.inv_tau = inv_tau
        self.scale_hopfield = scale_hopfield
        self.device = device

    def forward(self, image_features, text_features):
        return cloob_loss(
            image_features, text_features, self.inv_tau, self.scale_hopfield, self.device
        )
