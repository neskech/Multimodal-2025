# https://arxiv.org/pdf/2110.11316

from torch import nn
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '..',
                             'cloob-training'))
from cloob_training import loss

class CLOOBLoss(nn.Module):

    def __init__(self, inv_tau, scale_hopfield):
        super().__init__()
        self.inv_tau = inv_tau
        self.scale_hopfield = scale_hopfield

    def forward(self, image_features, text_features):
        return loss.cloob_loss(image_features, text_features, self.inv_tau,
                          self.scale_hopfield)
