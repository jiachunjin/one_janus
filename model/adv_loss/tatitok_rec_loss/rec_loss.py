import torch
import torch.nn as nn
from .discriminator import NLayerDiscriminator


class TatiTokRecLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.discriminator = NLayerDiscriminator()
        self.perceptual_loss = PerceptualLoss(
            loss_config.perceptual_loss).eval()