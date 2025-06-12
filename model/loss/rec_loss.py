import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminator import NLayerDiscriminator
from .perceptual_loss import PerceptualLoss


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


class RecLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rec_weight = config.rec_weight
        self.perceptual_weight = config.perceptual_weight
        self.discriminator_weight = config.discriminator_weight
        self.lecam_regularization_weight = config.lecam_regularization_weight

        self.discriminator_factor = config.discriminator_factor # ?
        self.lecam_ema_decay = config.get("lecam_ema_decay", 0.999)
        self.discriminator_start_iter = config.discriminator_start_iter
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.discriminator = NLayerDiscriminator()
        self.perceptual_loss = PerceptualLoss(config.perceptual_loss_name).eval()

    def forward(
            self,
            inputs: torch.Tensor,
            reconstructions: torch.Tensor,
            global_step: int,
            mode: str = "generator",
    ):
        # Both inputs and reconstructions are in range [0, 1].
        # inputs = inputs.float()
        # reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def should_discriminator_be_trained(self, global_step):
        return global_step >= self.discriminator_start_iter

    def _forward_generator(self, inputs, reconstructions, global_step):
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")

        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        total_loss = (
            self.rec_weight * reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.discriminator_weight * discriminator_factor * generator_loss
        )

        loss_dict = dict(
            total_loss           = total_loss.clone().detach().item(),
            reconstruction_loss  = reconstruction_loss.detach().item(),
            perceptual_loss      = perceptual_loss.detach().item(),
            gan_loss             = generator_loss.detach().item(),
        )

        return total_loss, loss_dict

    def _forward_discriminator(self, inputs, reconstructions, global_step):
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss = discriminator_loss.detach().item(),
            logits_real        = logits_real.detach().mean().item(),
            logits_fake        = logits_fake.detach().mean().item(),
            lecam_loss         = lecam_loss.detach().item(),
        )
        return discriminator_loss, loss_dict


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config/vit_pixel_decoder.yaml")
    rec_loss = RecLoss(config.rec_loss)
    print(rec_loss)