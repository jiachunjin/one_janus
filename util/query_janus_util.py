import torch
import torch.nn as nn
from model.dit import DiT
from diffusers import DDPMScheduler


def equip_janus(janus, config):
    """
    把learnable queries和DiT装到pretrained janus上
    """
    dit = DiT(config.dit)
    query = nn.Parameter(torch.randn(config.query.num_queries, config.query.query_dim))
    janus.requires_grad_(False)
    janus.eval()
    janus.query = query
    janus.query.requires_grad_(True)
    janus.dit = dit
    janus.dit.requires_grad_(True)

    train_scheduler = DDPMScheduler(
        beta_schedule          = "scaled_linear",
        beta_start             = 0.00085,
        beta_end               = 0.012,
        num_train_timesteps    = 1000,
        clip_sample            = False,
        prediction_type        = "v_prediction",
        # set_alpha_to_one       = True,
        steps_offset           = 1,
        trained_betas          = None,
        timestep_spacing       = "trailing",
        rescale_betas_zero_snr = True
    )

    return janus, train_scheduler