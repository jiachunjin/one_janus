import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module import Transformer_block, Final_layer, TimestepEmbedder


class Decoder_diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_pixel = config.is_pixel
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.seq_len = (self.img_size // self.patch_size) ** 2
        self.hidden_dim = config.hidden_dim
        self.latent_num = config.latent_num
        self.latent_dim = config.latent_dim
        self.depth = config.depth
        self.num_heads = config.num_heads

        self.x_embedder = nn.Linear(self.patch_size * self.patch_size * 3, self.hidden_dim)
        self.t_embedder = TimestepEmbedder(self.hidden_dim)
        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, self.hidden_dim))
        self.latent_pos_embed = nn.Parameter(torch.randn(1, self.latent_num, self.hidden_dim))


        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.blocks = nn.ModuleList([
            Transformer_block(self.hidden_dim, self.num_heads, use_adaLN=True) for _ in range(self.depth)
        ])
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        if self.is_pixel:
            self.final_layer = Final_layer(self.hidden_dim, self.patch_size, 3)
        else:
            self.final_layer = nn.Linear(self.hidden_dim, 768)
    
    def forward(self, x_t, z, t, no_patchify=False):
        """
        x_t: B C H W, noisy image
        z: B M D, latent code
        output: B C H W, denoised image
        """
        if self.is_pixel:
            x = F.unfold(x_t, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        else:
            x = x_t
        x = self.x_embedder(x) + self.pos_embed
        z = self.latent_proj(z) + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)

        t_embed = self.t_embedder(t)
        x = self.norm1(x)
        for block in self.blocks:
            x = block(x, t_embed)
        
        x = x[:, :-self.latent_num, :]
        x = self.norm2(x)

        if self.is_pixel:
            x = self.final_layer(x)
            x = self.unpatchify(x)
        else:
            x = self.final_layer(x)

        return x

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = 3
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))

        return imgs