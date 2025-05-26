import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from ..module import Transformer_block, Final_layer
from ..mar.vae import Decoder

class Decoder_ff(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.seq_len = (self.img_size // self.patch_size) ** 2
        self.hidden_dim = config.hidden_dim
        self.latent_num = config.latent_num
        self.latent_dim = config.latent_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.use_cnn_decoder = config.use_cnn_decoder

        self.mask_token = nn.Parameter(torch.randn(1, self.seq_len, self.hidden_dim))
        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.latent_pos_embed = nn.Parameter(torch.randn(1, self.latent_num, self.hidden_dim))

        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.blocks = nn.ModuleList([
            Transformer_block(self.hidden_dim, self.num_heads, use_adaLN=False) for _ in range(self.depth)
        ])
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        if self.use_cnn_decoder:
            self.feature_norm = nn.LayerNorm(16, elementwise_affine=False)
            self.proj_before_cnn = nn.Linear(self.hidden_dim, 16)
            self.cnn_decoder = Decoder(ch_mult=(1, 1, 2, 2, 4), z_channels=16)
            if self.config.cnn_decoder_ckpt is not None:
                ckpt = torch.load(self.config.cnn_decoder_ckpt, map_location="cpu")
                decoder_ckpt = {}
                for k in ckpt['model'].keys():
                    if "decoder." in k:
                        new_k = k.replace("decoder.", "")
                        decoder_ckpt[new_k] = ckpt['model'][k]
                self.cnn_decoder.load_state_dict(decoder_ckpt, strict=True)
                self.cnn_decoder.requires_grad_(False)
        else:
            self.final_layer = Final_layer(self.hidden_dim, self.patch_size, 3)


    def forward(self, z, residual=None):
        """
        z: B M D, latent code
        output: B C H W, reconstructed image
        """
        B, _, _ = z.shape

        x = self.mask_token.expand(B, -1, -1)
        z = self.latent_proj(z) + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)

        x = self.norm1(x)
        for block in self.blocks:
            x = block(x)
        
        x = x[:, :-self.latent_num, :]
        x = self.norm2(x)

        if self.use_cnn_decoder:
            x = self.proj_before_cnn(x) # (B HW 16)
            # x = self.feature_norm(x)
            x = torch.nn.functional.normalize(x, dim=-1)
            x = rearrange(x, "b (h w) c -> b c h w", h=int(self.img_size // self.patch_size))
            if residual is not None:
                x = x + residual
                # x = residual
            x = self.cnn_decoder(x)
        else:
            x = self.final_layer(x)
            x = self.unpatchify(x)

        return x
    
    def forward_before_cnn(self, quantized):
        B, _, _ = quantized.shape

        x = self.mask_token.expand(B, -1, -1)
        quantized = self.latent_proj(quantized) + self.latent_pos_embed
        x = torch.cat([x, quantized], dim=1)

        x = self.norm1(x)
        for block in self.blocks:
            x = block(x)
        
        x = x[:, :-self.latent_num, :]
        x = self.norm2(x)

        x = self.proj_before_cnn(x) # (B HW 16)
        # x = self.feature_norm(x) # (B HW 16)
        x = torch.nn.functional.normalize(x, dim=-1)
        x = rearrange(x, "b (h w) c -> b c h w", h=int(self.img_size // self.patch_size)) # (B 16 H W)

        return x

    
    def get_feature_residual(self, z, residual):
        B, _, _ = z.shape

        x = self.mask_token.expand(B, -1, -1)
        z = self.latent_proj(z) + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)

        x = self.norm1(x)
        for block in self.blocks:
            x = block(x)
        
        x = x[:, :-self.latent_num, :]
        x = self.norm2(x)

        x = self.proj_before_cnn(x) # (B HW 16)
        x = self.feature_norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=int(self.img_size // self.patch_size))

        return x, residual

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