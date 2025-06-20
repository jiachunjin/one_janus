import torch
import torch.nn as nn
from einops import rearrange

from ..module import Transformer_block
from ..mar.vae import Decoder

class Sem_Decoder_without_Reg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        self.feature_num = config.feature_num
        self.dim = config.dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.seq_len = config.seq_len
        if hasattr(config, "feature_bottleneck"):
            self.use_bottleneck = True
            self.feature_bottleneck = config.feature_bottleneck
            self.to_bottleneck = nn.Linear(self.feature_dim, self.feature_bottleneck)
            self.revert_bottleneck = nn.Linear(self.feature_bottleneck, self.feature_dim)
            self.feature_proj = nn.Linear(self.feature_bottleneck, self.dim)
        else:
            self.use_bottleneck = False
            self.feature_proj = nn.Linear(self.feature_dim, self.dim)
        self.feature_pos_embed = nn.Parameter(torch.randn(1, self.feature_num, self.dim))

        self.norm1 = nn.LayerNorm(config.dim)
        self.blocks = nn.ModuleList([
            Transformer_block(self.dim, self.num_heads, use_adaLN=False) for _ in range(self.depth)
        ])
        self.norm2 = nn.LayerNorm(config.dim)

        self.proj_before_cnn = nn.Linear(self.dim, 16)
        self.cnn_decoder = Decoder(ch_mult=(1, 1, 2, 2, 4), z_channels=16)
        if self.config.cnn_decoder_ckpt is not None:
            ckpt = torch.load(self.config.cnn_decoder_ckpt, map_location="cpu")
            decoder_ckpt = {}
            for k in ckpt['model'].keys():
                if "decoder." in k:
                    new_k = k.replace("decoder.", "")
                    decoder_ckpt[new_k] = ckpt['model'][k]
            self.cnn_decoder.load_state_dict(decoder_ckpt, strict=True)
            # self.cnn_decoder.requires_grad_(False)

    def forward(self, feature):
        """
        feature: (B, L, D)
        return: (B, C, H, W), reconstructed image
        """
        B, L, D = feature.shape
        if self.use_bottleneck:
            x = self.to_bottleneck(feature)
            if not self.training:
                print(f"after bottleneck x.shape: {x.shape}")
            x = self.feature_proj(x) + self.feature_pos_embed
        else:
            x = self.feature_proj(feature) + self.feature_pos_embed

        x = self.norm1(x)
        for block in self.blocks:
            x = block(x)

        x = self.norm2(x)

        x = self.proj_before_cnn(x) # (B HW 16)
        x = torch.nn.functional.normalize(x, dim=-1)
        x = rearrange(x, "b (h w) c -> b c h w", h=int(self.seq_len ** 0.5))
        if hasattr(self.config, "use_llava") and self.config.use_llava:
            x = self.cnn_decoder(x, last_upscale=1.75)
        else:
            x = self.cnn_decoder(x)

        return x
    
    def pass_through_bottleneck(self, feature):
        assert self.use_bottleneck == True
        # print('feature', feature.dtype, self.to_bottleneck.d)
        x = self.to_bottleneck(feature)
        return x
    
    def revert_from_bottleneck(self, feature):
        """
        feature: (B, L, D') where D' is the bottleneck dimension
        """
        assert self.use_bottleneck == True
        x = self.revert_bottleneck(feature)
        return x
    
    def forward_with_compressed_feature(self, x):
        # if not self.training:
            # print(f"x.shape: {x.shape}")
        x = self.feature_proj(x) + self.feature_pos_embed

        x = self.norm1(x)
        for block in self.blocks:
            x = block(x)

        x = self.norm2(x)

        x = self.proj_before_cnn(x) # (B HW 16)
        x = torch.nn.functional.normalize(x, dim=-1)
        x = rearrange(x, "b (h w) c -> b c h w", h=int(self.seq_len ** 0.5))
        if hasattr(self.config, "use_llava") and self.config.use_llava:
            x = self.cnn_decoder(x, last_upscale=1.75)
        else:
            x = self.cnn_decoder(x)

        return x