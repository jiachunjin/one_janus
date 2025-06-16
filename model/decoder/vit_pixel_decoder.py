import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .vit_pixel_decoder_module import Block, precompute_freqs_cis_2d


class VitPixelDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.patch_size = config.patch_size
        self.grid_size = config.grid_size
        self.input_dim = config.input_dim

        self.input_proj = nn.Linear(config.input_dim, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList([Block(config.hidden_size, config.num_heads) for _ in range(config.depth)])
        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
            Rearrange("b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=self.patch_size, p2=self.patch_size)
        )
        # self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)
        self.conv_out = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def forward(self, x):
        B, L, D = x.shape
        pos = self.fetch_pos(self.grid_size, self.grid_size, x.device)
        x = self.input_proj(x)
        x = self.norm1(x)
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm2(x)
        x = x.permute(0, 2, 1).reshape(B, self.hidden_size, self.grid_size, self.grid_size).contiguous()
        x = self.output_proj(x)
        x = self.conv_out(x)

        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from model.loss.rec_loss import RecLoss

    config = OmegaConf.load("/data1/jjc/codebase/one_janus/config/vit_pixel_decoder.yaml")
    decoder = VitPixelDecoder(config.decoder)
    num_parameters = sum(p.numel() for p in decoder.parameters())
    print(f"Number of parameters: {num_parameters}")
    x = torch.randn(1, 576, 1024)
    ori = torch.randn(1, 3, 384, 384)
    rec = decoder(x)
    
    print(x.shape, rec.shape)

    rec_loss = RecLoss(config.rec_loss)
    loss, loss_dict = rec_loss(ori, rec, 5000, "generator")
    print("generator loss", loss, loss_dict)

    loss, loss_dict = rec_loss(ori, rec, 5000, "discriminator")
    print("discriminator loss", loss, loss_dict)