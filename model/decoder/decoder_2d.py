import torch
import torch.nn as nn
from einops import rearrange
from ..mar.vae import Decoder
from .decoder_ff import Decoder_ff


class Decoder_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebook_size = 8192
        self.codebook_dim = 1024
        self.codebook = nn.Parameter(self.codebook_dim ** 0.5 * torch.randn(self.codebook_size, self.codebook_dim))
        self.code_proj = nn.Linear(self.codebook_dim, 16)


        self.cnn_decoder = Decoder(ch_mult=(1, 1, 2, 2, 4), z_channels=16)

        ckpt = torch.load("model/mar/kl16.ckpt", map_location="cpu")
        decoder_ckpt = {}
        for k in ckpt['model'].keys():
            if "decoder." in k:
                new_k = k.replace("decoder.", "")
                decoder_ckpt[new_k] = ckpt['model'][k]
        self.cnn_decoder.load_state_dict(decoder_ckpt, strict=True)
        self.cnn_decoder.requires_grad_(False)
    
    def forward(self, index):
        code_weight_probs = torch.nn.functional.one_hot(index, num_classes=self.codebook_size).to(self.codebook.dtype)  # (B M codebook_size)

        quantized = torch.einsum("bmn,nd->bmd", code_weight_probs, self.codebook)
        quantized = self.code_proj(quantized)
        quantized = rearrange(quantized, "b (h w) d -> b d h w", h=14)

        rec = self.cnn_decoder(quantized)

        return rec


class Decoder_1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebook_size = 8192
        self.codebook_dim = 1024
        self.codebook = nn.Parameter(self.codebook_dim ** 0.5 * torch.randn(self.codebook_size, self.codebook_dim))
        self.decoder_ff = Decoder_ff(config)
    
    def forward(self, index):
        code_weight_probs = torch.nn.functional.one_hot(index, num_classes=self.codebook_size).to(self.codebook.dtype)  # (B M codebook_size)

        quantized = torch.einsum("bmn,nd->bmd", code_weight_probs, self.codebook)
        rec = self.decoder_ff(quantized)

        return rec
        