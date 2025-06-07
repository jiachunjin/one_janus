import torch
import torch.nn as nn
from .module import DiTBlock, TimestepEmbedder, FinalLayer, precompute_freqs_cis_2d


class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.depth = config.depth
        self.x_dim = config.x_dim
        self.z_dim = config.z_dim

        self.x_proj = nn.Linear(self.x_dim, self.hidden_size)
        self.z_proj = nn.Linear(self.z_dim, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, self.num_heads) for _ in range(self.depth)
        ])
        self.final_layer = FinalLayer(self.hidden_size, self.x_dim)
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def forward(self, x_t, z, t):
        B, L, d = x_t.shape
        pos = self.fetch_pos(24, 24, x_t.device)
        x = self.x_proj(x_t)
        z = self.z_proj(z)

        t = self.t_embedder(t.view(-1), x_t.dtype).view(B, -1, self.hidden_size)
        c = t + z
        for i, block in enumerate(self.blocks):
            x = block(x, c, pos, None)
        
        x = self.final_layer(x, c)
        
        return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("/home/jjc/codebase/one_janus/config/query_janus.yaml")
    dit = DiT(config.dit)
    x_t = torch.randn(1, 576, 1024)
    z = torch.randn(1, 576, 2048)
    t = torch.randint(0, 1000, (1,))
    x = dit(x_t, z, t)
    print(x.shape)