import torch
import torch.nn as nn
from omegaconf import OmegaConf

from janus.models import MultiModalityCausalLM
from model.decoder.vit_pixel_decoder import VitPixelDecoder


class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = MultiModalityCausalLM.from_pretrained(config.janus_path, trust_remote_code=True).vision_model
        self.decoder = VitPixelDecoder(config.decoder)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))


if __name__ == "__main__":
    import os
    from util.dataloader import get_dataloader

    exp_dir = "/data1/jjc/experiment/decoder/0612_decoder"
    config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
    config.data.val_path = "/data1/LargeData/ImageNet/val"
    _, dataloader_val = get_dataloader(config.data)
    autoencoder = AutoEncoder(config)
    autoencoder.decoder.load_state_dict(torch.load(os.path.join(exp_dir, "Decoder-decoder-400k"), map_location="cpu", weights_only=True), strict=True)
    autoencoder.eval()
    # autoencoder.to(device)
    # autoencoder.decode.to(device)
    # autoencoder.encode.to(device)

    # for batch in dataloader_val:
    #     batch = batch.to(device)
    #     with torch.no_grad():
    #         output = autoencoder(batch)
    #     print(output.shape)