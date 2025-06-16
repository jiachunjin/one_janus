import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch_fidelity
import torch
import torchvision.transforms as pth_transforms
from tqdm.auto import tqdm
from accelerate import Accelerator
from omegaconf import OmegaConf

from autoencoder import AutoEncoder

from util.dataloader import get_dataloader

inversed_transform = pth_transforms.Compose([
    pth_transforms.ToPILImage(),
])

exp_dir = "/data1/jjc/experiment/decoder/0612_decoder"
config = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))
config.data.val_path = "/data1/LargeData/ImageNet/val"
config.data.batch_size = 1
_, dataloader_val = get_dataloader(config.data)
autoencoder = AutoEncoder(config)
autoencoder.decoder.load_state_dict(torch.load(os.path.join(exp_dir, "Decoder-decoder-400k"), map_location="cpu", weights_only=True), strict=True)
autoencoder.eval()

accelerator = Accelerator()
autoencoder, dataloader_val = accelerator.prepare(autoencoder, dataloader_val)
rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes

with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader_val)):
        x, _ = batch
        x_0 = x * 2 - 1

        rec = autoencoder(x_0)

        x_0 = ((x_0 + 1) / 2).clamp(0, 1)
        rec = ((rec + 1) / 2).clamp(0, 1)

        rec = inversed_transform(rec.cpu().squeeze(0))
        ori = inversed_transform(x_0.cpu().squeeze(0))

        rec.save(f"evaluation/rec_img/{rank}_{i}.png")
        ori.save(f"evaluation/ori_img/{rank}_{i}.png")

accelerator.wait_for_everyone()

metrics_dict = torch_fidelity.calculate_metrics(
    input1  = "evaluation/ori_img",
    input2  = "evaluation/rec_img",
    cuda    = True,
    isc     = True,
    fid     = True,
    kid     = True,
    prc     = True,
    verbose = True,
)
print(metrics_dict)