import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import torchvision
import argparse
import torch.distributed as dist
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from transformers import SiglipVisionModel, CLIPVisionModel, LlavaForConditionalGeneration
from janus.models import MultiModalityCausalLM

from model.adv_loss.hybrid_loss import Hybrid_Loss
from model.decoder import get_decoder
from util.dataloader import get_dataloader
from util.misc import flatten_dict

def get_accelerator(config):
    output_dir = os.path.join(config.root, config.exp_name, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, config.logging_dir)
    project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with=None if config.report_to == "no" else config.report_to,
        mixed_precision=config.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    return accelerator, output_dir

def main(args):
    config = OmegaConf.load(args.config)
    accelerator, output_dir = get_accelerator(config.train)
    dataloader = get_dataloader(config.data)
    decoder = get_decoder(config.decoder)
    if config.extractor == "siglip":
        extractor = SiglipVisionModel.from_pretrained("/data1/ckpts/google/siglip-base-patch16-224")
    elif config.extractor == "clip":
        extractor = CLIPVisionModel.from_pretrained("/data1/ckpts/openai/clip-vit-base-patch16")
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]
        normalize = torchvision.transforms.Normalize(
            mean=image_mean,
            std=image_std
        )
    elif config.extractor == "llava":
        llava = LlavaForConditionalGeneration.from_pretrained("/data1/ckpts/llava-hf/llava-1.5-7b-hf", device_map="cpu")
        extractor = llava.vision_tower
    elif config.extractor == "janus":
        janus = MultiModalityCausalLM.from_pretrained("/data1/ckpts/deepseek-ai_/Janus-Pro-1B", trust_remote_code=True)
        extractor = janus.vision_model

    hybrid_loss = Hybrid_Loss(config.hybrid_loss.disc_start, disc_weight=config.hybrid_loss.disc_weight, perceptual_weight=1.1)

    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu", weights_only=True)
        if config.train.skipped_keys:
            ckpt = {k: v for k, v in ckpt.items() if k not in config.train.skipped_keys}
        m, u = decoder.load_state_dict(ckpt, strict=False)
        if accelerator.is_main_process:
            print(f"miss: {m}, unexpected: {u}")

    if config.train.loss_resume_path is not None:
        ckpt = torch.load(config.train.loss_resume_path, map_location="cpu", weights_only=True)
        m, u = hybrid_loss.load_state_dict(ckpt, strict=False)
        if accelerator.is_main_process:
            print(f"Loss, miss: {m}, unexpected: {u}")

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(decoder.parameters())
    disc_params = list(hybrid_loss.discriminator.parameters())

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    optimizer_disc = torch.optim.AdamW(
        disc_params,
        lr           = 1e-5 / config.hybrid_loss.disc_weight,
        betas        = (0.9, 0.95),
        weight_decay = 5e-2,
        eps          = 1e-8,
    )

    if accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    decoder, hybrid_loss, dataloader, optimizer = accelerator.prepare(decoder, hybrid_loss, dataloader, optimizer)
    hybrid_loss = hybrid_loss.to(dtype)
    extractor = extractor.to(accelerator.device, dtype).eval()    

    config.device_count = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)

    training_done = False
    epoch = 0
    progress_bar = tqdm(
        total   = config.train.num_iter,
        initial = global_step,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )

    if accelerator.is_main_process:
        print(f"Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e6} M")

    if accelerator.is_main_process:
        print(f"decoder dtype: {next(decoder.parameters()).dtype}")
        print(f"Hybrid loss dtype: {next(hybrid_loss.parameters()).dtype}")
        print(f"extractor dtype: {next(extractor.parameters()).dtype}")
        print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    while not training_done:
        for batch in dataloader:
            with accelerator.accumulate([decoder, hybrid_loss]):
                decoder.train()
                hybrid_loss.train()
                if config.data.name == "blip3o" or config.data.name == "journeydb":
                    try:
                        x_0 = batch["pixel_values"].to(dtype)
                    except:
                        continue
                else:
                    x_0, _ = batch
                    x_0 = x_0.to(dtype)
                    x_0 = x_0 * 2 - 1
                    B, C, H, W = x_0.shape

                with torch.no_grad():
                    if config.extractor == "siglip":
                        feature = extractor(x_0).last_hidden_state
                    elif config.extractor == "clip":
                        x = (x_0 + 1) / 2
                        x = normalize(x)
                        feature = extractor(pixel_values=x).last_hidden_state[:, 1:]
                    elif config.extractor == "llava":
                        image_outputs = extractor(x_0, output_hidden_states=True)
                        feature = image_outputs.hidden_states[-2][:, 1:]
                    elif config.extractor == "janus":
                        feature = extractor(x_0) # (B, 576, 1024)

                rec = decoder(feature).to(dtype)

                loss = hybrid_loss(
                    inputs          = x_0,
                    reconstructions = rec,
                    optimizer_idx   = 0,
                    global_step     = global_step+1,
                    last_layer      = decoder.module.cnn_decoder.last_layer
                )
                # loss = F.mse_loss(rec, x_0, reduction="mean") + 1.1 * p_loss(rec, x_0).mean()

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                optimizer.step()

                # ---------- optimize discriminator ----------
                loss_disc = hybrid_loss(
                    inputs          = x_0,
                    reconstructions = rec,
                    optimizer_idx   = 1,
                    global_step     = global_step+1,
                    last_layer      = decoder.module.cnn_decoder.last_layer,
                )
                optimizer_disc.zero_grad()
                accelerator.backward(loss_disc)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(disc_params, 1.0)
                optimizer_disc.step()

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        siglip_decoder = accelerator.gather(loss.detach()).mean().item(),
                        loss_disc      = accelerator.gather(loss_disc.detach()).mean().item()
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

            if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                decoder.eval()
                state_dict = accelerator.unwrap_model(decoder).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"Decoder-{config.train.exp_name}-{global_step // 1000}k"))

                state_dict = accelerator.unwrap_model(hybrid_loss).state_dict()
                torch.save(state_dict, os.path.join(output_dir, f"Loss-{config.train.exp_name}-{global_step // 1000}k"))

            accelerator.wait_for_everyone()

            if global_step >= config.train.num_iter:
                training_done = True
                break
        epoch += 1
        accelerator.log({"epoch": epoch}, step=global_step)
    accelerator.end_training()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/bi_tok.yaml")
    args = parser.parse_args()
    main(args)