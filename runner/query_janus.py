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

from util.dataloader import get_dataloader
from util.misc import flatten_dict
from janus.models import MultiModalityCausalLM, VLChatProcessor
from util.query_janus_util import equip_janus

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

    janus = MultiModalityCausalLM.from_pretrained("/data1/ckpts/deepseek-ai_/Janus-Pro-1B", trust_remote_code=True)
    janus, train_scheduler = equip_janus(janus, config)

    if config.train.resume_path is not None:
        ckpt = torch.load(config.train.resume_path, map_location="cpu")
        raise NotImplementedError("Not implemented")
    

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in janus.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        params_to_learn,
        lr           = config.train.lr,
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

    janus, dataloader, optimizer = accelerator.prepare(janus, dataloader, optimizer)

    config.device_count = accelerator.num_processes
    if accelerator.is_main_process:
        accelerator.init_trackers(config.train.wandb_proj, config=flatten_dict(config))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)

    if accelerator.is_main_process:
        print(f"Learnable parameters: {sum(p.numel() for p in params_to_learn if p.requires_grad) / 1e6} M")

    if accelerator.is_main_process:
        print(f"janus dtype: {next(janus.parameters()).dtype}")
        print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    training_done = False
    epoch = 0
    progress_bar = tqdm(
        total   = config.train.num_iter,
        initial = global_step,
        desc    = "Steps",
        disable = not accelerator.is_local_main_process,
    )

    while not training_done:
        with accelerator.accumulate([janus]):
            for batch in dataloader:
                janus.train()
                texts = batch["texts"]
                attention_mask = batch["attention_mask"]
                pixel_values = batch["pixel_values"].to(dtype)
                img_features = janus.vision_model(pixel_values).to(dtype)
                # img_embedding = janus.aligner(img_features)

                B, L = texts.shape

                mask = torch.rand(B, 1) < config.train.cfg_drop_rate
                mask = mask.repeat(1, L)
                texts[mask] = 100002 # padding 
                boi_token = torch.ones((B, 1), dtype=torch.int64, device=accelerator.device) * 100003 # <｜begin▁of▁image｜>
                input_ids = torch.cat([texts, boi_token], dim=1)
                text_embedding = janus.language_model.get_input_embeddings()(input_ids)

                joint_embedding = torch.cat((text_embedding, janus.query.unsqueeze(0).repeat(B, 1, 1)), dim=1)

                img_mask = torch.ones((B, 1 + 576), dtype=torch.bool, device=accelerator.device)
                attention_mask = torch.cat([attention_mask, img_mask], dim=1)

                hidden_states = janus.language_model(
                    inputs_embeds        = joint_embedding,
                    attention_mask       = attention_mask,
                    output_hidden_states = True,
                ).hidden_states[-1]
                z = hidden_states[:, -576:, :] # use the hidden states of the query tokens, not the boi token

                timesteps = torch.randint(0, 1000, (B,), dtype=torch.int64, device=accelerator.device)
                noise = torch.randn_like(img_features, device=accelerator.device, dtype=z.dtype)
                noisy_latents = train_scheduler.add_noise(img_features, noise, timesteps)
                target = train_scheduler.get_velocity(img_features, noise, timesteps)
                pred = janus.dit(noisy_latents, z, timesteps)
                loss = F.mse_loss(pred, target)
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                optimizer.step()

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        query_diff_loss = accelerator.gather(loss.detach()).mean().item(),
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                    janus.eval()
                    state_dict = accelerator.unwrap_model(janus).dit.state_dict()
                    save_path = os.path.join(output_dir, f"dit-{config.train.exp_name}-{global_step}")
                    torch.save(state_dict, save_path)
                    print(f"DiT model saved to {save_path}")

                    state_dict = accelerator.unwrap_model(janus).query.detach().cpu()
                    save_path = os.path.join(output_dir, f"query-{config.train.exp_name}-{global_step}")
                    torch.save({"query": state_dict}, save_path)
                    print(f"Query saved to {save_path}")

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
    parser.add_argument("--config", type=str, default="config/query_janus.yaml")
    args = parser.parse_args()
    main(args)