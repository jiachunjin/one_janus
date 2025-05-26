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
from janus.models.diff_mlp import SimpleMLPAdaLN
from diffusers import DDPMScheduler
from einops import rearrange
from model.decoder import get_decoder

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

    diff_head = SimpleMLPAdaLN(
            in_channels = config.train.pred_dim,
            model_channels = 1024,
            out_channels = config.train.pred_dim,
            z_channels = 2048,
            num_res_blocks = 2,
        )
    if config.train.head_resume_path is not None:
        ckpt = torch.load(config.train.head_resume_path, map_location="cpu")
        diff_head.load_state_dict(ckpt, strict=True)
    if config.train.janus_resume_path is not None:
        ckpt = torch.load(config.train.janus_resume_path, map_location="cpu")
        janus.language_model.load_state_dict(ckpt, strict=True)

    decoder = get_decoder(config.decoder)
    ckpt = torch.load(config.decoder.ckpt, map_location="cpu")
    decoder.load_state_dict(ckpt, strict=True)
    to_bottleneck = decoder.to_bottleneck
    for param in to_bottleneck.parameters():
        param.requires_grad = False

    # for param in janus.parameters():
    #     param.requires_grad = False
    # vl_chat_processor = VLChatProcessor.from_pretrained("/data1/ckpts/deepseek-ai_/Janus-Pro-1B")

    global_step = config.train.global_step if config.train.global_step is not None else 0
    params_to_learn = list(p for p in janus.parameters() if p.requires_grad) + list(diff_head.parameters())

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

    janus, diff_head, dataloader, optimizer = accelerator.prepare(janus, diff_head, dataloader, optimizer)
    janus = janus.to(dtype)
    to_bottleneck = to_bottleneck.to(accelerator.device, dtype)
    to_bottleneck.eval()

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
        print(f"janus dtype: {next(janus.parameters()).dtype}")
        print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    train_scheduler = DDPMScheduler(
        beta_schedule          = "scaled_linear",
        beta_start             = 0.00085,
        beta_end               = 0.012,
        num_train_timesteps    = 1000,
        clip_sample            = False,
        prediction_type        = "v_prediction",
        # set_alpha_to_one       = True,
        steps_offset           = 1,
        trained_betas          = None,
        timestep_spacing       = "trailing",
        rescale_betas_zero_snr = True
    )

    while not training_done:
        with accelerator.accumulate([janus, diff_head]):
            for batch in dataloader:
                janus.train()
                diff_head.train()
                text = batch["texts"]
                pixel_values = batch["pixel_values"].to(dtype)
                img_features = janus.module.vision_model(pixel_values).to(dtype)
                # gt_feature = to_bottleneck(img_features)
                img_embeddings = janus.module.aligner(img_features)
                # img_embeddings = janus.module.low_dim_aligner(gt_feature)
                
                joint_embeddings = []
                B = pixel_values.shape[0]

                for i, input_ids in enumerate(text):
                    input_ids = torch.cat([input_ids, torch.tensor([100003], device=accelerator.device)])
                    text_embedding = janus.module.language_model.get_input_embeddings()(input_ids).unsqueeze(0)
                    img_embedding = img_embeddings[i].unsqueeze(0)
                    joint_embedding = torch.cat((text_embedding, img_embedding), dim=1)
                    joint_embeddings.append(joint_embedding)

                joint_embeddings = torch.cat(joint_embeddings, dim=0)

                hidden_states = janus.module.language_model(
                    inputs_embeds=joint_embeddings,
                    attention_mask=None,
                    output_hidden_states=True,
                ).hidden_states[-1]
                z = hidden_states[:, -576-1:-1, :]
                gt_feature = img_features.to(dtype)
                # gt_feature = to_bottleneck(gt_feature)
                # print(gt_feature.shape, gt_feature.min(), gt_feature.max(), gt_feature.mean())
                # print(z.shape, z.min(), z.max(), z.mean())
                # exit(0)

                z = rearrange(z, "B L D -> (B L) D")
                gt_feature = rearrange(gt_feature, "B L D -> (B L) D")
                B = z.shape[0]
                timesteps = torch.randint(0, 1000, (B,), dtype=torch.int64, device=z.device)
                noise = torch.randn_like(gt_feature, device=z.device, dtype=z.dtype)
                noisy_latents = train_scheduler.add_noise(gt_feature, noise, timesteps)
                target = train_scheduler.get_velocity(gt_feature, noise, timesteps)
                pred = diff_head(noisy_latents, timesteps, z)

                loss = F.mse_loss(pred.to(dtype), target)

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                optimizer.step()

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    logs = dict(
                        janus_rec_mse = accelerator.gather(loss.detach()).mean().item(),
                    )
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                if global_step > 0 and global_step % config.train.save_every == 0 and accelerator.is_main_process:
                    janus.eval()
                    diff_head.eval()
                    state_dict = accelerator.unwrap_model(janus).language_model.state_dict()
                    save_path = os.path.join(output_dir, f"janus-{config.train.exp_name}-{global_step // 1000}k")
                    torch.save(state_dict, save_path)

                    state_dict = accelerator.unwrap_model(diff_head).state_dict()
                    save_path = os.path.join(output_dir, f"diff_head-{config.train.exp_name}-{global_step // 1000}k")
                    torch.save(state_dict, save_path)
                    print(f"Model saved to {save_path}")

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