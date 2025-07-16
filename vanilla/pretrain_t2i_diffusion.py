#!/usr/bin/env python
# train_text_to_image_lora.py

import argparse
import os
import time
import torch
import wandb
import logging
import PIL
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from diffusers.optimization import get_scheduler

from data_loader import get_dataloader, get_dataloader_stream

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion with LoRA on a CSV dataset"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--csv_file",                 type=str, required=True)
    parser.add_argument("--resolution",               type=int, default=512)
    parser.add_argument("--batch_size",         type=int, default=1)
    parser.add_argument("--output_dir",               type=str, required=True)
    parser.add_argument("--checkpointing_steps",      type=int, default=5000)
    parser.add_argument("--logging_steps",            type=int, default=100)
    parser.add_argument("--log_with",                 type=str, default="tensorboard",
                        choices=["wandb", "tensorboard", "none"],
                        help="Logging backend to use")
    parser.add_argument("--wandb_project",            type=str, default="text-to-image-lora")
    parser.add_argument("--wandb_run_name",           type=str, default=None)
    return parser.parse_args()

def test_model(dataloader, model, device, text_encoder, noise_scheduler, accelerator, output_dir):
    vae, unet = model
    vae.eval()
    unet.eval()

    loss_list = []
    for i, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        ts = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
        )
        noisy = noise_scheduler.add_noise(latents, noise, ts)
        hidden = text_encoder(input_ids)[0]
        pred = unet(noisy, ts, hidden).sample
        loss = torch.nn.functional.mse_loss(pred, noise)
        loss_list.append(loss.item())

        if accelerator.is_main_process and i % 20 == 0:
            accelerator.print(f"Step {i}: Loss = {loss.item():.4f}")
            PIL.Image.fromarray(
                (pred[0].cpu().numpy() * 255).astype("uint8").transpose(1, 2, 0)
            ).save(os.path.join(output_dir, f"test_output_{i}.png"))

    return loss_list
            

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Initialize accelerator
    accelerator = Accelerator(mixed_precision="fp16", log_with=args.log_with if args.log_with != "none" else None)
    device = accelerator.device
    
    if args.log_with == "wandb":
        accelerator.init_trackers(
            args.wandb_project, 
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}}
        )
    elif args.log_with == "tensorboard":
        accelerator.init_trackers(
            "test_vanilla_pretrain_diffusion",
            config=vars(args),
            init_kwargs={"tensorboard": {"flush_secs": 30, "log_dir": os.path.join(args.output_dir, "tensorboard")}}
        )
    
    accelerator.print(f"[+] Running on {device}")
    accelerator.print(f"[+] Configuration: {vars(args)}")

    # 2) Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # 3) Load models **on CPU** (Accelerate will move them for us)
    accelerator.print("[+] Loading models (on CPU)…")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    # reduce peak memory: attention slicing + gradient checkpointing
    # unet.set_attention_slice("auto")
    # unet.enable_gradient_checkpointing()

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    

    # 4) Data profiling
    accelerator.print("[+] Profiling data pipeline (~200 examples)…")
    start_map = time.time()
    dataloader = get_dataloader(
        csv_file=args.csv_file,
        tokenizer=tokenizer,
        resolution=args.resolution,
        batch_size=args.batch_size,
    )
    seen = 0
    for batch in dataloader:
        seen += batch["pixel_values"].shape[0]
        if seen >= 200:
            break
    map_time = time.time() - start_map
    accelerator.print(f"[+] ~{seen} imgs in {map_time:.1f}s → {map_time/seen:.3f}s/img")
    dataloader = iter(dataloader)  # reset iterator for training


    # 5) **Wrap everything** so Accelerate moves it all to `device`
    unet, text_encoder, vae, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, vae, optimizer, dataloader, lr_scheduler
    )

    # 6) test pretrained model
    accelerator.print("[+] Testing pretrained model…")
    losses = test_model(dataloader, (vae, unet), device, text_encoder, noise_scheduler)
    accelerator.print(f"[+] Pretrained model loss: {sum(losses)/len(losses):.4f}")

    # 7) Final save & push
    accelerator.wait_for_everyone()
    accelerator.print(f"[+] Done — outputs in {args.output_dir}")
    
    if args.log_with == "wandb" and accelerator.is_main_process:
        wandb.finish()
    elif args.log_with != "none":
        accelerator.end_training()


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    main()