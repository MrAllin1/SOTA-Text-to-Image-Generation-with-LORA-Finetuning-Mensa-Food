#!/usr/bin/env python
# train_text_to_image_lora.py

import argparse
import os
import time
import torch
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from diffusers.optimization import get_scheduler

from data_loader import get_dataloader  # your data package

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion with LoRA on a CSV dataset"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--csv_file",                 type=str, required=True)
    parser.add_argument("--resolution",               type=int, default=512)
    parser.add_argument("--train_batch_size",         type=int, default=1)
    parser.add_argument("--max_train_steps",          type=int, default=30000)
    parser.add_argument("--learning_rate",            type=float, default=2e-4)
    parser.add_argument("--lr_scheduler",             type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps",          type=int, default=200)
    parser.add_argument("--rank",                     type=int, default=4)
    parser.add_argument("--output_dir",               type=str, required=True)
    parser.add_argument("--push_to_hub",              action="store_true")
    parser.add_argument("--hub_model_id",             type=str)
    parser.add_argument("--checkpointing_steps",      type=int, default=5000)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Accelerator picks the right device for us
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    accelerator.print(f"[+] Running on {device}")

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
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # 4) LoRA on the U-Net
    accelerator.print("[+] Preparing LoRA adapters…")
    unet = prepare_model_for_kbit_training(unet)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    trainable_params = filter(lambda p: p.requires_grad, unet.parameters())

    # 5) Data profiling
    accelerator.print("[+] Profiling data pipeline (~200 examples)…")
    start_map = time.time()
    dl_profile = get_dataloader(
        csv_file=args.csv_file,
        tokenizer=tokenizer,
        resolution=args.resolution,
        batch_size=args.train_batch_size,
    )
    seen = 0
    for batch in dl_profile:
        seen += batch["pixel_values"].shape[0]
        if seen >= 200:
            break
    map_time = time.time() - start_map
    accelerator.print(f"[+] ~{seen} imgs in {map_time:.1f}s → {map_time/seen:.3f}s/img")

    # real dataloader
    dataloader = get_dataloader(
        csv_file=args.csv_file,
        tokenizer=tokenizer,
        resolution=args.resolution,
        batch_size=args.train_batch_size,
    )

    # 6) Optimizer & LR scheduler
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 7) **Wrap everything** so Accelerate moves it all to `device`
    unet, text_encoder, vae, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, vae, optimizer, dataloader, lr_scheduler
    )

    # 8) 100-step benchmark
    accelerator.print("[+] 100-step benchmark…")
    start_bench = time.time()
    for i, batch in enumerate(dataloader):
        if i >= 100:
            break
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        # forward
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

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    bench_time = time.time() - start_bench
    sps = bench_time / 100.0
    total_h = sps * args.max_train_steps / 3600.0
    accelerator.print(f"[+] 100 steps: {bench_time:.1f}s → {sps:.3f}s/step")
    accelerator.print(f"[+] Est. full training: {total_h:.1f}h")

    # 9) Full training + checkpoints
    accelerator.print("[+] Training start…")
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

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

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 100 == 0:
                accelerator.print(f"Step {global_step}/{args.max_train_steps} — loss {loss.item():.4f}")
            if global_step % args.checkpointing_steps == 0:
                accelerator.wait_for_everyone()
                ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                unet.save_pretrained(ckpt, safe_serialization=True)
                accelerator.print(f"[+] Saved {ckpt}")

    # 10) Final save & push
    accelerator.wait_for_everyone()
    unet.save_pretrained(args.output_dir, safe_serialization=True)
    text_encoder.save_pretrained(args.output_dir)
    accelerator.print(f"[+] Done — outputs in {args.output_dir}")

    if args.push_to_hub and args.hub_model_id:
        from huggingface_hub import Repository
        repo = Repository(args.output_dir, clone_from=args.hub_model_id)
        repo.push_to_hub()
        accelerator.print(f"[+] Pushed to {args.hub_model_id}")

if __name__ == "__main__":
    main()
