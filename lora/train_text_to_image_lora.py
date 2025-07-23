#!/usr/bin/env python
"""Fine‑tune Stable Diffusion with LoRA on a CSV dataset.
Adds TensorBoard logging via 🤗 Accelerate’s built‑in tracker.
"""
from __future__ import annotations

import argparse
import os
import time
import torch
import wandb
import logging
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from diffusers.optimization import get_scheduler
from data_loader import get_dataloader  # custom dataloader

from data_loader import get_dataloader, get_dataloader_stream

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion with LoRA on a CSV dataset"
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--csv_file",                 type=str, required=True)
    parser.add_argument("--resolution",               type=int, default=512)
    parser.add_argument("--train_batch_size",         type=int, default=1)
    parser.add_argument("--max_train_steps",          type=int, default=15000)
    parser.add_argument("--learning_rate",            type=float, default=2e-4)
    parser.add_argument("--lr_scheduler",             type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps",          type=int, default=200)
    parser.add_argument("--rank",                     type=int, default=4)
    parser.add_argument("--output_dir",               type=str, required=True)
    parser.add_argument("--push_to_hub",              action="store_true")
    parser.add_argument("--hub_model_id",             type=str)
    parser.add_argument("--checkpointing_steps",      type=int, default=5000)
    parser.add_argument("--logging_steps",            type=int, default=100)
    parser.add_argument("--log_with",                 type=str, default="tensorboard",
                        choices=["wandb", "tensorboard", "none"],
                        help="Logging backend to use")
    parser.add_argument("--wandb_project",            type=str, default="text-to-image-lora")
    parser.add_argument("--wandb_run_name",           type=str, default=None)
    return parser.parse_args()

# ─────────────────────────── MAIN ───────────────────────────
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
            "train",
            config=vars(args),
            init_kwargs={"tensorboard": {"flush_secs": 30, "log_dir": os.path.join(args.output_dir, "tensorboard")}}
        )
    
    accelerator.print(f"[+] Running on {device}")
    accelerator.print(f"[+] Configuration: {vars(args)}")

    # 2) Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # 3) Load models (CPU → Accelerate moves to GPU)
    accelerator.print("[+] Loading SD backbone (CPU)…")
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
    unet.set_attention_slice("auto")          # returns None
    unet.enable_gradient_checkpointing()      # call separately

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # 4) LoRA
    accelerator.print("[+] Injecting LoRA adapters…")
    unet = prepare_model_for_kbit_training(unet)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    trainable_params = filter(lambda p: p.requires_grad, unet.parameters())
    
    # Log model information
    accelerator.print(f"[+] LoRA Configuration: {lora_config}")
    accelerator.print(f"[+] Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

    # 5) Data profiling
    accelerator.print("[+] Profiling data pipeline (~200 examples)…")
    start_map = time.time()
    dataloader = get_dataloader(
        csv_file=args.csv_file,
        tokenizer=tokenizer,
        resolution=args.resolution,
        batch_size=args.train_batch_size,
    )
    seen = 0
    for batch in dataloader:
        seen += batch["pixel_values"].shape[0]
        if seen >= 200:
            break
    map_time = time.time() - start_map
    accelerator.print(f"[+] ~{seen} imgs in {map_time:.1f}s → {map_time/seen:.3f}s/img")
    dataloader = iter(dataloader)  # reset iterator for training

    # 6) Optimizer & LR
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,              # "cosine"
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 7) Prepare (moves everything to device)
    (
        unet,
        text_encoder,
        vae,
        optimizer,
        dataloader,
        lr_scheduler,
    ) = accelerator.prepare(unet, text_encoder, vae, optimizer, dataloader, lr_scheduler)

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
    
    accelerator.log({
        "benchmark/time_per_step": sps,
        "benchmark/total_estimated_hours": total_h,
    }, step=0)

    # 8) Training loop
    accelerator.print("[+] Training…")
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break
            t_step = time.time()

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            ts = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.size(0),),
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

            step_time = time.time() - t_step
            global_step += 1

            if global_step % args.logging_steps == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                logs = {
                    "train/loss": loss.item(),
                    "train/learning_rate": current_lr,
                    "train/global_step": global_step,
                    "train/epoch": global_step / len(dataloader),
                }
                accelerator.log(logs, step=global_step)
                accelerator.print(f"Step {global_step}/{args.max_train_steps} — loss {loss.item():.4f}, lr {current_lr:.6f}")

            if global_step % args.checkpointing_steps == 0:
                accelerator.wait_for_everyone()
                ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                unet.save_pretrained(ckpt, safe_serialization=True)
                accelerator.print(f"[+] Saved {ckpt}")
                
                accelerator.log({
                    "checkpoint/step": global_step,
                    "checkpoint/path": ckpt,
                }, step=global_step)

    # 9) Final save & (optional) push
    accelerator.wait_for_everyone()
    unet.save_pretrained(args.output_dir, safe_serialization=True)
    text_encoder.save_pretrained(args.output_dir)
    accelerator.print(f"[+] Done — outputs in {args.output_dir}")
    
    accelerator.log({
        "final/global_step": global_step,
        "final/loss": loss.item(),
    }, step=global_step)

    if args.push_to_hub and args.hub_model_id:
        from huggingface_hub import Repository

        Repository(args.output_dir, clone_from=args.hub_model_id).push_to_hub()
        accelerator.print(f"[+] Pushed to {args.hub_model_id}")
        accelerator.log({
            "hub/pushed": True,
            "hub/model_id": args.hub_model_id,
        }, step=global_step)
    
    if args.log_with == "wandb" and accelerator.is_main_process:
        wandb.finish()
    elif args.log_with != "none":
        accelerator.end_training()

if __name__ == "__main__":
    main()