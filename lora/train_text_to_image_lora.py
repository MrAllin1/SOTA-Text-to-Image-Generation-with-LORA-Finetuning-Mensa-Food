#!/usr/bin/env python
"""Fine‑tune Stable Diffusion with LoRA on a CSV dataset.
Adds TensorBoard logging via 🤗 Accelerate’s built‑in tracker.
"""
from __future__ import annotations

import argparse, os, time, torch
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from diffusers.optimization import get_scheduler
from data_loader import get_dataloader  # custom dataloader

# ─────────────────────────── CLI ────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("SD‑LoRA trainer (CSV)")
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--csv_file",                 required=True)
    p.add_argument("--resolution",               type=int,   default=512)
    p.add_argument("--train_batch_size",         type=int,   default=1)
    p.add_argument("--max_train_steps",          type=int,   default=15000)
    p.add_argument("--learning_rate",            type=float, default=2e-4)
    p.add_argument("--lr_scheduler",             default="cosine")
    p.add_argument("--lr_warmup_steps",          type=int,   default=200)
    p.add_argument("--rank",                     type=int,   default=4)
    p.add_argument("--output_dir",               required=True)
    p.add_argument("--checkpointing_steps",      type=int,   default=5000)
    p.add_argument("--push_to_hub",              action="store_true")
    p.add_argument("--hub_model_id")
    return p.parse_args()

# ─────────────────────────── MAIN ───────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Accelerator & TensorBoard tracker
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    accelerator.init_trackers("lora_train", config=vars(args))
    device = accelerator.device
    accelerator.print(f"[+] Running on {device}")

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
    unet = get_peft_model(
        unet,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        ),
    )
    trainable_params = [p for p in unet.parameters() if p.requires_grad]

    # 5) Quick data pipeline profile
    accelerator.print("[+] Profiling dataloader (~200 samples)…")
    t0, seen = time.time(), 0
    for batch in get_dataloader(
        args.csv_file, tokenizer, args.resolution, args.train_batch_size
    ):
        seen += batch["pixel_values"].shape[0]
        if seen >= 200:
            break
    accelerator.print(f"    → {seen} imgs in {time.time() - t0:.1f}s")

    dataloader = get_dataloader(
        args.csv_file, tokenizer, args.resolution, args.train_batch_size
    )

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

    # helper for TensorBoard
    def log_scalars(step, loss_val, dt):
        accelerator.log(
            {
                "train/loss": loss_val,
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/step_time_s": dt,
            },
            step=step,
        )

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

            if global_step % 10 == 0:
                accelerator.print(
                    f"[step {global_step}] loss={loss.item():.4f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                )
                log_scalars(global_step, loss.item(), step_time)

            if global_step % args.checkpointing_steps == 0:
                accelerator.wait_for_everyone()                                                                                 
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                unet.save_pretrained(ckpt_dir, safe_serialization=True)
                accelerator.print(f"[+] Saved {ckpt_dir}")

    # 9) Final save & (optional) push
    accelerator.wait_for_everyone()
    unet.save_pretrained(args.output_dir, safe_serialization=True)
    text_encoder.save_pretrained(args.output_dir)
    accelerator.print("[+] Training complete! Artifacts in", args.output_dir)

    if args.push_to_hub and args.hub_model_id:
        from huggingface_hub import Repository

        Repository(args.output_dir, clone_from=args.hub_model_id).push_to_hub()
        accelerator.print(f"[+] Pushed to {args.hub_model_id}")


if __name__ == "__main__":
    main()
