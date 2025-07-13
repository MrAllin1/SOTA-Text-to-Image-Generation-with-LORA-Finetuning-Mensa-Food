#!/usr/bin/env python
# train_text_to_image_lora_multilang.py
"""
Fine-tune Stable Diffusion (LoRA) with an optional multilingual text-encoder.

Typical call:
--------------
accelerate launch train_text_to_image_lora_multilang.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1 \
  --text_encoder_name_or_path M-CLIP/XLM-Roberta-Large-Vit-L-14 \
  --tokenizer_name_or_path M-CLIP/XLM-Roberta-Large-Vit-L-14 \
  --csv_file data/meals.csv \
  --output_dir ./mensa_lora_multilang
"""
import argparse, os, time, torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from diffusers.optimization import get_scheduler
from data_loader import get_dataloader

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("LoRA fine-tune with multilingual text-encoder")
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--csv_file", required=True)
    p.add_argument("--output_dir", required=True)

    # NEW ↓
    p.add_argument("--text_encoder_name_or_path", default=None,
                   help="HuggingFace repo for the *multilingual* text-encoder "
                        "(defaults to the one inside --pretrained_model_name_or_path)")
    p.add_argument("--tokenizer_name_or_path", default=None,
                   help="Tokenizer to pair with the multilingual encoder "
                        "(defaults likewise)")

    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=15000)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lr_scheduler", default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=200)

    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lora_text", action="store_true",
                   help="also apply LoRA to the text-encoder")

    p.add_argument("--checkpointing_steps", type=int, default=5000)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", type=str)
    return p.parse_args()

# ---------- Projector that fixes hidden-size mismatches ----------
class TextProj(torch.nn.Module):
    """Maps text hidden states from <in_dim> → <out_dim> when they differ."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.need_proj = in_dim != out_dim
        if self.need_proj:
            self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, hidden):
        return self.proj(hidden) if self.need_proj else hidden

# ---------- main ----------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    accelerator.print(f"[+] Using {device}")

    # 1) Load tokenizer / text-encoder (multilingual if given)
    tok_path = args.tokenizer_name_or_path or args.text_encoder_name_or_path \
               or args.pretrained_model_name_or_path
    txt_path = args.text_encoder_name_or_path or args.pretrained_model_name_or_path

    tokenizer = CLIPTokenizer.from_pretrained(tok_path, subfolder="tokenizer" 
                                              if tok_path==args.pretrained_model_name_or_path else "")
    text_encoder = CLIPTextModel.from_pretrained(txt_path, subfolder="text_encoder"
                                              if txt_path==args.pretrained_model_name_or_path else "")

    # 2) Load VAE & U-Net from the Stable Diffusion base
    vae  = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch.float16
    )
    unet.set_attention_slice("auto")
    unet.enable_gradient_checkpointing()

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # 3) Bridge hidden-size mismatch (e.g. 1024 ← 768)
    proj = TextProj(text_encoder.config.hidden_size, unet.config.cross_attention_dim)

    # 4) LoRA on U-Net (and optionally on text-encoder)
    unet = prepare_model_for_kbit_training(unet)
    unet = get_peft_model(
        unet,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )

    if args.lora_text:
        text_encoder = prepare_model_for_kbit_training(text_encoder)
        text_encoder = get_peft_model(
            text_encoder,
            LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=["k_proj","q_proj","v_proj","out_proj"],  # CLIP-style names
            )
        )

    # Collect trainable params  
    trainable_params  = list(filter(lambda p: p.requires_grad, unet.parameters()))
    trainable_params += list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    if proj.parameters():
        trainable_params += proj.parameters()

    # 5) Data
    dataloader = get_dataloader(
        csv_file=args.csv_file,
        tokenizer=tokenizer,
        resolution=args.resolution,
        batch_size=args.train_batch_size,
    )

    # 6) Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.learning_rate, betas=(0.9,0.999), weight_decay=1e-2
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 7) Push to device
    unet, text_encoder, proj, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, proj, optimizer, dataloader, lr_scheduler
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
            if global_step >= args.max_train_steps: break
            pixel_values = batch["pixel_values"].to(device)
            input_ids    = batch["input_ids"].to(device)

            latents = vae.encode(pixel_values).latent_dist.sample()*0.18215
            noise   = torch.randn_like(latents)
            ts      = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                    (latents.size(0),), device=device)
            noisy   = noise_scheduler.add_noise(latents, noise, ts)
            hidden  = text_encoder(input_ids)[0]
            hidden  = proj(hidden)                  # <<< dimension-fix
            pred    = unet(noisy, ts, hidden).sample
            loss    = torch.nn.functional.mse_loss(pred, noise)

            accelerator.backward(loss)
            optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
            global_step += 1

            if global_step % 100 == 0:
                accelerator.print(f"{global_step}/{args.max_train_steps}  loss={loss.item():.4f}")
            if global_step % args.checkpointing_steps == 0:
                accelerator.wait_for_everyone()
                unet.save_pretrained(os.path.join(args.output_dir, f"ckpt-{global_step}"),
                                     safe_serialization=True)

    # 9) Save
    accelerator.wait_for_everyone()
    unet.save_pretrained(args.output_dir, safe_serialization=True)
    text_encoder.save_pretrained(args.output_dir)
    proj_path = os.path.join(args.output_dir,"text_hidden_proj.pt")
    torch.save(proj.state_dict(), proj_path)
    accelerator.print(f"✔️  Finished. LoRA + proj saved to {args.output_dir}")

    if args.push_to_hub and args.hub_model_id:
        from huggingface_hub import Repository
        repo = Repository(args.output_dir, clone_from=args.hub_model_id)
        repo.push_to_hub()

if __name__ == "__main__":
    main()
