#!/usr/bin/env python
# predict_vanilla.py

import argparse
import sys
import os
import glob
import torch
import pandas as pd
from accelerate import Accelerator
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

accelerator = Accelerator(mixed_precision="fp16", log_with=None)

def run_pretrained_inference(
    text_encoder: CLIPTextModel,
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDPMScheduler,
    prompt: str,
    num_images: int,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
):
    print("[+] Processing text prompt...")
    text_input = tokenizer(
        [prompt] * num_images, 
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    # Prepare latent noise
    generator = torch.Generator(device).manual_seed(42)
    latents = torch.randn(
        (num_images, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
    )

    # Denoising loop
    print(f"[+] Generating {num_images} image(s)...")
    scheduler.set_timesteps(num_inference_steps)
    
    for t in tqdm(scheduler.timesteps):
        latent_model_input = latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample
        
        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    print("[+] Decoding images...")
    latents = latents / 0.18215
    with torch.no_grad():
        images = vae.decode(latents).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    
    root = os.path.join(os.path.dirname(__file__), "..", "data/eval/vanilla_output")
    os.makedirs(root, exist_ok=True)
    
    existing = glob.glob(os.path.join(root, "prediction_*"))
    nums = [
        int(os.path.basename(d).split("_")[1])
        for d in existing
        if os.path.basename(d).split("_")[1].isdigit()
    ]
    next_num = max(nums + [0]) + 1
    out_dir = os.path.join(root, f"prediction_{next_num}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    
    from PIL import Image
    for idx, img in enumerate(images):
        img = Image.fromarray((img * 255).astype("uint8"))
        filename = os.path.join(out_dir, f"output_{idx}.png")
        img.save(filename)
        print(f"    ✔️  Saved {filename}")
    
    print(f"[+] Done — {len(images)} image(s) in `{out_dir}`.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with vanilla VAE/UNet components"
    )
    parser.add_argument(
        "--prompt", type=str, default="None"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--num_images", type=int, default=2,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Image height"
    )
    parser.add_argument(
        "--width", type=int, default=512,
        help="Image width"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="CFG scale"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="Number of denoising steps"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # prompt_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/eval/gt_set/meals_eval_de.csv"))
    prompt_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/eval/gt_set/meals_eval_en.csv"))
    device = accelerator.device
    print(f"[+] Running on {device}")

    print("[+] Loading components...")
    pretrained_path = args.pretrained_model_name_or_path
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet", torch_dtype=torch.float16).to(device)
    scheduler = DDPMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
    text_encoder, vae, unet = accelerator.prepare(text_encoder, vae, unet)

    for i, prompt in enumerate(prompt_csv['description'].tolist()):
        args.prompt = prompt
        print(f"Running inference for prompt {i+1}/{len(prompt_csv)}: {prompt}")
        run_pretrained_inference(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            prompt=args.prompt,
            num_images=args.num_images,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
    )