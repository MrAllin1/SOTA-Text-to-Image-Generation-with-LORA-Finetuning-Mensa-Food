#!/usr/bin/env python
# predict_lora.py

import argparse
import sys
import os
import glob
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def run_lora_inference(
    prompt: str,
    base_model_id: str,
    lora_checkpoint: str,
    num_images: int,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
):
    # 0) device & dtype
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type=="cuda" else torch.float32

    print(f"[+] Running on {device} (dtype={dtype})")
    if device.type!="cuda":
        print("    ⚠️  FP16 LoRA only on CUDA; inference will be slower.")

    # 1) tokenizer + YOUR fine-tuned text_encoder
    from transformers import AutoTokenizer, AutoModel

    multilingual_clip = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    print(f"[+] Loading tokenizer `{multilingual_clip}`…")
    tokenizer = AutoTokenizer.from_pretrained(multilingual_clip, use_fast=False)

    # parent dir of checkpoint-30000 should contain the saved text_encoder
    text_encoder_dir = os.path.dirname(lora_checkpoint)
    print(f"[+] Loading fine-tuned text_encoder from `{text_encoder_dir}`…")
    text_encoder = AutoModel.from_pretrained(
        text_encoder_dir,
        torch_dtype=dtype,
    )

    # 2) base Stable Diffusion pipeline
    print(f"[+] Loading Stable Diffusion `{base_model_id}`…")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        torch_dtype=dtype,
    ).to(device)

    # 3) patch U-Net with your LoRA adapter
    print(f"[+] Loading U-Net LoRA adapter from `{lora_checkpoint}`…")
    try:
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet,
            lora_checkpoint,
            torch_dtype=dtype,
        ).to(device)
    except Exception as e:
        print("❌ Error loading U-Net LoRA:", e)
        sys.exit(1)

    # 4) optional: xformers on CUDA
    if device.type=="cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass

    # 5) inference
    print(f"[+] Generating {num_images} image(s) for prompt:\n    {prompt}")
    generator = torch.Generator(device).manual_seed(42)
    out = pipe(
        [prompt]*num_images,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    images = out.images

    # 6) save
    root = "./output"
    os.makedirs(root, exist_ok=True)
    existing = glob.glob(os.path.join(root, "prediction_*"))
    nums = [int(os.path.basename(d).split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_num = max(nums+[0]) + 1
    out_dir = os.path.join(root, f"prediction_{next_num}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    for idx, img in enumerate(images):
        path = os.path.join(out_dir, f"output_{idx}.png")
        img.save(path)
        print(f"    ✔️  Saved {path}")

    print(f"[+] Done — images in `{out_dir}`.")

def parse_args():
    p = argparse.ArgumentParser("Run Stable Diffusion + LoRA inference")
    p.add_argument("--prompt",         type=str, required=True)
    p.add_argument("--base_model_id",  type=str,
                   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora_checkpoint", type=str, required=True,
                   help="folder with adapter_model.safetensors & adapter_config.json")
    p.add_argument("--num_images",     type=int,   default=3)
    p.add_argument("--height",         type=int,   default=512)
    p.add_argument("--width",          type=int,   default=512)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    return p.parse_args()

if __name__=="__main__":
    args = parse_args()
    run_lora_inference(
        prompt=args.prompt,
        base_model_id=args.base_model_id,
        lora_checkpoint=args.lora_checkpoint,
        num_images=args.num_images,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )
