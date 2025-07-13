#!/usr/bin/env python
import argparse, os, glob, sys
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel

def run_lora_inference(
    prompt, base_model_id, lora_checkpoint,
    num_images, height, width, guidance_scale, num_inference_steps
):
    # device & dtype
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available():          device = torch.device("cuda")
    else:                                    device = torch.device("cpu")
    dtype = torch.float16 if device.type=="cuda" else torch.float32

    print(f"[+] Running on {device} (dtype={dtype})")
    if device.type!="cuda":
        print("⚠️  FP16 LoRA only on CUDA; inference will be slower.")

    # tokenizer + base text_encoder from Hub
    clip_model = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    print(f"[+] Loading tokenizer & text encoder `{clip_model}`…")
    tokenizer   = AutoTokenizer.from_pretrained(clip_model, use_fast=False)
    text_encoder= AutoModel.from_pretrained(clip_model, torch_dtype=dtype)

    # base Stable Diffusion pipeline
    print(f"[+] Loading Stable Diffusion `{base_model_id}`…")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        torch_dtype=dtype,
    ).to(device)

    # load your U-Net LoRA adapter
    print(f"[+] Loading U-Net LoRA adapter from `{lora_checkpoint}`…")
    try:
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet,
            lora_checkpoint,
            torch_dtype=dtype
        ).to(device)
    except Exception as e:
        print("❌ Error loading U-Net LoRA:", e)
        sys.exit(1)

    # optional: xformers
    if device.type=="cuda":
        try: pipe.enable_xformers_memory_efficient_attention()
        except: pass

    # inference + save
    print(f"[+] Generating {num_images} image(s) for prompt:\n    {prompt}")
    gen = torch.Generator(device).manual_seed(42)
    out = pipe([prompt]*num_images,
               height=height, width=width,
               guidance_scale=guidance_scale,
               num_inference_steps=num_inference_steps,
               generator=gen)
    images = out.images

    root = "./output"; os.makedirs(root, exist_ok=True)
    existing = glob.glob(os.path.join(root, "prediction_*"))
    idxs = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    nxt = max(idxs+[0]) + 1
    odir = os.path.join(root, f"prediction_{nxt}")
    os.makedirs(odir, exist_ok=True)
    with open(os.path.join(odir, "prompt.txt"), "w") as f: f.write(prompt)
    for i, img in enumerate(images):
        p = os.path.join(odir, f"output_{i}.png")
        img.save(p); print(f"✔️  Saved {p}")
    print(f"[+] Done — images in `{odir}`.")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",            required=True)
    p.add_argument("--base_model_id",     default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora_checkpoint",   required=True,
                   help="folder with adapter_model.safetensors & adapter_config.json")
    p.add_argument("--num_images",        type=int, default=2)
    p.add_argument("--height",            type=int, default=512)
    p.add_argument("--width",             type=int, default=512)
    p.add_argument("--guidance_scale",    type=float, default=7.5)
    p.add_argument("--num_inference_steps", type=int, default=50)
    args = p.parse_args()
    run_lora_inference(
        args.prompt, args.base_model_id, args.lora_checkpoint,
        args.num_images, args.height, args.width,
        args.guidance_scale, args.num_inference_steps
    )
