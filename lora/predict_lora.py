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
    lora_weights_dir: str,
    num_images: int,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
):
    # 0) pick device in priority: mps (macOS) > cuda > cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # pick dtype: fp16 only on cuda; otherwise fp32
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[+] Running on {device} (dtype={dtype})")
    if device.type != "cuda":
        print("    ⚠️  Warning: this will be slower than GPU, and LoRA in FP16 only runs on CUDA.")

    # 1) load base pipeline
    print(f"[+] Loading base model `{base_model_id}` in {dtype}…")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    # 2) wrap unet with your LoRA adapter
    print(f"[+] Loading LoRA weights from `{lora_weights_dir}`…")
    try:
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet,
            lora_weights_dir,
            torch_dtype=dtype,
        )
        pipe.unet = pipe.unet.to(device)
    except Exception as e:
        print("❌ Error loading LoRA weights:", e)
        print(
            "\nIf you see a safetensors `HeaderTooLarge` error,\n"
            " re-run training with `safe_serialization=False` so your adapter is saved as a .bin,\n"
            " or switch to a CUDA device for FP16 support."
        )
        sys.exit(1)

    # optional: memory‐efficient attention (xformers) on CUDA
    if device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass

    # 3) inference
    print(f"[+] Generating {num_images} image(s) for prompt:\n    {prompt}")
    generator = torch.Generator(device).manual_seed(42)
    output = pipe(
        [prompt] * num_images,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    images = output.images

    # ← new: auto-make a fresh output folder under ./output
    root = "./output"
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

    # save the prompt
    with open(os.path.join(out_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    # 4) save outputs
    for idx, img in enumerate(images):
        filename = os.path.join(out_dir, f"output_{idx}.png")
        img.save(filename)
        print(f"    ✔️  Saved {filename}")

    print(f"[+] Done — {len(images)} image(s) in `{out_dir}`.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with Stable Diffusion + LoRA (on MPS/CUDA/CPU)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Your menu description, e.g. 'Tortellini gefüllt mit Ricotta...'"
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace repo of the base Stable Diffusion model"
    )
    parser.add_argument(
        "--lora_weights_dir",
        type=str,
        default="./lora-adapters",
        help="Directory where your LoRA adapters live"
    )
    parser.add_argument(
        "--num_images", type=int, default=3,
        help="How many images to generate"
    )
    parser.add_argument(
        "--height", type=int, default=256,
        help="Image height (must match fine-tune)"
    )
    parser.add_argument(
        "--width", type=int, default=256,
        help="Image width (must match fine-tune)"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="Denoising steps"
    )
    return parser.parse_args()

if __name__ == "__main__":
    opts = parse_args()
    run_lora_inference(
        prompt=opts.prompt,
        base_model_id=opts.base_model_id,
        lora_weights_dir=opts.lora_weights_dir,
        num_images=opts.num_images,
        height=opts.height,
        width=opts.width,
        guidance_scale=opts.guidance_scale,
        num_inference_steps=opts.num_inference_steps,
    )
