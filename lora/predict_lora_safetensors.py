#!/usr/bin/env python
# predict_lora.py

import argparse
import sys
import os
import glob
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def run_lora_inference(args):
    prompt = args.prompt
    base_model_id = args.base_model_id
    lora_weights_dir = args.lora_weights_dir
    num_images = args.num_images
    height = args.height
    width = args.width
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_steps
    output_dir = args.output_dir
    
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

    # 2) load and apply LoRA weights
    print(f"[+] Loading LoRA weights from `{lora_weights_dir}`…")
    config_file = os.path.join(lora_weights_dir, "adapter_config.json")
    model_file = os.path.join(lora_weights_dir, "adapter_model.safetensors")
    
    if os.path.exists(config_file):
        try:
            pipe.unet = PeftModel.from_pretrained(
                pipe.unet,
                lora_weights_dir,
                torch_dtype=dtype,
                weight_name=model_file
            )
        except Exception as e:
            print("❌ Error loading safetensors file:", e)
            sys.exit(1)
        if hasattr(pipe, 'get_list_adapters'):
            print(f"LoRA adapters loaded: {pipe.get_list_adapters()}")

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

    # ← auto-make a fresh output folder under ./output
    root = os.path.join(os.path.dirname(__file__), "../data/eval", output_dir)
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
        required=True,
        help="Path to your LoRA weights file (.safetensors or .bin)"
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
    parser.add_argument(
        "--output_dir", type=str, default="./lora_output",
        help="Relative directory under data to save output images"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    import pandas as pd
    # prompt_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/eval/gt_set/meals_eval_de.csv"))
    prompt_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/eval/gt_set/meals_eval_en.csv"))

    for i, prompt in enumerate(prompt_csv['description'].tolist()):
        args.prompt = prompt
        print(f"Running inference for prompt {i+1}/{len(prompt_csv)}: {prompt}")
        run_lora_inference(args)