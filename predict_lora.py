#!/usr/bin/env python
# predict_lora.py

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def run_lora_inference(
    base_model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_weights_dir: str = "./lora-adapters",
    prompt: str = "a delicious gourmet meal, photography",
    num_images: int = 3,
    height: int = 256,
    width: int = 256,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
):
    # 0) Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Running on {device}")
    if device == "cpu":
        print("    ⚠️  Warning: running on CPU will be very slow.")

    # 1) Load the base pipeline in half-precision & move to device
    print(f"[+] Loading base model `{base_model_id}`…")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
    ).to(device)

    # 2) Wrap its UNet with your LoRA adapters and move that to device too
    print(f"[+] Loading LoRA weights from `{lora_weights_dir}`…")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_weights_dir).to(device)

    # optional: enable memory-efficient attention if your diffusers build supports it
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # 3) Run inference
    print(f"[+] Generating {num_images} image(s) for prompt: {prompt!r}")
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

    # 4) Save to disk
    for idx, img in enumerate(images):
        out_path = f"lora_output_{idx}.png"
        img.save(out_path)
        print(f"    ✔️  Saved {out_path}")
    print(f"[+] Done—{len(images)} image(s) written.")

if __name__ == "__main__":
    run_lora_inference(
        base_model_id="runwayml/stable-diffusion-v1-5",
        lora_weights_dir="./lora-adapters",
        prompt="a photorealistic portrait of a futuristic city at sunset",
        num_images=4,
        height=256,   # match your fine-tune resolution
        width=256,
        guidance_scale=7.0,
        num_inference_steps=50,
    )
