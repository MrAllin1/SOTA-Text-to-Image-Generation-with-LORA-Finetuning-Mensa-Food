#!/usr/bin/env python
# infer_mensafood.py

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from transformers import CLIPTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load base Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    ).to(device)

    # 2) Apply LoRA adapter
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet,
        "/work/dlclarge2/latifajr-dl_lab_project/mensa_t2i/lora/lora-adapters-forth-train/checkpoint-30000",
        torch_dtype=pipe.unet.dtype
    ).to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass

    # 3) Add & resize for the Mensafood token
    tokenizer = pipe.tokenizer
    token = "Mensafood"
    if token not in tokenizer.get_vocab():
        tokenizer.add_tokens([token])
        pipe.text_encoder.resize_token_embeddings(len(tokenizer))

    # 4) Inject learned embedding
    state = torch.load("learned_embeds.pt", map_location="cpu")
    emb = state[token].to(device)
    token_id = tokenizer.convert_tokens_to_ids(token)
    with torch.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[token_id] = emb

    # 5) Prompts (German & English)
    prompts = [
        "Mensafood Chickpea polenta with ratatouille; sheep’s cheese with mint",
        "Mensafood Tortellini filled with ricotta and spinach in basil cheese sauce; spring onions; diced tomatoes; grated Emmental",
        "Mensafood Pasta with tomatoes on top",
        "Mensafood Curry sausage or Planted™ curry sausage; French fries",
        "Mensafood Tortellini gefullt mit Ricotta und Spinat Basilikum-Kasesauce Fruhlingszwiebel Tomatenwurfel und geriebener Emmentaler",
        "Mensafood Pasta-Kreationen aus unserer eigenen Pasta-Manufaktur mit verschiedenen Saucen und Toppings Beilagensalat oder Regio-Apfel",
        "Mensafood Currywurst oder planted Currywurst Pommes frites",
        "Mensafood Kichererbsenpolenta mit Ratatouille; Schafskäse mit Minze",
    ]

    # 6) Generate & save
    for p in prompts:
        images = pipe(
            p,
            height=512,
            width=512,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images
        safe_name = p.replace(" ", "_").replace(";", "").replace("’","'")[:80]
        out_path = f"output_{safe_name}.png"
        images[0].save(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
