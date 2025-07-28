#!/usr/bin/env python
"""
train_token_embedding.py

Textual Inversion: train a single token embedding ("Mensafood")
for your mensa‑food image dataset, without touching your LoRA pipeline.
"""

import os
import argparse
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from pathlib import Path

class TextualInversionDataset(Dataset):
    def __init__(self, df, tokenizer, resolution, data_root):
        self.df = df
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.caption_col = "description"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = row["image_path"]
        path = os.path.join(self.data_root, rel)
        if not os.path.exists(path):
            fname = Path(rel).name
            matches = list(Path(self.data_root).rglob(fname))
            if not matches:
                raise FileNotFoundError(f"Cannot find {rel} in {self.data_root}")
            path = str(matches[0])
        img = Image.open(path).convert("RGB")
        pixel_values = self.transform(img)

        prompt = "Mensafood " + str(row[self.caption_col])
        toks = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": toks.input_ids.squeeze(),
            "attention_mask": toks.attention_mask.squeeze(),
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        required=True, help="CSV with image_path + description")
    parser.add_argument("--data_root",  required=True, help="Root dir for images")
    parser.add_argument("--model_name", required=True, help="e.g. runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", default=".",    help="Where to save learned_embeds.pt")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--steps",      type=int, default=1000)
    parser.add_argument("--device",     default=None, help="cuda or cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1) Tokenizer + new token
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_name, subfolder="tokenizer", mean_resizing=False
    )
    new_token = "Mensafood"
    if new_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([new_token])
        print(f"Added token '{new_token}' → vocab size {len(tokenizer)}")
    else:
        print(f"Token '{new_token}' already exists")

    # 2) Text encoder
    text_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder")
    text_encoder.resize_token_embeddings(len(tokenizer))
    new_id = tokenizer.convert_tokens_to_ids(new_token)

    # Optional init from 'food'
    if "food" in tokenizer.get_vocab():
        fid = tokenizer.convert_tokens_to_ids("food")
        with torch.no_grad():
            emb = text_encoder.get_input_embeddings().weight
            emb[new_id] = emb[fid].clone()
        print("Initialized new token from 'food'")

    # 3) Load & freeze UNet, VAE, scheduler
    unet = UNet2DConditionModel.from_pretrained(
        args.model_name, subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae").to(device)
    sched = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    for p in unet.parameters(): p.requires_grad = False
    for p in vae.parameters():  p.requires_grad = False

    # Freeze all text_encoder except embedding matrix
    for name, p in text_encoder.named_parameters():
        p.requires_grad = False
    text_encoder.get_input_embeddings().weight.requires_grad = True
    text_encoder.to(device)

    # 4) DataLoader
    df = pd.read_csv(args.csv)
    dataset = TextualInversionDataset(df, tokenizer, args.resolution, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 5) Optimizer on the embedding only
    emb_param = text_encoder.get_input_embeddings().weight
    optimizer = torch.optim.AdamW([emb_param], lr=args.lr)

    # 6) Training loop
    step = 0
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            pv = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)

            # Compute text embeddings with grad so embedding can update
            te_out = text_encoder(input_ids=ids, attention_mask=am)
            txt_embeds = te_out.last_hidden_state.to(device=device, dtype=unet.dtype)

            # Encode images under no_grad
            with torch.no_grad():
                lat = vae.encode(pv).latent_dist.sample() * 0.18215
                lat = lat.to(device=device, dtype=unet.dtype)

            noise = torch.randn_like(lat)
            ts = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=device).long()
            noisy = sched.add_noise(lat, noise, ts).to(device=device, dtype=unet.dtype)

            # UNet forward under autocast
            with torch.amp.autocast(device.type, enabled=(device.type=="cuda")):
                pred = unet(noisy, ts, encoder_hidden_states=txt_embeds).sample
                loss = torch.nn.functional.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()

            # Zero out all grads except for our new token
            g = emb_param.grad
            if g is not None:
                mask = torch.ones_like(g)
                mask[new_id] = 0
                g.mul_(mask)

            optimizer.step()
            step += 1

            if step % 100 == 0:
                print(f"[step {step}/{args.steps}] loss={loss.item():.6f}")

    # 7) Save the learned embedding
    learned = emb_param[new_id].detach().cpu()
    torch.save({new_token: learned}, os.path.join(args.output_dir, "learned_embeds.pt"))
    print("Saved learned_embeds.pt")

if __name__ == "__main__":
    main()
