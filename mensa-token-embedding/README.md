# Textual Inversion: Mensafood Token

This folder contains everything to train a new token embedding `<Mensafood>`
for your Mensa food dataset, using your existing Stable Diffusion model.

## Files

- **train_token_embedding.py**  
  Main script.  
  Usage:
    cd mensa-token-embedding
    python train_token_embedding.py
    --csv /path/to/meals.csv
    --model_name runwayml/stable-diffusion-v1-5
    --output_dir ./
    --resolution 512
    --batch_size 1
    --lr 5e-4
    --steps 1000

this part will:
1. Add the token `"Mensafood"` to the tokenizer.
2. Initialize its embedding (from `"food"` if available).
3. Freeze all other model weights.
4. Prepend `"Mensafood "` to every prompt on-the-fly.
5. Train only the new token embedding.
6. Save `learned_embeds.pt` (and full tokenizer/text encoder) in the folder.