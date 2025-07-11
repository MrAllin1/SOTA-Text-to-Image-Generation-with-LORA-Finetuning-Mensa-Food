I fine‐tuned the runwayml/stable-diffusion-v1-5 base model with LoRA adapters on the dataset, using the following settings:

LoRA configuration: rank 4, alpha 4

Dataset: 1,667 meal images, 256×256 resolution (CSV input)

Batch size & precision: 1 image per step, mixed‐precision (fp16) via accelerate

Training steps: 15,000 with AdamW optimizer (learning rate 1 × 10⁻⁴)

LR scheduler: cosine decay with 200 warm-up steps

Checkpointing: saved adapter weights every 5,000 steps

Output: final LoRA weights pushed to username/my-lora-model on the Hugging Face Hub.


aka i used this command
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="$MODEL" \
  --csv_file="data/meals_raw_2025-07-09_2024-01-01.csv" \
  --resolution=256 \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-4 \
  --rank=4 \
  --output_dir="./lora-adapters" \
  --push_to_hub \
  --hub_model_id="username/my-lora-model"
