#!/bin/bash
#SBATCH --job-name=lora-train
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ── load CUDA (adjust to your cluster)
module load cuda/11.7

# ── activate Conda env that has diffusers / accelerate / tensorboard
CONDA_ROOT=/work/dlclarge2/alidemaa-dl_lab/conda/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate dllab

cd /work/dlclarge2/alidemaa-dl_lab/mensa_t2i/lora

# sanity check: PyTorch sees the GPU
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU name :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

# ── training ───────────────────────────────────────────────────────────
export MODEL="runwayml/stable-diffusion-v1-5"

accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path "$MODEL" \
  --csv_file "/work/dlclarge2/alidemaa-dl_lab/mensa_t2i/data/meals_augmented.csv" \
  --resolution 512 \
  --train_batch_size 1 \
  --max_train_steps 30000 \
  --learning_rate 2e-4 \
  --rank 8 \
  --checkpointing_steps 5000 \
  --output_dir "./lora-adapters-forth-train" \
  --push_to_hub \
  --hub_model_id "username/my-lora-model-forth-train"

# ── after (or during) the job ──────────────────────────────────────────
# On the login node or via SSH tunnel:
#   tensorboard --logdir /work/dlclarge2/alidemaa-dl_lab/mensa_t2i/lora/lora-adapters-third-train
#
# Open http://localhost:6006 to watch loss & LR curves live.
