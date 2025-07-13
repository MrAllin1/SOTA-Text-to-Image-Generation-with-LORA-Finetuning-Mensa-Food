#!/bin/bash
#SBATCH --job-name=lora-prediction-test
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# load CUDA drivers
module load cuda/11.7

# activate your conda
CONDA_ROOT=/work/dlclarge2/alidemaa-dl_lab/conda/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate dllab

# go to your SD+LoRA inference folder
cd /work/dlclarge2/alidemaa-dl_lab/mensa_t2i/lora

# verify GPU (optional)
python - <<EOF
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
EOF

# --- user params ---
LORA_CKPT="/work/dlclarge2/alidemaa-dl_lab/mensa_t2i/lora/lora-adapters-third-train/checkpoint-30000"
NUM_IMAGES=2
HEIGHT=512
WIDTH=512

# list your prompts here
PROMPTS=(
  "Borek mit Spinat Karottencurry Erbsenpuree"
  "Tortellini gefüllt mit Ricotta und Spinat, dazu Tomatensauce"
  "Currywurst mit Pommes und veganer Mayo"
)

# loop over prompts
for prompt in "${PROMPTS[@]}"; do
  echo "▶️  Generating for prompt: $prompt"
  python predict_lora.py \
    --prompt "$prompt" \
    --base_model_id "runwayml/stable-diffusion-v1-5" \
    --lora_checkpoint "$LORA_CKPT" \
    --num_images "$NUM_IMAGES" \
    --height "$HEIGHT" \
    --width "$WIDTH"
done
