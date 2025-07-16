#!/bin/bash
#SBATCH --job-name=pretrain_diffusion_test
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# load CUDA drivers (pick your installed version)
module load cuda/11.7

#! Change to your own paths
cd /work/dlclarge2/shengw-sw_sapce
source venv_mensa/bin/activate

cd /work/dlclarge2/shengw-sw_sapce/mensa_t2i/vanilla

# double-check PyTorch sees the GPU
python - <<EOF
import torch
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
EOF

# launch with Accelerate
export MODEL="runwayml/stable-diffusion-v1-5"
accelerate launch pretrain_t2i_diffusion.py \
  --pretrained_model_name_or_path="$MODEL" \
  --csv_file="/work/dlclarge2/shengw-sw_sapce/mensa_t2i/data/meals_translated_en_with_gpt_03.csv" \
  --resolution=512 \
  --batch_size=1 \
  --output_dir="./output/" \
  --log_with="none" \
  
