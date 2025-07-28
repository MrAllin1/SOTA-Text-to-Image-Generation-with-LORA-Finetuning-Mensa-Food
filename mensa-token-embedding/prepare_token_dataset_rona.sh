#!/bin/bash -l
#SBATCH --job-name=tokenization
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-infer-%j.out
#SBATCH --error=slurm-infer-%j.err

# ── load your login profile so “module” is available ────────────────────────
# note the “-l” in the shebang above; that makes this a login shell

# ── load CUDA (adjust to your cluster) ────────────────────────────────────
module load cuda/11.7

# ── activate Conda env with diffusers/accelerate etc ──────────────────────
CONDA_ROOT=/work/dlclarge2/latifajr-dl_lab_project/conda/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate dl_lab_env

cd /work/dlclarge2/latifajr-dl_lab_project/mensa_t2i/mensa-token-embedding

python prepare_token_dataset.py \
  --input_csv /work/dlclarge2/latifajr-dl_lab_project/mensa_t2i/data/meals_augmented.csv \
  --output_csv meals_for_token.csv
