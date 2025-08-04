#!/bin/bash -l
#SBATCH --job-name=mensa-infer
#SBATCH --partition=dllabdlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:15:00

# Load your Conda environment
CONDA_ROOT=/work/dlclarge2/latifajr-dl_lab_project/conda/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate dl_lab_env

# Move into the folder containing infer_mensafood.py
cd /work/dlclarge2/latifajr-dl_lab_project/mensa_t2i/mensa-token-embedding

# Run the inference script
python infer_mensafood.py
