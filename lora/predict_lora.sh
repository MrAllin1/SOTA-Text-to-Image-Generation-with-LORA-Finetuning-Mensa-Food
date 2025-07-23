#!/bin/bash -l
#SBATCH --job-name=lora-infer
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
CONDA_ROOT=/work/dlclarge2/alidemaa-dl_lab/conda/miniconda3
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate dllab

# ── cd to your LoRA project ───────────────────────────────────────────────
cd /work/dlclarge2/alidemaa-dl_lab/mensa_t2i/lora

# ── sanity check GPU ──────────────────────────────────────────────────────
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY

prompts=(
  "Mensafood Chickpea polenta with ratatouille; sheep’s cheese with mint"
  "Mensafood Tortellini filled with ricotta and spinach in basil cheese sauce; spring onions; diced tomatoes; grated Emmental"
  "Mensafood Pasta with tomatoes on top"
  "Mensafood Curry sausage or Planted™ curry sausage; French fries"
)
prompts_german=(
    "Tortellini gefullt mit Ricotta und Spinat Basilikum-Kasesauce Fruhlingszwiebel Tomatenwurfel und geriebener Emmentaler"
    "Pasta-Kreationen aus unserer eigenen Pasta-Manufaktur mit verschiedenen Saucen und Toppings Beilagensalat oder Regio-Apfel"
    "Currywurst oder planted Currywurst Pommes frites"
    "Kichererbsenpolenta mit Ratatouille; Schafskäse mit Minze"
)
# ── inference loop ────────────────────────────────────────────────────────
for prompt in "${prompts_german[@]}"; do
  echo "=========================================="
  echo "🔹 Generating for prompt: $prompt"
  python predict_lora.py \
    --prompt "$prompt" \
    --lora_weights_dir "./lora/lora-adapters-forth-train/checkpoint-30000" \
    --num_images 2 \
    --height 256 \
    --width 256
  echo
done

echo "✅ All done!"
