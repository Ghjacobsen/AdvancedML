#!/bin/bash
# ===========================================================================
# train_vae.sh — LSF job array: train VAE with 3 priors × 3 seeds = 9 jobs
#
# Each array task trains a single (prior, seed) combination.
# Task mapping:
#   1-3  →  gaussian  (seeds 42, 43, 44)
#   4-6  →  mog       (seeds 42, 43, 44)
#   7-9  →  flow      (seeds 42, 43, 44)
#
# Prerequisites:  bash batch/setup_gpu.sh   (one-time, on login node)
# Submit:         bsub < batch/train_vae.sh
# ===========================================================================
#BSUB -J vae_train[1-9]
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -o logs/vae_%J_%I.out
#BSUB -e logs/vae_%J_%I.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────
REPO_DIR="$HOME/AdvancedML"
cd "$REPO_DIR"

module load python3/3.12.7
module load cuda/12.2.2
module load cudnn/v9.1.1.17-prod-cuda-12.X
module load nccl/2.21.5-1-cuda-12.2.2

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

mkdir -p logs models reports/figures

# ── Map array index → (prior, seed) ──────────────────────────────────
PRIORS=("gaussian" "gaussian" "gaussian" "mog" "mog" "mog" "flow" "flow" "flow")
SEEDS=(42 43 44 42 43 44 42 43 44)

IDX=$(( ${LSB_JOBINDEX} - 1 ))
PRIOR="${PRIORS[$IDX]}"
SEED="${SEEDS[$IDX]}"
RUN_ID=$(( (IDX % 3) ))

SAVE_DIR="models/${PRIOR}/run_${RUN_ID}"

echo "========================================"
echo "  Job array index : ${LSB_JOBINDEX}"
echo "  Prior           : ${PRIOR}"
echo "  Seed            : ${SEED}"
echo "  Run ID          : ${RUN_ID}"
echo "  Save dir        : ${SAVE_DIR}"
echo "  Device          : $(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "========================================"

# ── Train ─────────────────────────────────────────────────────────────
python src/project/train.py \
    --prior "$PRIOR" \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-3 \
    --latent-dim 20 \
    --hidden-dim 256 \
    --seed "$SEED" \
    --save-dir "$SAVE_DIR" \
    --data-dir "./data"

echo ""
echo "Done: ${PRIOR} prior, seed=${SEED}"
