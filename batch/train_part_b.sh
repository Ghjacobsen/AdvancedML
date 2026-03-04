#!/bin/bash
# ===========================================================================
# train_part_b.sh — LSF single job: Part B sampling quality experiments
#
# Trains (sequentially in one GPU job):
#   1. Image-space DDPM (U-Net, T=1000, 100 epochs) on standard MNIST
#   2. Gaussian VAE  (latent_dim=20, 50 epochs) on standard MNIST
#   3. Latent DDPM   (MLP, T=1000, 100 epochs) on VAE latent codes
#   4. Evaluation: FID scores, FID vs T, sampling times, all plots
#
# Prerequisites:  bash batch/setup_gpu.sh   (one-time, on login node)
# Submit:         bsub < batch/train_part_b.sh
# ===========================================================================
#BSUB -J part_b_train
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o logs/part_b_%J.out
#BSUB -e logs/part_b_%J.err

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────
REPO_DIR="$HOME/AdvancedML"
cd "$REPO_DIR"

module load python3/3.12.7
module load cuda/12.2.2
module load cudnn/v9.1.1.17-prod-cuda-12.X
module load nccl/2.21.5-1-cuda-12.2.2

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

mkdir -p logs models reports/figures

echo "========================================"
echo "  Part B: Sampling Quality Experiments"
echo "  Device: $(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "========================================"

# ── Run full Part B pipeline ───────────────────────────────────────────
# --skip-training reuses existing DDPM/VAE/LatentDDPM checkpoints;
# the β sweep always trains fresh models per β in models/beta_sweep/.
python src/project/run_part_b.py \
    --ddpm-epochs 100 \
    --vae-epochs 50 \
    --latent-ddpm-epochs 100 \
    --latent-dim 20 \
    --hidden-dim 256 \
    --ddpm-base-channels 64 \
    --T 1000 \
    --batch-size 128 \
    --n-fid-samples 5000 \
    --part-a-prior mog \
    --output-dir "./reports" \
    --models-dir "./models" \
    --data-dir "./data" \
    --skip-training

echo ""
echo "Part B complete. Results in reports/part_b_results.json"
