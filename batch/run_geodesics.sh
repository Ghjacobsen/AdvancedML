#!/bin/bash
# ===========================================================================
# run_geodesics.sh — Compute geodesics + CoV analysis from trained models
#
# Run AFTER train_ensemble.sh completes.
# Computes:
#   1. Part A geodesics (K=1 single decoder) 
#   2. Part B geodesics (K=1,2,3 ensemble)
#   3. CoV analysis across 10 retrainings
#   4. All plots (saved to models/ensemble/results/)
#
# Submit:   bsub < batch/run_geodesics.sh
# ===========================================================================
#BSUB -J ens_geodesics
#BSUB -q gpua10
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o logs/ens_geodesics_%J.out
#BSUB -e logs/ens_geodesics_%J.err

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────
REPO_DIR="/work3/s214374/AdvancedML"
cd "$REPO_DIR"

module load python3/3.12.7
module load cuda/12.2.2
module load cudnn/v9.1.1.17-prod-cuda-12.X

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src:$PYTHONPATH"

mkdir -p logs models/ensemble/results

DEVICE=$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')
echo "========================================"
echo "  Ensemble Geodesics + CoV"
echo "  Device: ${DEVICE}"
echo "========================================"

# ── 1. Compute geodesics (Part A: K=1, Part B: K=1,2,3) ──────────────
echo ""
echo "=== Stage 1: Geodesic computation ==="
python src/project2/experiment.py geodesics \
    --save-dir models/ensemble \
    --num-decoders 3 \
    --num-curves 25 \
    --num-t 20 \
    --geo-lr 0.01 \
    --geo-steps 1000 \
    --num-mc 50 \
    --retrain-idx 0 \
    --device "$DEVICE" \
    --verbose

# ── 2. CoV analysis across retrainings ────────────────────────────────
echo ""
echo "=== Stage 2: CoV analysis ==="
python src/project2/experiment.py cov \
    --save-dir models/ensemble \
    --num-decoders 3 \
    --num-reruns 10 \
    --num-cov-pairs 10 \
    --num-t 20 \
    --geo-lr 0.01 \
    --geo-steps 1000 \
    --num-mc 50 \
    --device "$DEVICE" \
    --verbose

echo ""
echo "Done. Results in models/ensemble/results/"
ls -la models/ensemble/results/
