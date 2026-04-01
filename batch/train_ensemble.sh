#!/bin/bash
# ===========================================================================
# train_ensemble.sh — Train all M retrainings × K decoders for ensemble VAE
#
# Uses LSF job array: each task trains one full retrain (3 decoders).
# Array index 1..10 maps to retrain_0..retrain_9.
#
# Queue: gpua10 (lowest wait times on DTU HPC)
#
# Submit:   bsub < batch/train_ensemble.sh
# ===========================================================================
#BSUB -J ens_train[1-10]
#BSUB -q gpua10
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -o logs/ens_train_%J_%I.out
#BSUB -e logs/ens_train_%J_%I.err

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

mkdir -p logs models/ensemble

# ── Map array index → retrain_id ──────────────────────────────────────
RETRAIN_ID=$(( ${LSB_JOBINDEX} - 1 ))
SAVE_DIR="models/ensemble/retrain_${RETRAIN_ID}"

echo "========================================"
echo "  Job array index : ${LSB_JOBINDEX}"
echo "  Retrain ID      : ${RETRAIN_ID}"
echo "  Save dir        : ${SAVE_DIR}"
echo "  Device          : $(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "  GPU             : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

# ── Train 3 decoders for this retrain ─────────────────────────────────
# Each retrain gets a unique seed derived from RETRAIN_ID
python -c "
import torch, sys
sys.path.insert(0, 'src')
from project2.data import get_data_loaders
from project2.train import train_ensemble

torch.manual_seed(${RETRAIN_ID} * 1000)

train_loader, _, _, _ = get_data_loaders(batch_size=32, data_dir='data/')

train_ensemble(
    num_decoders=3,
    data_loader=train_loader,
    M=2,
    epochs_per_decoder=50,
    lr=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='${SAVE_DIR}',
    verbose=True,
)
"

echo ""
echo "Done: retrain_${RETRAIN_ID}"
