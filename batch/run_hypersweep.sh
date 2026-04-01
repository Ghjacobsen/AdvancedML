#!/bin/bash
# ===========================================================================
# run_hypersweep.sh — Grid search over geodesic optimizer hyperparameters
#
# Uses already-trained models. Tries different (lr, steps, mc, num_t, decay)
# combos and ranks by CoV behavior. Auto-stops at ~55 min.
#
# Submit:   bsub < batch/run_hypersweep.sh
# ===========================================================================
#BSUB -J geo_sweep
#BSUB -q gpua10
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -o logs/geo_sweep_%J.out
#BSUB -e logs/geo_sweep_%J.err

set -euo pipefail

REPO_DIR="/work3/s214374/AdvancedML"
cd "$REPO_DIR"

module load python3/3.12.7
module load cuda/12.2.2
module load cudnn/v9.1.1.17-prod-cuda-12.X

export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src:$PYTHONPATH"

mkdir -p logs

DEVICE=$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')
echo "Device: $DEVICE"
echo "Start: $(date)"

python src/project2/hypersweep.py \
    --save-dir models/ensemble \
    --num-reruns 10 \
    --num-decoders 3 \
    --num-pairs 10 \
    --device "$DEVICE" \
    --max-time 3300

echo "End: $(date)"
