#!/bin/bash
# ===========================================================================
# setup_gpu.sh — One-time GPU environment setup (run interactively on login node)
#
# Creates a .venv with CUDA-enabled PyTorch for GPU training on DTU HPC.
# The pyproject.toml ships with CPU-only wheels; this script overrides
# torch/torchvision to use CUDA 12.1 wheels.
#
# Usage:  bash batch/setup_gpu.sh
# ===========================================================================

set -euo pipefail

REPO_DIR="/zhome/02/6/167678/AdvancedML"
cd "$REPO_DIR"

echo "========================================"
echo "  VAE HPC Environment Setup (GPU)"
echo "========================================"

# ── 1. Ensure uv is available ─────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo "[1/4] uv $(uv --version)"

# ── 2. Create venv with Python 3.12 ──────────────────────────────────
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.12
    echo "[2/4] Created .venv"
else
    echo "[2/4] .venv already exists"
fi

# ── 3. Install project dependencies (CPU torch from pyproject.toml) ───
echo "[3/4] Installing project dependencies ..."
uv sync

# ── 4. Override torch + torchvision with CUDA 12.1 wheels ────────────
echo "[4/4] Replacing CPU torch with CUDA 12.1 build ..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps

echo ""
echo "========================================"
echo "  Setup complete!"
echo "  Verify:  source .venv/bin/activate && python -c 'import torch; print(torch.cuda.is_available())'"
echo "========================================"
