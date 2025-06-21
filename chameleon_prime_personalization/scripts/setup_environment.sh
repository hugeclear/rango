#!/usr/bin/env bash
set -e

ENV_DIR=".venv"

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

# Detect CUDA
if python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    CUDA_SPEC="cuda"
else
    CUDA_SPEC="cpu"
fi

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_SPEC
pip install transformers datasets peft sentence-transformers nevergrad scikit-learn matplotlib

