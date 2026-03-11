#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINTIME_DIR="$ROOT_DIR/vendor/mintime"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
ROCM_TORCH_INDEX_URL="${ROCM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}"

if [[ ! -d "$MINTIME_DIR" ]]; then
  echo "Missing MINTIME checkout at $MINTIME_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$MINTIME_DIR/.venv-mintime"
source "$MINTIME_DIR/.venv-mintime/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install ROCm-compatible PyTorch first, then add the libraries MINTIME needs.
python -m pip install \
  torch torchvision torchaudio \
  --index-url "$ROCM_TORCH_INDEX_URL"

python -m pip install \
  albumentations==0.5.2 \
  av==10.0.0 \
  efficientnet-pytorch==0.7.1 \
  einops==0.6.0 \
  facenet-pytorch==2.5.3 \
  imageio==2.22.4 \
  imgaug==0.4.0 \
  matplotlib==3.8.4 \
  networkx==3.3 \
  numpy==1.26.3 \
  opencv-python==4.11.0.86 \
  pandas==2.2.3 \
  pillow==10.2.0 \
  pytorchvideo==0.1.5 \
  pyyaml==6.0.2 \
  scikit-image==0.22.0 \
  scikit-learn==1.5.2 \
  scipy==1.13.1 \
  shapely==2.0.7 \
  tabulate==0.9.0 \
  termcolor==2.5.0 \
  torchsummary==1.5.1 \
  tqdm==4.67.1 \
  yacs==0.1.8

echo "MINTIME ROCm environment ready at $MINTIME_DIR/.venv-mintime"
echo "Next: download MINTIME weights into $MINTIME_DIR/weights and point .env at them."
