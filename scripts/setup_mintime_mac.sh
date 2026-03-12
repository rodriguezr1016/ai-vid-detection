#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINTIME_DIR="$ROOT_DIR/vendor/mintime"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if [[ ! -d "$MINTIME_DIR" ]]; then
  echo "Missing MINTIME checkout at $MINTIME_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$MINTIME_DIR/.venv-mintime"
source "$MINTIME_DIR/.venv-mintime/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Use the regular PyTorch wheels on macOS. They run on CPU and can use MPS when available.
python -m pip install torch torchvision torchaudio

python -m pip install \
  albumentations==0.5.2 \
  efficientnet-pytorch==0.7.1 \
  einops==0.6.0 \
  facenet-pytorch==2.5.3 \
  imageio==2.37.0 \
  imgaug==0.4.0 \
  matplotlib==3.8.4 \
  networkx==3.3 \
  numpy==1.26.3 \
  opencv-python==4.11.0.86 \
  pandas==2.2.3 \
  pillow==10.2.0 \
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

echo "MINTIME macOS environment ready at $MINTIME_DIR/.venv-mintime"
echo "This macOS path skips PyAV/pytorchvideo because MINTIME prediction does not require them."
echo "Set MINTIME_DEVICE=cpu in .env first. If MPS works for your machine, try MINTIME_DEVICE=mps later."
