#!/usr/bin/env bash
# One-command launcher for PlasmidGPT Studio on macOS/Linux.
# Double-click in Finder (after "chmod +x start.sh") or run: ./start.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR=".venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo
  echo "First-time setup: creating PlasmidGPT's local Python environment..."
  echo "This normally takes a few minutes and happens only once."
  echo

  if command -v python3.10 >/dev/null 2>&1; then
    PY=python3.10
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    echo
    echo "Python 3.10+ was not found on this computer."
    echo "Install it from https://www.python.org/downloads/ and run this script again."
    echo
    exit 1
  fi

  "$PY" -m venv "$VENV_DIR"
fi

if ! "$PYTHON_BIN" -c "import streamlit" >/dev/null 2>&1; then
  echo
  echo "First-time setup: installing PlasmidGPT's dependencies..."
  echo "This normally takes a few minutes and happens only once."
  echo
  "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
  "$PYTHON_BIN" -m pip install -r requirements-app.txt
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! "$PYTHON_BIN" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo
    echo "NVIDIA GPU detected - installing the CUDA-enabled PyTorch build..."
    echo "This only needs to happen once and may take a few minutes."
    echo
    gpu_torch_ok=0
    for tag in cu129 cu128 cu126 cu124 cu121; do
      echo "Trying PyTorch build $tag..."
      "$PYTHON_BIN" -m pip install --force-reinstall torch --index-url "https://download.pytorch.org/whl/$tag" >/dev/null 2>&1 || true
      if "$PYTHON_BIN" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
        gpu_torch_ok=1
        break
      fi
    done
    if [ "$gpu_torch_ok" -eq 0 ]; then
      echo
      echo "Could not install a CUDA-enabled PyTorch build automatically."
      echo "PlasmidGPT will continue in CPU mode this run. See"
      echo "https://pytorch.org/get-started/locally/ to install one manually."
      echo
    fi
  fi
fi

if [ ! -f "pretrained_model/pretrained_model.pt" ]; then
  echo
  echo "The pretrained model file is missing: pretrained_model/pretrained_model.pt"
  echo
  echo "If you downloaded this project as a ZIP file, Git LFS assets are not"
  echo "included. Clone it with Git instead so large files download correctly:"
  echo
  echo "  git lfs install"
  echo "  git clone https://github.com/mkazbekov/PlasmidGPT-app.git"
  echo
  exit 1
fi

echo
echo "Starting PlasmidGPT Studio..."
echo "Your browser will open automatically."
echo "Keep this terminal window open while using PlasmidGPT."
echo

"$PYTHON_BIN" -m streamlit run app.py
