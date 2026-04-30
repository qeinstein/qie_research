#!/bin/bash
# RunPod environment bootstrap for QIE research sweep.
# Run this once after SSHing into the pod, from inside the repo directory.
# Usage: bash setup_runpod.sh
set -euo pipefail

echo "=== [1/3] Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements-runpod.txt

echo "=== [2/3] Installing qie_research package ===
pip install -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
echo \"export PYTHONPATH=\$PYTHONPATH:$(pwd)/src\" >> ~/.bashrc


echo "=== [3/3] Verifying environment ==="
python3 -c "
try:
    import torch
    print(f'  torch version : {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('  torch not installed (expected on pure CPU pod)')

try:
    import qie_research
    print('  qie_research package: OK')
except ImportError:
    print('  ERROR: qie_research package not found in path')
    exit(1)
"

mkdir -p data/raw data/processed results/metrics

echo ""
echo "=== Setup complete. ==="
echo "Run the sweep with:"
echo "  ./run_full_sweep.sh --skip-install"
