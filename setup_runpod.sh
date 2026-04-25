#!/bin/bash
# RunPod environment bootstrap for QIE research sweep.
# Run this once after SSHing into the pod, from inside the repo directory.
# Usage: bash setup_runpod.sh
set -euo pipefail

echo "=== [1/3] Installing Python dependencies ==="
pip install -r requirements-runpod.txt

echo "=== [2/3] Installing qie_research package ==="
pip install -e .

echo "=== [3/3] Verifying torch GPU access ==="
python3 -c "
import torch
print(f'  torch version : {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU           : {torch.cuda.get_device_name(0)}')
else:
    print('  WARNING: CUDA not available — torch paths will run on CPU')
"

mkdir -p data/raw data/processed results/metrics

echo ""
echo "=== Setup complete. ==="
echo "Run the sweep with:"
echo "  ./run_full_sweep.sh --skip-install"
