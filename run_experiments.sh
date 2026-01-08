#!/usr/bin/env bash
# ==========================================
# Run full CIFAR-100 experiment suite
# ==========================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "CIFAR-100 Contrastive / Semi-supervised Runs"
echo "=========================================="

# Check python
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found. Activate conda env first:"
  echo "  conda activate py3-torch"
  exit 1
fi

echo "Using python: $(which python)"
echo ""

python run_experiments.py

echo ""
echo "=========================================="
echo "ALL TRAINING PROCESSES FINISHED"
echo "=========================================="
