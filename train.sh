#!/bin/bash

# ======================================================
# STL-10 Contrastive / Semi-Supervised Training Script
# (ResNet-18 + Projection Head)
# ======================================================
export CUDA_VISIBLE_DEVICES=1

set -e

echo "======================================================"
echo "START TRAINING ON STL-10"
echo "======================================================"

# ------------------------------------------------------
# 1. Baseline
# ------------------------------------------------------
echo "[1/5] Training: Baseline (CE)"
python train.py --config config/baseline_stl10.yaml
echo "------------------------------------------------------"

# ------------------------------------------------------
# 2. Contrastive Loss (Hadsell et al. 2006)
# ------------------------------------------------------
echo "[2/5] Training: Contrastive Loss (2006)"
python train.py --config config/contrastive_2006_stl10.yaml
echo "------------------------------------------------------"

# ------------------------------------------------------
# 3. Triplet Loss
# ------------------------------------------------------
echo "[3/5] Training: Triplet Loss"
python train.py --config config/triplet_stl10.yaml
echo "------------------------------------------------------"

# ------------------------------------------------------
# 4. InfoNCE
# ------------------------------------------------------
echo "[4/5] Training: InfoNCE Loss"
python train.py --config config/info_nce_stl10.yaml
echo "------------------------------------------------------"

# ------------------------------------------------------
# 5. Align + Uniform
# ------------------------------------------------------
echo "[5/5] Training: Align + Uniform Loss"
python train.py --config config/align_uniform_stl10.yaml
echo "------------------------------------------------------"


echo ""
echo "======================================================"
echo "ALL STL-10 TRAINING PROCESSES FINISHED"
echo "======================================================"
