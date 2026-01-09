#!/bin/bash

# ======================================================
# Full CIFAR-100 Semi-Supervised Contrastive Pipeline
# ======================================================

set -e

echo "======================================================"
echo "START FULL SEMI-SUPERVISED PIPELINE (CIFAR-100)"
echo "======================================================"

# ------------------------------------------------------
# Step 1: Training (resume + early stopping supported)
# ------------------------------------------------------
echo ""
echo "[STEP 1] Training all experiments"
echo "------------------------------------------------------"

# bash train.sh

echo ""
echo "[STEP 1] Training finished."
echo ""

# ------------------------------------------------------
# Step 2: Evaluation (best checkpoint)
# ------------------------------------------------------
echo "[STEP 2] Evaluating best checkpoints"
echo "------------------------------------------------------"

bash eval.sh

echo ""
echo "[STEP 2] Evaluation finished."
echo ""

# ------------------------------------------------------
# Step 3: t-SNE Visualization (representation quality)
# ------------------------------------------------------
echo "[STEP 3] Generating t-SNE figures"
echo "------------------------------------------------------"

python tsne.py

echo ""
echo "[STEP 3] t-SNE generation finished."
echo ""

# ------------------------------------------------------
# Step 4: Comparison plots (accuracy / geometry)
# ------------------------------------------------------
echo "[STEP 4] Generating comparison plots"
echo "------------------------------------------------------"

python comparation.py

echo ""
echo "[STEP 4] Comparison plots finished."
echo ""

echo "======================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "======================================================"
