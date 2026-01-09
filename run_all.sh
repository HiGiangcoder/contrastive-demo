#!/bin/bash

# ======================================================
# Full STL-10 Semi-Supervised Contrastive Pipeline
# ======================================================

set -e

echo "======================================================"
echo "START FULL SEMI-SUPERVISED PIPELINE (STL-10)"
echo "======================================================"

# ------------------------------------------------------
# Step 1: Training (resume + early stopping supported)
# ------------------------------------------------------
echo ""
echo "[STEP 1] Training all experiments"
echo "------------------------------------------------------"

# Chạy script train.sh (đảm bảo file này gọi python train.py cho từng config)
bash train.sh

echo ""
echo "[STEP 1] Training finished."
echo ""

# ------------------------------------------------------
# Step 2: Evaluation (best checkpoint)
# ------------------------------------------------------
echo "[STEP 2] Evaluating best checkpoints"
echo "------------------------------------------------------"

# Chạy eval.sh nếu bạn có script này để tính metrics chi tiết
if [ -f "eval.sh" ]; then
    bash eval.sh
else
    # Fallback: Chạy evaluate.py trực tiếp nếu không có eval.sh
    python evaluate.py
fi

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
# Step 4: Paper-style Visualization (Align & Uniformity)
# ------------------------------------------------------
echo "[STEP 4] Generating paper-style geometric analysis"
echo "------------------------------------------------------"

# Script vẽ biểu đồ histogram alignment và vòng tròn uniformity
python visualize_paper.py

echo ""
echo "[STEP 4] Paper-style visualization finished."
echo ""

# ------------------------------------------------------
# Step 5: Comparison plots (Step-by-step for slides)
# ------------------------------------------------------
echo "[STEP 5] Generating comparison plots for slides"
echo "------------------------------------------------------"

# Script tổng hợp số liệu và vẽ biểu đồ so sánh theo từng giai đoạn
python comparation.py

echo ""
echo "[STEP 5] Comparison plots finished."
echo ""

echo "======================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "Check output in ./figures/"
echo "======================================================"