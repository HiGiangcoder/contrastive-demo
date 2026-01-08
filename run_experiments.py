#!/usr/bin/env python3
"""
Run all CIFAR-100 contrastive / semi-supervised experiments sequentially.

Usage:
  conda activate py3-torch
  python run_experiments.py
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
TRAIN_SCRIPT = ROOT / "train.py"

# ======================================================
# Explicit list of experiments (ONLY what you have)
# ======================================================
CONFIG_FILES = [
    # ---------- supervised ----------
    "supcon.yaml",
    "contrastive.yaml",
    "triplet.yaml",
    "info_nce.yaml",
    "align_uniform.yaml",

    # ---------- semi-supervised ----------
    "supcon_semi.yaml",
    "contrastive_semi.yaml",
    "triplet_semi.yaml",
    "info_nce_semi.yaml",
    "info_nce_semi_u2pl.yaml",
    "align_uniform_semi.yaml",
]

def run_train(cfg_path: Path):
    cmd = ["python", str(TRAIN_SCRIPT), "--config", str(cfg_path)]
    print("=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)

    # run synchronously (SAFE)
    subprocess.run(cmd, check=True)


def main():
    print("Starting CIFAR-100 experiment suite")
    print("===================================")

    for cfg_name in CONFIG_FILES:
        cfg_path = CONFIG_DIR / cfg_name
        if not cfg_path.exists():
            print(f"[SKIP] Missing config: {cfg_name}")
            continue

        run_train(cfg_path)

    print("\nALL EXPERIMENTS FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
