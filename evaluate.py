import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

# =========================
# Imports (Match train.py)
# =========================
from models.backbone import ResNet18, SmallCNN
from models.classifier import Model
from data.cifar100 import get_cifar100_semi
from data.stl10 import get_stl10_semi

# ======================================================
# Device
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Evaluating all experiments")
print("=" * 60)

# ======================================================
# Walk through all runs/*/*/checkpoints/best.pth
# ======================================================
for root, _, files in os.walk("runs"):
    if "best.pth" not in files:
        continue

    ckpt_path = os.path.join(root, "best.pth")
    print(f"\n[Evaluating] {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"[ERROR] Could not load checkpoint: {e}")
        continue
        
    cfg = ckpt.get("config", None)
    if cfg is None:
        print("[SKIP] Config not found in checkpoint")
        continue

    # ==================================================
    # Dataset switch
    # ==================================================
    dataset_name = cfg.get("dataset", "cifar100")
    class_names = None

    if dataset_name == "stl10":
        # Dùng get_stl10_semi lấy test loader (phần tử thứ 3) cho an toàn
        _, _, test_loader = get_stl10_semi(
            batch_size=256, 
            seed=cfg.get("seed", 42), 
            num_workers=4
        )
        # Tên lớp STL-10
        if hasattr(test_loader.dataset, 'classes'):
            class_names = test_loader.dataset.classes
        else:
            class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        num_classes = 10

    elif dataset_name == "cifar100":
        _, _, test_loader = get_cifar100_semi(
            batch_size=256,
            seed=cfg.get("seed", 42),
            num_workers=4
        )
        num_classes = 100
    else:
        print(f"[SKIP] Unknown dataset: {dataset_name}")
        continue

    # ==================================================
    # Build model (Dynamic Backbone)
    # ==================================================
    backbone_name = cfg.get("backbone", "resnet18")
    
    if backbone_name == "smallcnn":
        backbone = SmallCNN()
        feat_dim = 256
    else:
        # Mặc định ResNet18
        backbone = ResNet18(pretrained=False) # Không cần load pretrain lại, vì sẽ load từ ckpt
        feat_dim = 512

    model = Model(
        backbone=backbone,
        feat_dim=feat_dim,
        emb_dim=cfg.get("emb_dim", 128),
        num_classes=num_classes,
    ).to(device)

    # Load Weights (Load Student)
    if "student" in ckpt:
        model.load_state_dict(ckpt["student"])
    elif "model" in ckpt: # Fallback cho code cũ
        model.load_state_dict(ckpt["model"])
    else:
        print("[ERROR] No model weights found in checkpoint")
        continue
        
    model.eval()

    # ==================================================
    # Evaluation Logic
    # ==================================================
    all_preds = []
    all_targets = []
    
    print(f"--> Running inference on {dataset_name} ({backbone_name})...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    # Tính Accuracy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = (all_preds == all_targets).mean()

    print(f"--> Method: {cfg.get('name', 'Unknown')}")
    print(f"--> Accuracy: {acc:.4f}")
    
    # Detailed Report
    print("-" * 30)
    print("Detailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
    print("=" * 60)

print("\nAll evaluations finished.")