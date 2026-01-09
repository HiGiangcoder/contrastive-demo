import os
import torch
from tqdm import tqdm

from models.backbone import ResNet18
from models.classifier import Model

from data.cifar100 import get_cifar100
from data.stl10 import get_stl10_eval

# ======================================================
# Device
# ======================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Evaluating all experiments")
print("=" * 60)

# ======================================================
# Walk through all runs/*/*/bs_*/checkpoints/best.pth
# ======================================================
for root, _, files in os.walk("runs"):
    if "best.pth" not in files:
        continue

    ckpt_path = os.path.join(root, "best.pth")
    print(f"\n[Evaluating] {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    # ==================================================
    # Dataset switch
    # ==================================================
    dataset = cfg.get("dataset", "cifar100")

    if dataset == "cifar100":
        _, test_loader = get_cifar100(
            batch_size=256,
            contrastive=False
        )
        num_classes = 100

    elif dataset == "stl10":
        test_loader = get_stl10_eval(
            batch_size=256,
        )
        num_classes = 10

    else:
        print(f"[SKIP] Unknown dataset: {dataset}")
        continue

    # ==================================================
    # Build model (EXACTLY like training)
    # ==================================================
    assert cfg["backbone"] == "resnet18", "Only ResNet18 is supported"

    backbone = ResNet18(
        pretrained=True,
        weight_path="models/resnet18.pth",
    )

    model = Model(
        backbone=backbone,
        emb_dim=cfg["emb_dim"],
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # ==================================================
    # Evaluation
    # ==================================================
    correct = total = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(
        f"Dataset={dataset} | "
        f"Method={cfg['name']} | "
        f"Acc={acc:.4f}"
    )

print("\nEvaluation finished.")
