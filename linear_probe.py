import os
import torch
import torch.nn as nn
from tqdm import tqdm

from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.backbone import ResNet18, ResNet34
from models.classifier import Model
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Full supervised CIFAR-100 loader (for probe)
# =====================
def get_cifar100_full(batch_size=256):
    transform = T.Compose([
        T.ToTensor(),
    ])

    train_set = CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = CIFAR100(
        "./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_loader, test_loader

# =====================
# Linear probe
# =====================
def linear_probe(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    # ---- build encoder EXACTLY like training ----
    backbone = ResNet18(
        pretrained=True,
        weight_path=f"models/resnet18.pth",
    )

    model = Model(backbone, cfg["emb_dim"], 100).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    classifier = nn.Linear(cfg["emb_dim"], 100).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    train_loader, test_loader = get_cifar100_full()

    # ---- train linear head ----
    for epoch in range(10):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                _, z = model(x)
            logits = classifier(z)
            loss = nn.functional.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ---- eval ----
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, z = model(x)
            pred = classifier(z).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

# =====================
# Run for all experiments
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--pattern", type=str, default=None,
                        help="optional substring to filter run paths")
    args = parser.parse_args()

    print("Running linear probe on best checkpoints under:", args.runs_dir)
    for root, _, files in os.walk(args.runs_dir):
        if "best.pth" in files:
            if args.pattern and args.pattern not in root:
                continue
            ckpt_path = os.path.join(root, "best.pth")
            try:
                acc = linear_probe(ckpt_path)
                print(f"[Linear Probe] {root}: {acc:.4f}")
            except Exception as e:
                print(f"[Linear Probe] Failed {root}: {e}")


if __name__ == "__main__":
    main()
