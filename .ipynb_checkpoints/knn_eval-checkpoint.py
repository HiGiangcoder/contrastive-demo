import os
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.backbone import ResNet18, ResNet34
from models.classifier import Model
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Full CIFAR-100 loader
# =====================
def get_cifar100_full(batch_size=256):
    transform = T.Compose([
        T.ToTensor(),
    ])

    train_set = CIFAR100("./data", train=True, download=True, transform=transform)
    test_set = CIFAR100("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

# =====================
# kNN evaluation
# =====================
def knn_eval(ckpt_path, k=20):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    backbone = ResNet18(
        pretrained=True,
        weight_path=f"models/resnet18.pth",
    )
    model = Model(backbone, cfg["emb_dim"], 100).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    train_loader, test_loader = get_cifar100_full()

    Z_train, Y_train = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            _, z = model(x)
            Z_train.append(z.cpu())
            Y_train.append(y)
    Z_train = torch.cat(Z_train).numpy()
    Y_train = torch.cat(Y_train).numpy()

    Z_test, Y_test = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            _, z = model(x)
            Z_test.append(z.cpu())
            Y_test.append(y)
    Z_test = torch.cat(Z_test).numpy()
    Y_test = torch.cat(Y_test).numpy()

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(Z_train, Y_train)
    return knn.score(Z_test, Y_test)

# =====================
# Run all
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--pattern", type=str, default=None,
                        help="optional substring to filter run paths")
    args = parser.parse_args()

    print("Running k-NN evaluation on best checkpoints under:", args.runs_dir)
    for root, _, files in os.walk(args.runs_dir):
        if "best.pth" in files:
            if args.pattern and args.pattern not in root:
                continue
            ckpt_path = os.path.join(root, "best.pth")
            try:
                acc = knn_eval(ckpt_path)
                print(f"[kNN] {root}: {acc:.4f}")
            except Exception as e:
                print(f"[kNN] Failed {root}: {e}")


if __name__ == "__main__":
    main()
