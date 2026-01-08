import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data.cifar100 import get_cifar100
from models.backbone import ResNet18
from models.classifier import Model

device = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "figures/tsne"
os.makedirs(OUT_DIR, exist_ok=True)

# Load full test set
_, loader = get_cifar100(batch_size=512, contrastive=False)

def plot_tsne(Z, Y, title, save_path):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        init="pca",
        random_state=42,
    )
    Z2 = tsne.fit_transform(Z)

    plt.figure(figsize=(6, 6))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=Y, s=6, cmap="tab20")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def find_best_checkpoints(root="runs"):
    ckpts = []
    for dirpath, _, filenames in os.walk(root):
        if "best.pth" in filenames:
            ckpts.append(os.path.join(dirpath, "best.pth"))
    return ckpts

best_ckpts = find_best_checkpoints()
print(f"Found {len(best_ckpts)} best checkpoints")

for ckpt_path in best_ckpts:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    method = cfg["name"]
    bs = cfg["batch_size"]

    print(f"t-SNE for {method} | bs={bs}")

    model = Model(
        ResNet18(),
        cfg["emb_dim"],
        100
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    Z, Y = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, z = model(x)
            Z.append(z.cpu())
            Y.append(y)

    Z = torch.cat(Z).numpy()
    Y = torch.cat(Y).numpy()

    exp_dir = os.path.join(OUT_DIR, method)
    os.makedirs(exp_dir, exist_ok=True)

    # 5 plots: 20 classes
    for i in range(5):
        cls = np.arange(i * 20, (i + 1) * 20)
        mask = np.isin(Y, cls)
        plot_tsne(
            Z[mask],
            Y[mask],
            f"{method} | classes {i*20}-{(i+1)*20-1}",
            os.path.join(exp_dir, f"tsne_20cls_{i}.png")
        )

    # all 100 classes
    plot_tsne(
        Z, Y,
        f"{method} | all classes",
        os.path.join(exp_dir, "tsne_100cls.png")
    )
