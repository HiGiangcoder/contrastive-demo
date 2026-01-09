import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# Imports từ project của bạn
from data.cifar100 import get_cifar100_semi
from data.stl10 import get_stl10_semi
from models.backbone import ResNet18, SmallCNN
from models.classifier import Model

device = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "figures/tsne"
os.makedirs(OUT_DIR, exist_ok=True)

def plot_tsne(Z, Y, title, save_path, num_classes):
    """
    Hàm vẽ t-SNE chung.
    """
    print(f"--> Computing t-SNE for {title}...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        init="pca",
        random_state=42,
        n_jobs=-1 # Sử dụng đa luồng để nhanh hơn
    )
    Z2 = tsne.fit_transform(Z)

    plt.figure(figsize=(10, 10))
    # Sử dụng colormap 'tab20' nếu lớp <= 20, nếu nhiều hơn dùng 'nipy_spectral'
    cmap_name = "tab20" if num_classes <= 20 else "nipy_spectral"
    
    scatter = plt.scatter(Z2[:, 0], Z2[:, 1], c=Y, s=10, cmap=cmap_name, alpha=0.7)
    
    # Chỉ hiện legend nếu số lớp ít để đỡ rối
    if num_classes <= 20:
        plt.legend(*scatter.legend_elements(), title="Classes", loc="best", bbox_to_anchor=(1, 1))
        
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"--> Saved to {save_path}")

def find_best_checkpoints(root="runs"):
    ckpts = []
    for dirpath, _, filenames in os.walk(root):
        if "best.pth" in filenames:
            ckpts.append(os.path.join(dirpath, "best.pth"))
    return ckpts

def main():
    best_ckpts = find_best_checkpoints()
    print(f"Found {len(best_ckpts)} best checkpoints")

    for ckpt_path in best_ckpts:
        print(f"\n[Processing] {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            print(f"Cannot load {ckpt_path}: {e}")
            continue

        cfg = ckpt.get("config", None)
        if cfg is None:
            print("Config not found in checkpoint, skipping...")
            continue

        method = cfg.get("name", "unknown")
        dataset_name = cfg.get("dataset", "cifar100")
        backbone_name = cfg.get("backbone", "resnet18")
        
        print(f"Setup: {method} | Dataset: {dataset_name} | Backbone: {backbone_name}")

        # ==========================
        # 1. Setup Data & Model Params
        # ==========================
        if dataset_name == "stl10":
            # Lấy test loader (index 2)
            _, _, loader = get_stl10_semi(batch_size=256, seed=42, num_workers=4)
            num_classes = 10
        else: # cifar100
            _, _, loader = get_cifar100_semi(batch_size=256, seed=42, num_workers=4)
            num_classes = 100

        # ==========================
        # 2. Setup Backbone
        # ==========================
        if backbone_name == "smallcnn":
            backbone = SmallCNN()
            feat_dim = 256
        else:
            backbone = ResNet18(pretrained=False)
            feat_dim = 512

        # ==========================
        # 3. Setup Model
        # ==========================
        # SỬA LỖI: Truyền đủ 4 tham số: backbone, feat_dim, emb_dim, num_classes
        model = Model(
            backbone,
            feat_dim,
            cfg.get("emb_dim", 128),
            num_classes
        ).to(device)

        # Load weights (ưu tiên 'student', fallback 'model')
        if "student" in ckpt:
            model.load_state_dict(ckpt["student"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            print("Weight key not found (neither 'student' nor 'model'). Skipping.")
            continue
            
        model.eval()

        # ==========================
        # 4. Extract Features
        # ==========================
        Z, Y = [], []
        print("Extracting features...")
        with torch.no_grad():
            for x, y in tqdm(loader, leave=False):
                x = x.to(device)
                _, z = model(x) # Model trả về logits, features
                Z.append(z.cpu())
                Y.append(y)

        Z = torch.cat(Z).numpy()
        Y = torch.cat(Y).numpy()

        # ==========================
        # 5. Plotting Logic
        # ==========================
        exp_dir = os.path.join(OUT_DIR, f"{method}_{dataset_name}_{backbone_name}")
        os.makedirs(exp_dir, exist_ok=True)

        if dataset_name == "stl10":
            # STL-10 chỉ có 10 class, vẽ 1 hình duy nhất là đủ
            plot_tsne(
                Z, Y,
                f"{method} (STL-10) | All Classes",
                os.path.join(exp_dir, "tsne_all.png"),
                num_classes
            )
        else:
            # CIFAR-100: Vẽ tổng thể và chia nhỏ
            # 1. Vẽ tất cả 100 class (sẽ rất dày đặc)
            plot_tsne(
                Z, Y,
                f"{method} (CIFAR-100) | All 100 Classes",
                os.path.join(exp_dir, "tsne_100cls.png"),
                num_classes
            )

            # 2. Vẽ từng nhóm 20 class để dễ nhìn
            print("Splitting into subsets for clearer visualization...")
            for i in range(5):
                cls_start = i * 20
                cls_end = (i + 1) * 20
                cls_indices = np.arange(cls_start, cls_end)
                
                # Lọc dữ liệu thuộc các class này
                mask = np.isin(Y, cls_indices)
                if mask.sum() == 0: continue
                
                plot_tsne(
                    Z[mask],
                    Y[mask],
                    f"{method} | Classes {cls_start}-{cls_end-1}",
                    os.path.join(exp_dir, f"tsne_20cls_part_{i}.png"),
                    20 # Pass 20 để dùng colormap tab20
                )

if __name__ == "__main__":
    main()