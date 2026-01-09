import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm import tqdm
from torchvision import transforms

# Imports từ project của bạn
from models.backbone import ResNet18, SmallCNN
from models.classifier import Model
from data.cifar100 import get_cifar100_semi
from data.stl10 import get_stl10_semi

# ==============================================================================
# CONFIG
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG_DIR = "figures/paper_style"
os.makedirs(FIG_DIR, exist_ok=True)

# Transform để tạo 2 view cho việc tính Alignment
# (Mô phỏng augmentation đơn giản để tạo positive pairs)
eval_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_features_pairs(model, loader, device):
    """
    Lấy feature của 2 view (x1, x2) từ cùng 1 ảnh gốc để tính Alignment.
    """
    model.eval()
    z1_list, z2_list, labels_list = [], [], []
    
    print("--> Extracting features (generating 2 views on-the-fly)...")
    with torch.no_grad():
        # Lấy data gốc từ loader (bỏ qua transform mặc định của loader nếu cần)
        # Ở đây ta giả định loader trả về ảnh gốc chưa augment mạnh hoặc ta augment lại
        for x, y in tqdm(loader, leave=False):
            # Với mỗi batch ảnh x, ta tự tạo 2 view x1, x2
            # Lưu ý: x ở đây thường đã là Tensor qua transform của loader. 
            # Để đơn giản, ta coi x là view 1, và tạo view 2 bằng cách nhiễu nhẹ hoặc lật.
            # Cách chuẩn nhất: Dùng raw PIL image, nhưng để tương thích code hiện tại:
            
            x = x.to(device)
            
            # Forward view 1
            _, z1 = model(x)
            
            # Tạo view 2: Flip ảnh x (Giả lập view khác)
            x2 = torch.flip(x, dims=[3]) 
            _, z2 = model(x2)

            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            
            z1_list.append(z1.cpu())
            z2_list.append(z2.cpu())
            labels_list.append(y.cpu())

    z1 = torch.cat(z1_list, dim=0).numpy()
    z2 = torch.cat(z2_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    return z1, z2, labels

def plot_alignment_histogram(z1, z2, ax, color='skyblue'):
    """
    Vẽ biểu đồ cột Alignment (Positive Pair Feature Distances)
    """
    # Tính L2 distance giữa các cặp positive
    diff = z1 - z2
    l2_distances = np.linalg.norm(diff, axis=1)
    
    sns.histplot(l2_distances, bins=30, kde=False, color=color, edgecolor='black', alpha=0.7, ax=ax)
    
    mean_dist = np.mean(l2_distances)
    ax.axvline(mean_dist, color='black', linestyle='--', label=f'Mean: {mean_dist:.2f}')
    
    ax.set_title("Alignment\nPositive Pair Distances", fontsize=10, fontweight='bold')
    ax.set_xlabel("$l_2$ Distances")
    ax.set_ylabel("Counts")
    ax.set_xlim(0, 2.0) # Khoảng cách tối đa trên hypersphere là 2
    ax.legend(fontsize=8)

def plot_uniformity_circle(z, labels, ax, title="Uniformity", subset_class=None):
    """
    Vẽ phân bố feature trên đường tròn (PCA -> 2D -> Normalize)
    """
    # Nếu chỉ vẽ class cụ thể
    if subset_class is not None:
        idx = np.where(labels == subset_class)[0]
        if len(idx) < 10: return # Bỏ qua nếu quá ít mẫu
        z = z[idx]
        title = f"Class {subset_class}"

    # 1. PCA giảm chiều về 2D (Nếu feature > 2D)
    if z.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        z_2d = pca.fit_transform(z)
    else:
        z_2d = z

    # 2. Chuẩn hóa về đường tròn đơn vị (S1)
    # Norm theo từng hàng
    norms = np.linalg.norm(z_2d, axis=1, keepdims=True)
    z_circle = z_2d / (norms + 1e-8) # Tránh chia cho 0

    x = z_circle[:, 0]
    y = z_circle[:, 1]

    # 3. Vẽ KDE Plot (Kernel Density Estimation) - Màu mè giống paper
    # cmap='Spectral_r' hoặc 'jet' tạo hiệu ứng nhiệt
    try:
        sns.kdeplot(x=x, y=y, fill=True, cmap="Spectral_r", thresh=0.05, levels=20, ax=ax, alpha=0.8)
    except:
        # Fallback nếu lỗi KDE (do ít điểm) thì vẽ scatter
        ax.scatter(x, y, alpha=0.5, s=5)

    # Vẽ đường tròn biên giới hạn
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--', alpha=0.3)
    ax.add_artist(circle)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Features ($S^1$)")

# ==============================================================================
# MAIN
# ==============================================================================
def find_best_checkpoints(root="runs"):
    ckpts = []
    for dirpath, _, filenames in os.walk(root):
        if "best.pth" in filenames:
            ckpts.append(os.path.join(dirpath, "best.pth"))
    return ckpts

def main():
    checkpoints = find_best_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints.")

    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            cfg = ckpt.get("config", {})
        except:
            continue
            
        method_name = cfg.get("name", "Unknown")
        dataset_name = cfg.get("dataset", "cifar100")
        
        print(f"\nProcessing: {method_name} | Dataset: {dataset_name}")
        
        # --- 1. Load Data ---
        if dataset_name == "stl10":
            _, _, loader = get_stl10_semi(batch_size=256, seed=42)
            num_classes = 10
            target_classes = [0, 2, 6, 9] # Chọn vài class để vẽ riêng (airplane, car, ...)
        else: # cifar100
            _, _, loader = get_cifar100_semi(batch_size=256, seed=42)
            num_classes = 100
            target_classes = [0, 25, 50, 75] # Chọn đại diện 4 class

        # --- 2. Build Model ---
        backbone_name = cfg.get("backbone", "resnet18")
        if backbone_name == "smallcnn":
            backbone = SmallCNN()
            feat_dim = 256
        else:
            backbone = ResNet18(pretrained=False)
            feat_dim = 512
            
        model = Model(backbone, feat_dim, cfg.get("emb_dim", 128), num_classes).to(DEVICE)
        
        if "student" in ckpt:
            model.load_state_dict(ckpt["student"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            continue
            
        # --- 3. Get Features ---
        z1, z2, labels = get_features_pairs(model, loader, DEVICE)
        
        # --- 4. Plotting Setup ---
        # Layout: 1 hàng, 6 cột (Alignment | Uniformity All | Class 1 | Class 2 | Class 3 | Class 4)
        fig = plt.figure(figsize=(18, 3.5), constrained_layout=True)
        spec = fig.add_gridspec(1, 6) # 1 row, 6 cols

        # Plot 1: Alignment Histogram
        ax_align = fig.add_subplot(spec[0, 0])
        plot_alignment_histogram(z1, z2, ax_align)

        # Plot 2: Uniformity (All classes)
        ax_unif = fig.add_subplot(spec[0, 1])
        plot_uniformity_circle(z1, labels, ax_unif, title="Feature Dist.\n(All Classes)")

        # Plot 3-6: Specific Classes
        for i, cls_idx in enumerate(target_classes):
            if i >= 4: break
            ax_cls = fig.add_subplot(spec[0, 2 + i])
            # Mapping class index to name (optional)
            cls_name = f"Class {cls_idx}"
            plot_uniformity_circle(z1, labels, ax_cls, title=cls_name, subset_class=cls_idx)

        # Save
        filename = f"{dataset_name}_{method_name}_analysis.png"
        save_path = os.path.join(FIG_DIR, filename)
        
        # Thêm super title
        acc = ckpt.get('acc', 0.0)
        if isinstance(acc, dict): acc = 0 # Handle legacy format
        plt.suptitle(f"Method: {method_name} | Dataset: {dataset_name}", fontsize=14)
        
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"--> Saved to {save_path}")

if __name__ == "__main__":
    main()