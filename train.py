import argparse
import yaml
import torch
import os
import csv
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
from shutil import copyfile

# =========================
# Data
# =========================
from data.cifar100 import get_cifar100, get_cifar100_semi
from data.stl10 import get_stl10_semi

# =========================
# Model & Loss
# =========================
from models.backbone import ResNet18
from models.classifier import Model
from losses.contrastive_2006 import ContrastiveLoss2006
from losses.triplet import TripletLoss
from losses.info_nce import InfoNCELoss
from losses.align_uniform import AlignUniformLoss

# ======================================================
# Args & Config
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config))

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Experiment identity
# ======================================================
method = cfg["name"]
dataset = cfg.get("dataset", "cifar100")
batch_size = cfg["batch_size"]
label_ratio = cfg.get("label_ratio", 1.0)
loss_type = cfg["loss_type"]              # <<< QUAN TRỌNG

run_dir = f"runs/{method}/resnet18/bs_{batch_size}"
ckpt_dir = f"{run_dir}/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

latest_path = f"{ckpt_dir}/latest.pth"
best_path = f"{ckpt_dir}/best.pth"

copyfile(args.config, f"{run_dir}/config.yaml")

# ======================================================
# Logging
# ======================================================
log_txt = open(f"{run_dir}/train.log", "a")
log_csv = open(f"{run_dir}/metrics.csv", "a", newline="")
csv_writer = csv.writer(log_csv)

if log_csv.tell() == 0:
    csv_writer.writerow([
        "epoch",
        "loss",
        "ce_loss",
        "contra_loss",
        "val_acc",
        "alignment",
        "uniformity",
        "label_ratio",
        "num_labeled",
        "num_unlabeled",
    ])

def log(msg):
    print(msg)
    log_txt.write(msg + "\n")
    log_txt.flush()

# ======================================================
# Embedding metrics
# ======================================================
def alignment(z, labels):
    mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    dist = torch.cdist(z, z)
    return dist[mask].pow(2).mean()

def uniformity(z):
    dist = torch.cdist(z, z)
    return torch.log(torch.exp(-2 * dist.pow(2)).mean())


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total

# ======================================================
# Data loading
# ======================================================
if dataset == "cifar100":
    if label_ratio >= 1.0:
        train_loader, test_loader = get_cifar100(batch_size, contrastive=True)
        labeled_loader = train_loader
        unlabeled_loader = None
    else:
        labeled_loader, unlabeled_loader, test_loader = get_cifar100_semi(
            batch_size=batch_size,
            label_ratio=label_ratio,
            contrastive=True,
            seed=cfg.get("seed", 42),
        )
elif dataset == "stl10":
    labeled_loader, unlabeled_loader, test_loader = get_stl10_semi(
        batch_size=batch_size,
        seed=cfg.get("seed", 42),
        num_workers=cfg.get("num_workers", 0)
    )
else:
    raise ValueError(f"Unknown dataset: {dataset}")

num_labeled = len(labeled_loader.dataset)
num_unlabeled = 0 if unlabeled_loader is None else len(unlabeled_loader.dataset)

log(f"Dataset={dataset} | loss_type={loss_type}")
log(f"Labeled={num_labeled}, Unlabeled={num_unlabeled}")

# ======================================================
# Model (ResNet18 + projection head)
# ======================================================
backbone = ResNet18(pretrained=True, weight_path="models/resnet18.pth")
model = Model(
    backbone=backbone,
    emb_dim=cfg["emb_dim"],
    num_classes=100 if dataset == "cifar100" else 10,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

# ======================================================
# Loss functions
# ======================================================
loss_fns = {
    "contrastive_2006": ContrastiveLoss2006(),
    "triplet": TripletLoss(),
    "info_nce": InfoNCELoss(),
    "align_uniform": AlignUniformLoss(),
}

contra_loss_fn = loss_fns.get(loss_type, None)

# ======================================================
# Resume & Early stopping
# ======================================================
start_epoch = 0
best_acc = 0.0
epochs_no_improve = 0

early_cfg = cfg.get("early_stopping", {})
use_early_stop = early_cfg.get("enable", False)
patience = early_cfg.get("patience", 10)
min_delta = early_cfg.get("min_delta", 0.0)

if os.path.exists(latest_path):
    ckpt = torch.load(latest_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt["best_acc"]
    epochs_no_improve = ckpt.get("epochs_no_improve", 0)
    log(f"[RESUME] epoch={start_epoch}, best_acc={best_acc:.4f}")
else:
    log("[START] training from scratch")

# ======================================================
# Training loop
# ======================================================
unlabeled_iter = cycle(unlabeled_loader) if unlabeled_loader is not None else None

for epoch in range(start_epoch, cfg["epochs"]):
    model.train()

    sum_loss = sum_ce = sum_contra = 0.0
    sum_align = sum_unif = 0.0
    count = 0

    for x_l, y_l in tqdm(labeled_loader, desc=f"[{method}] Epoch {epoch}"):
        x_l, y_l = x_l.to(device), y_l.to(device)
        logits, z_l = model(x_l)

        ce_loss = F.cross_entropy(logits, y_l)

        # ---------- build embedding batch ----------
        if unlabeled_iter is not None and loss_type != "sup_only":
            x_u, _ = next(unlabeled_iter)
            x_u = x_u.to(device)
            _, z_u = model(x_u)

            z_all = torch.cat([z_l, z_u], dim=0)
            fake_labels = torch.arange(z_u.size(0), device=device) + y_l.max() + 1
            all_labels = torch.cat([y_l, fake_labels])
        else:
            z_all = z_l
            all_labels = y_l

        # ---------- contrastive ----------
        if loss_type == "sup_only":
            contra_loss = torch.zeros_like(ce_loss)
        elif loss_type == "align_uniform":
            contra_loss = alignment(z_all, all_labels) + uniformity(z_all)
        else:
            contra_loss = contra_loss_fn(z_all, all_labels)

        loss = ce_loss + cfg["lambda_c"] * contra_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        sum_ce += ce_loss.item()
        sum_contra += contra_loss.item()
        sum_align += alignment(z_all, all_labels).item()
        sum_unif += uniformity(z_all).item()
        count += 1

    avg_loss = sum_loss / count
    avg_ce = sum_ce / count
    avg_contra = sum_contra / count
    avg_align = sum_align / count
    avg_unif = sum_unif / count

    val_acc = evaluate(model, test_loader)

    improved = val_acc > best_acc + min_delta
    if improved:
        best_acc = val_acc
        epochs_no_improve = 0
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": cfg,
            },
            best_path,
        )
        log(f"[BEST] Epoch {epoch} | Acc {val_acc:.4f}")
    else:
        epochs_no_improve += 1

    csv_writer.writerow([
        epoch,
        avg_loss,
        avg_ce,
        avg_contra,
        val_acc,
        avg_align,
        avg_unif,
        label_ratio,
        num_labeled,
        num_unlabeled,
    ])
    log_csv.flush()

    log(
        f"Epoch {epoch} | "
        f"Loss {avg_loss:.4f} | "
        f"CE {avg_ce:.4f} | "
        f"Contra {avg_contra:.4f} | "
        f"Acc {val_acc:.4f}"
    )

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "epochs_no_improve": epochs_no_improve,
            "config": cfg,
        },
        latest_path,
    )

    if use_early_stop and epochs_no_improve >= patience:
        log(f"[EARLY STOP] No improvement for {epochs_no_improve} epochs")
        break

log_txt.close()
log_csv.close()
