import argparse
import yaml
import torch
import os
import csv
import copy
import math
import logging
import numpy as np # Cần thêm numpy
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
from shutil import copyfile
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report # [NEW] Thư viện tính chỉ số chi tiết

# =========================
# Imports
# =========================
from data.cifar100 import get_cifar100_semi
from data.stl10 import get_stl10_semi
from models.backbone import ResNet18, SmallCNN
from models.classifier import Model
from losses.contrastive_2006 import ContrastiveLoss2006
from losses.triplet import TripletLoss
from losses.info_nce import InfoNCELoss
from losses.align_uniform import AlignUniformLoss

# ======================================================
# Utils
# ======================================================
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name; self.fmt = fmt; self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def get_logger(log_path):
    logger = logging.getLogger()
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        fh = logging.FileHandler(log_path); fh.setFormatter(formatter); logger.addHandler(fh)
        sh = logging.StreamHandler(); sh.setFormatter(formatter); logger.addHandler(sh)
    return logger

class EarlyStopping:
    def __init__(self, patience=15, delta=0):
        self.patience = patience; self.counter = 0; self.best_score = None; self.early_stop = False; self.delta = delta
    def __call__(self, score):
        if self.best_score is None: self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else: self.best_score = score; self.counter = 0

@torch.no_grad()
def update_ema(student, teacher, alpha):
    # 1. Weights: EMA update
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(alpha).add_(ps.data, alpha=1 - alpha)
    # 2. Buffers (Batch Norm): Copy trực tiếp (Sync stats)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)

def adjust_learning_rate(optimizer, epoch, cfg):
    """Cosine Annealing LR"""
    lr_max = cfg['lr']
    lr_min = cfg.get('min_lr', 1e-5)
    epochs = cfg['epochs']
    warmup = cfg.get('warmup_epochs', 0)
    
    if epoch < warmup:
        lr = lr_max * (epoch + 1) / (warmup + 1)
    else:
        curr = epoch - warmup
        tot = epochs - warmup
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * curr / tot))
    
    for pg in optimizer.param_groups: pg['lr'] = lr
    return lr

def get_ema_decay(epoch, total_epochs, base_ema=0.996):
    """Dynamic EMA Decay"""
    return 1 - (1 - base_ema) * (math.cos(math.pi * epoch / total_epochs) + 1) / 2

# ======================================================
# Train Function
# ======================================================
def train_one_epoch(epoch, student, teacher, labeled_loader, unlabeled_loader, 
                    optimizer, scaler, cfg, device, loss_type, contra_loss_fn, use_contrastive, current_ema):
    student.train()
    teacher.eval() 

    losses = AverageMeter('Loss', ':.4f')
    losses_sup = AverageMeter('Sup', ':.4f')
    losses_uns = AverageMeter('Unsup', ':.4f')
    losses_con = AverageMeter('Con', ':.4f')
    mask_ratios = AverageMeter('Mask', ':.2f')

    unlabeled_iter = cycle(unlabeled_loader) if unlabeled_loader else None
    
    base_threshold = cfg.get("pseudo_threshold", 0.95)
    
    pbar = tqdm(labeled_loader, desc=f"Epoch {epoch:03d}", leave=False)
    
    for x_l, y_l in pbar:
        x_l, y_l = x_l.to(device), y_l.to(device)
        
        x_u = None
        if unlabeled_iter:
            try: u_batch = next(unlabeled_iter); x_u = u_batch[0].to(device)
            except: unlabeled_iter = cycle(unlabeled_loader); u_batch = next(unlabeled_iter); x_u = u_batch[0].to(device)

        with autocast():
            # 1. Supervised
            logits_l, z_l = student(x_l)
            sup_loss = F.cross_entropy(logits_l, y_l)

            # 2. Unsupervised
            unsup_loss = torch.tensor(0.0, device=device)
            contra_loss = torch.tensor(0.0, device=device)
            mask_count = 0.0

            if x_u is not None:
                with torch.no_grad():
                    logits_u_t, _ = teacher(x_u)
                    probs = torch.softmax(logits_u_t, dim=1)
                    max_probs, pseudo = probs.max(dim=1)
                    
                    curr_threshold = base_threshold
                    if epoch < 5: curr_threshold = min(0.8, base_threshold)
                    
                    mask = max_probs.ge(curr_threshold).float()
                    mask_count = mask.mean().item()

                logits_u_s, z_u_s = student(x_u)
                
                if mask.sum() > 0:
                    unsup_loss = (F.cross_entropy(logits_u_s, pseudo, reduction='none') * mask).mean()
                    
                    if use_contrastive:
                        mask_bool = mask.bool()
                        z_u_strong = z_u_s[mask_bool]
                        pseudo_strong = pseudo[mask_bool]
                        
                        if z_u_strong.size(0) > 0:
                            z_all = torch.cat([z_l, z_u_strong], dim=0)
                            labels_all = torch.cat([y_l, pseudo_strong], dim=0)
                            
                            if loss_type == "align_uniform":
                                z_all = F.normalize(z_all, dim=1)
                                dist = torch.cdist(z_all, z_all, p=2)
                                same = labels_all.unsqueeze(1) == labels_all.unsqueeze(0)
                                align = dist[same].pow(2).mean()
                                unif = torch.log(torch.exp(-2 * dist.pow(2)).mean())
                                contra_loss = align + unif
                            elif contra_loss_fn:
                                contra_loss = contra_loss_fn(z_all, labels_all)

            loss = sup_loss + (cfg.get("lambda_u", 1.0) * unsup_loss) + (cfg.get("lambda_c", 1.0) * contra_loss)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        update_ema(student, teacher, current_ema)

        losses.update(loss.item())
        losses_sup.update(sup_loss.item())
        losses_uns.update(unsup_loss.item())
        losses_con.update(contra_loss.item())
        mask_ratios.update(mask_count)
        
        pbar.set_postfix({'L': f"{losses.avg:.3f}", 'Msk': f"{mask_ratios.avg:.2f}"})

    return {k: v.avg for k, v in [('loss', losses), ('sup', losses_sup), ('unsup', losses_uns), ('contra', losses_con), ('mask', mask_ratios)]}

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for data in loader:
        x = data[0].to(device); y = data[1].to(device)
        logits, _ = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item(); total += y.size(0)
    return correct / total

# [NEW] Hàm Evaluate chi tiết
@torch.no_grad()
def detailed_evaluate(model, loader, device, class_names=None):
    model.eval()
    all_preds = []
    all_targets = []
    
    for data in loader:
        x = data[0].to(device)
        y = data[1].to(device)
        logits, _ = model(x)
        preds = logits.argmax(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        
    # Tạo báo cáo
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    return report

# ======================================================
# Main
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_dir = f"runs/{cfg['name']}/{cfg.get('backbone', 'cnn')}"
    ckpt_dir = f"{run_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    copyfile(args.config, f"{run_dir}/config.yaml")
    logger = get_logger(f"{run_dir}/train.log")
    
    csv_path = f"{run_dir}/metrics.csv"
    log_csv = open(csv_path, "a", newline="")
    csv_writer = csv.writer(log_csv)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["epoch", "loss", "sup", "unsup", "contra", "mask", "st_acc", "te_acc", "best_acc", "lr", "ema"])

    # Data
    logger.info(f"Dataset: {cfg.get('dataset')}")
    class_names = None
    if cfg.get("dataset") == "stl10":
        labeled, unlabeled, test = get_stl10_semi(cfg["batch_size"], cfg.get("seed", 42), cfg.get("num_workers", 4))
        # Lấy tên lớp nếu có thể (thường STL10 có thuộc tính classes)
        if hasattr(test.dataset, 'classes'):
            class_names = test.dataset.classes 
        else:
            class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    else:
        labeled, unlabeled, test = get_cifar100_semi(cfg["batch_size"], cfg.get("seed", 42), cfg.get("num_workers", 4))
        # CIFAR100 có 100 lớp, có thể sẽ không in tên lớp nếu quá dài

    # Model
    bb_name = cfg.get("backbone", "resnet18")
    if bb_name == "smallcnn":
        backbone = SmallCNN()
        feat_dim = 256
    else:
        backbone = ResNet18(pretrained=True, weight_path="models/resnet18.pth")
        feat_dim = 512

    student = Model(backbone, feat_dim, cfg["emb_dim"], cfg.get("num_classes", 10)).to(device)
    teacher = copy.deepcopy(student)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters(): p.requires_grad = False
    
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg["lr"])
    scaler = GradScaler()

    # Loss
    loss_fns = {"contrastive_2006": ContrastiveLoss2006(), "triplet": TripletLoss(), "info_nce": InfoNCELoss(), "align_uniform": AlignUniformLoss()}
    loss_type = cfg.get("loss_type", "none")
    contra_fn = loss_fns.get(loss_type, None)

    # Resume & Early Stop
    early_stopping = EarlyStopping(patience=cfg.get('patience', 20))
    start_epoch, best_acc = 0, 0.0
    
    if cfg.get("resume", False) and os.path.exists(f"{ckpt_dir}/last.pth"):
        ckpt = torch.load(f"{ckpt_dir}/last.pth", map_location=device)
        student.load_state_dict(ckpt['student']); teacher.load_state_dict(ckpt['teacher'])
        optimizer.load_state_dict(ckpt['optimizer']); start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, Best Acc: {best_acc}")

    # ================= Training Loop =================
    logger.info("Start Training...")
    for epoch in range(start_epoch, cfg["epochs"]):
        curr_lr = adjust_learning_rate(optimizer, epoch, cfg)
        curr_ema = get_ema_decay(epoch, cfg['epochs'], base_ema=cfg.get('ema_decay', 0.99))
        
        stats = train_one_epoch(
            epoch, student, teacher, labeled, unlabeled, 
            optimizer, scaler, cfg, device, 
            loss_type, contra_fn, loss_type in loss_fns,
            curr_ema
        )
        
        st_acc = evaluate(student, test, device)
        te_acc = evaluate(teacher, test, device)
        
        # [LOGIC BEST]
        current_max_acc = max(st_acc, te_acc)
        is_best = current_max_acc > best_acc
        if is_best:
            best_acc = current_max_acc

        # [UPDATE] Thêm Con Loss vào log
        logger.info(f"Epoch {epoch:03d} | Loss: {stats['loss']:.4f} | Sup: {stats['sup']:.4f} | Uns: {stats['unsup']:.4f} | Con: {stats['contra']:.4f} | Mask: {stats['mask']:.2f} | EMA: {curr_ema:.5f} | Best: {best_acc:.4f}")
        
        # [NEW] Detailed Evaluation every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info("="*30)
            logger.info(f"Detailed Report @ Epoch {epoch}")
            logger.info("="*30)
            
            # Student Report
            st_report = detailed_evaluate(student, test, device, class_names)
            logger.info(f"\n[Student Model Report]\n{st_report}")
            
            # Teacher Report
            te_report = detailed_evaluate(teacher, test, device, class_names)
            logger.info(f"\n[Teacher Model Report]\n{te_report}")
            logger.info("="*30)

        csv_writer.writerow([epoch, stats["loss"], stats["sup"], stats["unsup"], stats["contra"], stats["mask"], st_acc, te_acc, best_acc, curr_lr, curr_ema])
        log_csv.flush()

        save_state = {
            "epoch": epoch,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "config": cfg
        }
        torch.save(save_state, f"{ckpt_dir}/last.pth")

        if is_best:
            torch.save(save_state, f"{ckpt_dir}/best.pth")
            logger.info(f"--> Saved New Best Acc: {best_acc:.4f}")

        early_stopping(current_max_acc)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered."); break
            
    log_csv.close()
    logger.info("Training Finished.")

if __name__ == "__main__":
    main()