import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ======================================================
# CONFIGURATION
# ======================================================
RUNS_DIR = "runs"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Cấu hình thứ tự xuất hiện trên Slide. 
# Key: Từ khóa trong tên folder run. Value: Tên hiển thị trên biểu đồ
# Bạn hãy sửa lại keyword khớp với tên folder thực tế của bạn
STAGES = [
    {"keys": ["baseline"], "label": "Supervised (Baseline)"},     # Slide 1
    {"keys": ["contrastive_2006"],       "label": "+ Contrastive"},             # Slide 2
    {"keys": ["triplet"],              "label": "+ Triplet Loss"},            # Slide 3
    {"keys": ["info", "nce"],                      "label": "+ InfoNCE"}, # Slide 4
    {"keys": ["align", "uniform"],     "label": "+ Align-Uniform Loss"}       # Slide 5
]

# Style chung
plt.style.use('seaborn-v0_8-whitegrid')
# Bảng màu mở rộng để đủ cho nhiều đường hơn
COLORS = sns.color_palette("bright", 10) 

# ======================================================
# Load Data
# ======================================================
def load_experiments(runs_dir):
    data = {}
    for root, _, files in os.walk(runs_dir):
        if "metrics.csv" in files:
            exp_name = os.path.basename(os.path.dirname(root)) 
            if exp_name in ['checkpoints'] or exp_name.isdigit():
                exp_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
                
            csv_path = os.path.join(root, "metrics.csv")
            try:
                df = pd.read_csv(csv_path)
                data[exp_name] = df
            except Exception as e:
                print(f"[WARN] Skip {csv_path}: {e}")
    return data

def get_display_name(exp_name, active_stages):
    """Mapping tên folder sang tên hiển thị dựa trên stage"""
    # Tìm stage phù hợp nhất (ưu tiên stage nằm sau cùng trong danh sách)
    for stage in reversed(active_stages):
        for k in stage["keys"]:
            if k in exp_name.lower():
                # Trả về Label chuẩn + Tên gốc (ngắn gọn) để phân biệt các lần chạy khác nhau
                short_id = exp_name[-6:] if len(exp_name) > 6 else exp_name
                return f"{stage['label']} ({short_id})"
    return exp_name

# ======================================================
# Plotting Functions
# ======================================================

def plot_accuracy(df_dict, save_path):
    plt.figure(figsize=(10, 6))
    
    # Sắp xếp để vẽ theo thứ tự đẹp (Baseline vẽ trước, cái mới vẽ sau)
    sorted_items = sorted(df_dict.items()) 
    
    for i, (name, df) in enumerate(sorted_items):
        if "te_acc" not in df.columns: continue
        
        color = COLORS[i % len(COLORS)]
        plt.plot(df["epoch"], df["te_acc"], label=name, linewidth=2.5, alpha=0.8, color=color)
        
        # Mark Best Accuracy
        best_idx = df["te_acc"].argmax()
        best_epoch = df.iloc[best_idx]["epoch"]
        best_acc = df.iloc[best_idx]["te_acc"]
        
        plt.scatter(best_epoch, best_acc, color=color, s=120, marker='*', zorder=10, edgecolors='white')
        plt.text(best_epoch, best_acc + 0.005, f"{best_acc:.2%}", fontsize=9, fontweight='bold', ha='center', color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.title("Test Accuracy Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=10, loc='lower right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_loss(df_dict, save_path):
    plt.figure(figsize=(10, 6))
    sorted_items = sorted(df_dict.items())

    for i, (name, df) in enumerate(sorted_items):
        if "loss" not in df.columns: continue
        color = COLORS[i % len(COLORS)]
        plt.plot(df["epoch"], df["loss"], label=name, linewidth=2, alpha=0.8, color=color)

    plt.title("Training Loss Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss Value (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_align_uniform(df_dict, save_path):
    plt.figure(figsize=(8, 8))
    sorted_items = sorted(df_dict.items())
    
    has_data = False
    for i, (name, df) in enumerate(sorted_items):
        color = COLORS[i % len(COLORS)]
        
        # Logic lấy dữ liệu Align/Uniform (hoặc Proxy)
        x_val, y_val = None, None
        
        # Ưu tiên 1: Cột chuẩn (nếu code train đã log)
        if "align" in df.columns and "uniform" in df.columns:
            x_val = df["align"].values
            y_val = df["uniform"].values
        # Ưu tiên 2: Proxy (Contra Loss ~ Align, Sup Loss ~ Uniform/Separability)
        # Đây là giả định để có hình vẽ nếu chưa tính chính xác Align/Uniform
        elif "contra" in df.columns and "sup" in df.columns:
            x_val = df["contra"].values
            y_val = df["sup"].values
            
        if x_val is not None:
            has_data = True
            # Vẽ quỹ đạo (Trajectory)
            plt.plot(x_val, y_val, color=color, alpha=0.4, linewidth=1.5)
            
            # Điểm bắt đầu (nhỏ)
            plt.scatter(x_val[0], y_val[0], color=color, s=30, marker='o', alpha=0.5)
            
            # Điểm kết thúc (lớn + sao)
            plt.scatter(x_val[-1], y_val[-1], color=color, s=180, label=name, edgecolors='black', marker='X', zorder=5)

    if not has_data:
        plt.text(0.5, 0.5, "Data metrics (align/uniform) not found in CSV", ha='center')
    else:
        plt.legend(loc='upper right', fontsize=9)

    plt.title("Alignment vs Uniformity Landscape\n(Goal: Lower Left ↙)", fontsize=14, fontweight='bold')
    plt.xlabel("Alignment Metric (Compactness)", fontsize=12) 
    plt.ylabel("Uniformity Metric (Separability)", fontsize=12)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ======================================================
# Main Loop
# ======================================================
def main():
    all_data = load_experiments(RUNS_DIR)
    if not all_data:
        print(f"No data found in {RUNS_DIR}/")
        return

    print(f"Found {len(all_data)} experiments.")
    print(f"Stages defined: {[s['label'] for s in STAGES]}")

    accumulated_keys = []
    
    for stage_idx, stage in enumerate(STAGES):
        # Tạo tên folder: step_1_Supervised, step_2_Contrastive...
        folder_name = f"step_{stage_idx+1}_{stage['label'].replace(' ', '_').replace('+', '').strip()}"
        step_dir = os.path.join(FIG_DIR, folder_name)
        os.makedirs(step_dir, exist_ok=True)
        
        # Tích lũy key để lọc
        accumulated_keys.extend(stage["keys"])
        
        # Lọc run:
        # 1. Tên run phải chứa key của các stage ĐÃ qua (hoặc hiện tại).
        # 2. Tên run KHÔNG được chứa key của các stage TƯƠNG LAI.
        current_runs = {}
        
        # Lấy key tương lai để loại trừ
        future_keys = []
        for future_stage in STAGES[stage_idx+1:]:
            future_keys.extend(future_stage["keys"])
            
        for exp_name, df in all_data.items():
            name_lower = exp_name.lower()
            
            # Check 1: Thuộc quá khứ/hiện tại?
            is_past_or_present = False
            for k in accumulated_keys:
                if k in name_lower:
                    is_past_or_present = True
                    break
            
            # Check 2: Thuộc tương lai?
            is_future = False
            for k in future_keys:
                if k in name_lower:
                    is_future = True
                    break
            
            # Logic: Phải thuộc (Quá khứ/Hiện tại) VÀ Không thuộc (Tương lai)
            # Ngoại lệ: Nếu key tương lai trùng key quá khứ (hiếm), cần cẩn thận. 
            # Với setup hiện tại các key khá riêng biệt (sup, 2006, triplet, nce, align) nên ổn.
            if is_past_or_present and not is_future:
                # Chỉ lấy danh sách stage tính đến hiện tại để đặt tên
                display_label = get_display_name(exp_name, STAGES[:stage_idx+1])
                current_runs[display_label] = df

        print(f"--> Generating {folder_name} ({len(current_runs)} methods)...")
        
        if current_runs:
            plot_accuracy(current_runs, os.path.join(step_dir, "accuracy.png"))
            plot_loss(current_runs, os.path.join(step_dir, "loss.png"))
            plot_align_uniform(current_runs, os.path.join(step_dir, "align_uniform.png"))
        else:
            print("   (Empty - check your folder names in 'runs/')")

    print("\nDone! Figures saved to ./figures/")

if __name__ == "__main__":
    main()