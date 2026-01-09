# Contrastive Learning Comparison Demo


Dự án này là một bộ khung (framework) thực nghiệm để so sánh hiệu quả của các phương pháp Contrastive Learning phổ biến trên bộ dữ liệu **STL-10** (và CIFAR-100). Dự án tập trung vào việc minh họa trực quan sự khác biệt giữa các hàm Loss (Contrastive 2006, Triplet, InfoNCE, Alignment & Uniformity) thông qua các biểu đồ t-SNE và phân tích hình học trên hypersphere.

## 📌 Tính năng chính

* **Training Pipeline:** Hỗ trợ huấn luyện nhiều phương pháp:
    * **Baseline:** Supervised Cross-Entropy (Softmax).
    * **Contrastive Loss (2006):** Yann LeCun et al.
    * **Triplet Loss:** FaceNet, Schroff et al.
    * **InfoNCE:** SimCLR, MoCo (Oord et al.).
    * **Align-Uniform Loss:** Wang & Isola (2020).
* **Visualization:**
    * **t-SNE:** Giảm chiều dữ liệu để quan sát sự phân cụm.
    * **Hypersphere Analysis:** Vẽ biểu đồ phân bố Feature và Histogram khoảng cách (giống paper Wang & Isola).
    * **Metrics Comparison:** So sánh Accuracy, Loss, Alignment/Uniformity giữa các model.

---

## 🛠️ Cài đặt Môi trường

Dự án yêu cầu **Python 3.8+** và **PyTorch**.

### 1. Tạo môi trường từ `environment.yml`
Nếu bạn đã có file `environment.yml`:

```bash
conda env create -f environment.yml
conda activate contrastive-demo

```

### 2. Cài đặt thủ công (nếu chưa có file yml)

```bash
# Cài đặt các thư viện cơ bản
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn tqdm pyyaml

```

---

## 🚀 Hướng dẫn Sử dụng

### 1. Huấn luyện (Training)

Bạn có thể chạy training cho từng phương pháp bằng cách chỉ định file config tương ứng.

**Chạy đơn lẻ:**

```bash
# Baseline (Supervised)
python train.py --config config/baseline_stl10.yaml

# Contrastive Loss (2006)
python train.py --config config/contrastive_2006_stl10.yaml

# Triplet Loss
python train.py --config config/triplet_stl10.yaml

# InfoNCE
python train.py --config config/info_nce_stl10.yaml

# Align-Uniform
python train.py --config config/align_uniform_stl10.yaml

```

**Chạy toàn bộ các training cho mọi method:**
```bash
bash train.sh
```

**Chạy tất cả (Automation):**
Nếu bạn muốn chạy toàn bộ thực nghiệm qua đêm:

```bash
bash run_all.sh

```

*(Kết quả training, checkpoint và log sẽ được lưu trong thư mục `runs/`)*

---

### 2. Đánh giá (Evaluation) & Vẽ t-SNE

Sau khi có checkpoint (file `.pth`) trong thư mục `runs/`, bạn có thể chạy script t-SNE để visualize không gian embedding.

```bash
python tsne.py

```

* Script sẽ tự động tìm các checkpoint tốt nhất (`best.pth`) trong `runs/`.
* Kết quả lưu tại: `figures/tsne/`.
* Biểu đồ hiển thị sự phân tách lớp của các method khác nhau.

---

### 3. Phân tích Hình học (Paper Style Visualization)

Script này tạo ra các biểu đồ Alignment (Histogram khoảng cách) và Uniformity (KDE trên đường tròn) để đánh giá chất lượng feature representation (theo phong cách paper *Understanding Contrastive Representation...*).

```bash
python visualize_paper.py

```

* Kết quả lưu tại: `figures/paper_style/`.
* Giúp bạn trả lời câu hỏi: *"Feature có phân bố đều trên mặt cầu không? Các cặp positive có gần nhau không?"*

---

### 4. So sánh Tổng hợp (Comparison Plots)

Script này tổng hợp dữ liệu từ `metrics.csv` của tất cả các lần chạy để vẽ biểu đồ so sánh (Accuracy, Loss, Alignment-Uniformity Trade-off). Nó tự động chia kết quả thành từng Step để đưa vào Slide thuyết trình.

```bash
python comparation.py

```

* Kết quả lưu tại: `figures/step_1...`, `figures/step_2...`, v.v.
* **Step 1:** Chỉ Baseline.
* **Step 2:** Baseline + Contrastive 2006.
* **Step 3:** Baseline + Contra + Triplet.
* **Step 4:** Baseline + Contra + Triplet + InfoNCE.
* **Step 5:** Đầy đủ các phương pháp.

---

## 📂 Cấu trúc Thư mục

```
.
├── config/                 # Chứa file cấu hình (.yaml) cho từng method
├── data/                   # Code xử lý dữ liệu (CIFAR100, STL10)
├── losses/                 # Cài đặt các hàm Loss (Triplet, InfoNCE, etc.)
├── models/                 # Backbone (ResNet, SmallCNN) và Projection Head
├── runs/                   # Nơi lưu Checkpoint, Log và Metrics sau khi train
├── figures/                # Nơi lưu tất cả biểu đồ đầu ra
│   ├── paper_style/        # Biểu đồ Alignment/Uniformity
│   ├── step_X.../          # Biểu đồ so sánh theo giai đoạn
│   └── tsne/               # Biểu đồ t-SNE
├── train.py                # Script huấn luyện chính
├── tsne.py                 # Script vẽ t-SNE
├── visualize_paper.py      # Script vẽ biểu đồ lý thuyết (Align/Unif)
└── comparation.py          # Script vẽ biểu đồ so sánh tổng hợp

```

## 📝 Ghi chú

* Dữ liệu **STL-10** sẽ tự động được tải về thư mục `data/` trong lần chạy đầu tiên.
* Để chỉnh sửa tham số (Learning rate, Batch size, Epochs), hãy sửa trực tiếp trong các file `.yaml` tại thư mục `config/`.
