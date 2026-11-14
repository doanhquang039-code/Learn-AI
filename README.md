# Chủ đề 04: Nhận dạng hoạt động qua cảm biến smartphone (UCI HAR)

Bài tập lớn môn Nhập môn Trí tuệ nhân tạo.
Thành viên nhóm: Đặng Quang Doanh, Cao Bá Sơn, Nguyễn Đức Minh, Lê Tuấn Nghĩa, Trần Minh Tú.

## 1. Bài toán
Phân loại 6 hoạt động (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`) từ vector 561 đặc trưng cảm biến smartphone (gia tốc kế và con quay hồi chuyển) của bộ dữ liệu UCI Human Activity Recognition Using Smartphones.

- **Đầu vào:** vector 561 đặc trưng cho mỗi cửa sổ tín hiệu (2.56 giây, 128 mẫu đọc ở tần số 50 Hz, chồng lấn 50%).
- **Đầu ra:** nhãn hoạt động dự đoán và xác suất của từng lớp.
- **Baseline:** Logistic Regression có chuẩn hóa (`StandardScaler`, `C=1.0`).
- **Mô hình so sánh:** MLP nhỏ (`StandardScaler`, `hidden_layer_sizes=(128,)`, `alpha=0.01`).
- **Chỉ số đánh giá:** Accuracy, Macro-F1, kèm F1 từng lớp và ma trận nhầm lẫn (confusion matrix).

## 2. Dữ liệu
- **Nguồn:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones (DOI 10.24432/C54S4K)
- **Giấy phép:** CC BY 4.0 (được dùng và chỉnh sửa, phải ghi nguồn).
- **Quy mô:** 10.299 mẫu, 30 người; chia sẵn train (7.352 mẫu, 21 người) và test (2.947 mẫu, 9 người), không trùng người.
- **Trích dẫn:** D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition using Smartphones.* ESANN, 2013.

## 3. Kết quả đánh giá mô hình (Tập test, Seed 42)

| Mô hình | Accuracy | Macro-F1 | Ghi chú |
|---|---|---|---|
| **Logistic Regression (Baseline)** | **0,9498** | **0,9500** | Chuẩn hóa StandardScaler, C=1.0 |
| **MLP (Mạng Nơ-ron)** | **0,9477** | **0,9480** | StandardScaler, 1 lớp ẩn (128,) |

- Cả hai mô hình đạt hiệu năng xấp xỉ ~95%.
- Cặp **SITTING / STANDING** là cặp bị nhầm lẫn nhiều nhất ở cả hai mô hình (do góc nghiêng và véc-tơ trọng trường ở thắt lưng của hai tư thế tĩnh gần như trùng khớp).

## 4. Cấu trúc repo

```
├── README.md
├── requirements.txt
├── har_baseline_mlp (1).py
├── web_app.py
├── index.html
├── real_12_rows.csv              # 12 mẫu kiểm thử thực tế kèm nhãn thật (~80 KB)
├── sample_data/                  # Dữ liệu mẫu kiểm thử và kịch bản lỗi
│   ├── real_12_rows.csv
│   ├── real_12_rows_features_only.csv
│   ├── real_12_rows_labels.txt
│   ├── real_20_sequence.csv
│   ├── real_20_sequence.json
│   ├── bad_column_count.csv
│   ├── bad_contains_text.csv
│   └── bad_empty_cell.csv
├── outputs/                      # Kết quả đánh giá khoa học và mô hình
│   ├── cm_LogisticRegression.png
│   ├── cm_MLP.png
│   ├── lr_har.joblib
│   ├── mlp_har.joblib
│   └── model_metrics.json        # Accuracy, Macro-F1, F1 từng lớp, bảng 5 seed
└── tests/
    └── test_web_app.py           # 11 ca kiểm thử đơn vị tự động
```

## 5. Các tính năng nổi bật trên Web App

1. **Thử nhanh mẫu có sẵn (Quick Predict):**
   - Tích hợp 12 mẫu kiểm thử thực tế từ tập test (`real_12_rows.csv` kèm nhãn thật).
   - Chọn mẫu từ menu thả xuống, bấm 1 nút là thấy ngay kết quả dự đoán và nhãn thật.
2. **Biểu đồ xác suất 6 lớp dạng thanh ngang:**
   - Trực quan hóa xác suất từng lớp cho `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`.
   - Giúp nhận diện rõ ràng trạng thái mô hình phân vân giữa `SITTING` và `STANDING`.
3. **So sánh khoa học Logistic Regression và MLP:**
   - Tùy chọn so sánh trực tiếp cả 2 mô hình (LR vs MLP) trên cùng một mẫu dữ liệu.
4. **Dán trực tiếp 561 số:**
   - Ô văn bản dán vector 561 đặc trưng kèm bộ kiểm tra định dạng chặt chẽ (sai số lượng cột, chứa chữ, ô trống, NaN/Inf).
   - Có sẵn các nút bấm nhanh kịch bản kiểm tra lỗi (`bad_*.csv`).
5. **Đọc trực tiếp kết quả từ `outputs/`:**
   - Tab "Kết quả mô hình" nạp trực tiếp bảng Accuracy/Macro-F1, bảng F1 từng lớp, bảng 5 seed và 2 ảnh Confusion Matrix từ `outputs/`.
6. **Tải CSV hàng loạt & Mini-Colab:**
   - Cho phép tải CSV nhiều dòng kèm tệp nhãn thật tùy chọn. Nếu có nhãn, web tự động tính Accuracy, Macro-F1 và vẽ Ma trận nhầm lẫn tương tác ngay trên web!
   - Cảnh báo độ tin cậy thấp khi xác suất cao nhất < 60%.
   - Nút tải kết quả dự đoán thành tệp CSV đầy đủ.
7. **Mô phỏng chuỗi 20 mẫu (Sequence Replay):**
   - Phát lại 20 cửa sổ dữ liệu kiểm thử liên tiếp theo thời gian thực (1 giây/bước) để quan sát chuyển đổi trạng thái hoạt động.
   - Có thông báo lưu ý rõ ràng: *Mô phỏng từ dữ liệu kiểm thử ghi trước, không phải cảm biến thời gian thực sống.*
8. **Trang Hạn chế & Nguồn dữ liệu:**
   - Giấy phép CC BY 4.0, trích dẫn khoa học chính thức, quy trình trích xuất đặc trưng và các giới hạn thực tế của bài toán.

## 6. Hướng dẫn chạy

### Chạy trên máy nội bộ

```bash
pip install -r requirements.txt
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt tại: `http://127.0.0.1:8000`
Tài liệu API Swagger tại: `http://127.0.0.1:8000/docs`

### Chạy kiểm thử tự động (Unit Tests)

```bash
python -m unittest discover -s tests -v
```

### Chạy trên Google Colab

```python
!git clone https://github.com/doanhquang039-code/Learn-AI.git /content/Learn-AI
%cd /content/Learn-AI
!pip install -r requirements.txt
```

```python
import subprocess
import time
import urllib.request
from google.colab.output import eval_js

server = subprocess.Popen([
    "python", "-m", "uvicorn", "web_app:app",
    "--host", "0.0.0.0", "--port", "8000"
])
time.sleep(2)
print(eval_js("google.colab.kernel.proxyPort(8000)"))
```
