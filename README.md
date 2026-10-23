# Chủ đề 04: Nhận dạng hoạt động qua cảm biến smartphone

Bài tập lớn môn Nhập môn Trí tuệ nhân tạo.
Thành viên nhóm : Đặng Quang Doanh , Cao Bá Sơn , Nguyễn Đức Minh , Lê Tuấn Nghĩa , Trần Minh Tú

## 1. Bài toán
Phân loại 6 hoạt động (walking, upstairs, downstairs, sitting, standing, laying) từ vector 561 đặc trưng cảm biến smartphone (gia tốc kế và con quay hồi chuyển) của bộ dữ liệu UCI Human Activity Recognition Using Smartphones.

- **Đầu vào:** vector 561 đặc trưng cho mỗi cửa sổ tín hiệu.
- **Đầu ra:** nhãn hoạt động dự đoán và xác suất.
- **Baseline:** Logistic Regression có chuẩn hóa (StandardScaler).
- **Mô hình so sánh:** MLP nhỏ (scikit-learn).
- **Chỉ số đánh giá:** Accuracy, Macro-F1, kèm F1 từng lớp và confusion matrix.

## 2. Dữ liệu
- Nguồn: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones (DOI 10.24432/C54S4K)
- Giấy phép: CC BY 4.0 (được dùng và chỉnh sửa, phải ghi nguồn).
- 10.299 mẫu, 30 người; chia sẵn train (7.352 mẫu, 21 người) và test (2.947 mẫu, 9 người), không trùng người.
- **Dữ liệu gốc không đưa lên repo.** Tải về theo mục 4.

Trích dẫn: D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz. *A Public Domain Dataset for Human Activity Recognition using Smartphones.* ESANN, 2013.

## 3. Phương pháp
- Chuẩn hóa bằng StandardScaler, chỉ fit trên tập train.
- Validation tách từ train **theo người** (GroupShuffleSplit theo subject), có kiểm tra không trùng người; chọn siêu tham số theo Macro-F1 trên validation.
- Logistic Regression: thử C trong {0,01; 0,1; 1; 10}.
- MLP: thử lớp ẩn (128) và (128, 64), alpha trong {0,0001; 0,01}.
- Sau khi chọn, huấn luyện lại trên toàn bộ train rồi đánh giá một lần trên test.
- Seed cố định: 42.

## 4. Cách chạy lại
```bash
pip install -r requirements.txt

# Tải và giải nén dữ liệu (Linux/Colab)
wget -O har.zip "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
unzip -q har.zip
unzip -q "UCI HAR Dataset.zip"      # file zip nằm bên trong file zip đầu

# Huấn luyện, đánh giá và chạy demo
python har_baseline_mlp.py --data "UCI HAR Dataset" --demo
```
Trên Google Colab, thêm dấu `!` ở đầu mỗi dòng lệnh. Kết quả (ảnh confusion matrix, mô hình `.joblib`) được lưu trong thư mục `outputs/`.

## 5. Kết quả (tập test, seed 42)
| Mô hình | Accuracy | Macro-F1 |
|---|---|---|
| Logistic Regression (baseline) | 0,9498 | 0,9500 |
| MLP | 0,9477 | 0,9480 |

- Hai mô hình gần như tương đương; cặp **sitting/standing** là cặp bị nhầm nhiều nhất ở cả hai.
- Kết quả có thể lệch nhẹ nếu dùng phiên bản scikit-learn khác.

## 6. Demo
Hàm `predict_one(vector_561)` trong `har_baseline_mlp.py` nhận một vector 561 đặc trưng, trả về nhãn và xác suất. Chạy với `--demo` để thử trên dòng test số 0.

## 7. Hạn chế
Dữ liệu chỉ thu ở một loại thiết bị (Samsung Galaxy S II), một vị trí đeo (thắt lưng), 30 người và điều kiện cụ thể, nên không khẳng định ứng dụng đại trà. Đây là nhận dạng hoạt động, không dùng để suy đoán bệnh lý hay sức khỏe cá nhân.

## 8. Cấu trúc repo
```
├── README.md
├── requirements.txt
├── har_baseline_mlp.py
└── outputs/          # ảnh confusion matrix và mô hình đã huấn luyện
```
