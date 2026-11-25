# Báo cáo Giai đoạn 2B (Thành viên C) — Kiểm tra chéo Tuyến B (Prosodic)

> Tài liệu này do **Thành viên C** lập theo phân công Giai đoạn 2B: độc lập chạy lại mã nguồn, kiểm chứng số liệu thực nghiệm và đánh giá chất lượng kỹ thuật của **Tuyến B (Prosodic & Chất giọng)** do Thành viên B thực hiện.

---

## 1. Phương pháp kiểm tra độc lập

Thành viên C thực hiện quy trình kiểm tra chéo theo đúng hợp đồng Bước 0:
1. Sử dụng môi trường ảo độc lập trên máy C (`.venv`, Python 3.11, Docker PostgreSQL 16).
2. Tải ma trận đặc trưng từ bảng `voice.feat_prosodic` và tham số `voice.norm_params` (đã nạp ở bước B2).
3. Chạy lệnh đánh giá chính thức trên tập kiểm thử:
   ```bash
   python -m common.evaluate tracks.track_b_prosodic --split test
   ```
4. Đối soát tự động (diff & `pandas.equals`) giữa file kết quả chạy lại và file báo cáo gốc của B ([`report/B/eval_track_b_prosodic_test.csv`](file:///e:/study/BT/csdl/voice_search/report/B/eval_track_b_prosodic_test.csv)).

---

## 2. Kết quả đối soát số liệu thực nghiệm

### 2.1 Bảng tổng hợp chỉ số trên tập `test` (30 truy vấn)

| Nhóm truy vấn | Chỉ số | Báo cáo của B | Máy C chạy lại | Độ lệch (Diff) | Kết luận |
|---|---|---:|---:|:---:|:---:|
| **Seen** (20 query) | **Top-1** | **0.250** | **0.250** | 0.000 | Trùng khớp 100% |
| | **P@5** | **0.180** | **0.180** | 0.000 | Trùng khớp 100% |
| | **MAP@5** | **0.167** | **0.167** | 0.000 | Trùng khớp 100% |
| **Unseen** (10 query) | **group@5** | **0.800** | **0.800** | 0.000 | Trùng khớp 100% |
| | **lang@5** | **0.940** | **0.940** | 0.000 | Trùng khớp 100% |

### 2.2 Đối soát từng truy vấn (Row-by-Row Check)
- Chạy kiểm tra tự động qua script: `df_b.equals(df_c) == True`.
- **Toàn bộ 30 dòng** (20 câu seen từ Q21–Q40 và 10 câu unseen từ Q01–Q10) đều trả về danh sách `top5` file giống hệt nhau, thứ tự xếp hạng không lệch một vị trí nào.

---

## 3. Nhận xét chuyên môn về Tuyến B

### 3.1 Ưu điểm & Điểm sáng kỹ thuật
1. **Mã nguồn phòng thủ cao (Defensive Programming):**
   - File [`tracks/track_b_prosodic.py`](file:///e:/study/BT/csdl/voice_search/tracks/track_b_prosodic.py) bọc `try...except` riêng biệt cho từng khối trích xuất của Praat (`Pitch`, `Intensity`, `PointProcess` cho Jitter/Shimmer, `Harmonicity` cho HNR).
   - Tránh hoàn toàn lỗi crash khi gặp file âm thanh quá ngắn (< 1s) hoặc đoạn nói thì thầm không có thanh (unvoiced).
   - Sử dụng `np.nan_to_num` ở đầu ra đảm bảo vector luôn là số thực hữu hạn (`float64`).
2. **Xử lý độ đo Mahalanobis thông minh:**
   - Tuyến B chọn Mahalanobis làm độ đo mặc định (`DEFAULT_METRIC = "mahalanobis"`).
   - Mã nguồn đã thêm thành phần điều chuẩn Tikhonov (`cov + 1e-3 * np.eye(...)`) trước khi nghịch đảo ma trận hiệp phương sai, giải quyết triệt để nguy cơ ma trận kỳ dị do đa cộng tuyến giữa các biến F0 (`f0_mean`, `f0_min`, `f0_max`).
3. **Phân hóa vùng miền & ngôn ngữ rất tốt:**
   - Dù chỉ có 11 đặc trưng, Tuyến B đạt `group@5 = 0.800` và `lang@5 = 0.940` trên tập Unseen.
   - Thống kê F0 trung bình tách bạch rõ rệt giữa các nhóm giọng: Nam (144.3 Hz) > US (118.1 Hz) > UK (112.7 Hz).

### 3.2 Hạn chế nội tại của bộ đặc trưng
1. **Khả năng định danh người nói đơn lẻ thấp:**
   - Trên tập Seen, Top-1 chỉ đạt 25.0% (5/20 câu đúng #1: Q21, Q22, Q31, Q32, Q35), MAP@5 chỉ đạt 0.167 (kém xa Embedding 1.000 và Spectral 0.368).
   - *Nguyên nhân kỹ thuật:* 11 đặc trưng prosodic đo lường đặc tính âm học vĩ mô (cao độ, tốc độ, rung giọng). Hai người nói khác nhau hoàn toàn có thể có cùng tông giọng trầm và cùng nhịp độ nói, do đó không đủ thông tin nhận diện danh tính (*speaker identity*).
2. **Vai trò trong hệ thống hợp nhất:**
   - Việc B phân bổ trọng số $w_{\text{pros}} = 0.05$ trong module [`tracks/fusion.py`](file:///e:/study/BT/csdl/voice_search/tracks/fusion.py) là hoàn toàn hợp lý: Prosodic đóng vai trò tinh chỉnh nhịp điệu và ngữ điệu phụ trợ, nhường quyền quyết định chính cho Embedding (0.80) và Spectral (0.15).

---

## 4. Kết luận kiểm tra chéo

- **Tình trạng:** **ĐẠT (PASSED)**.
- Kết quả thực nghiệm của Tuyến B hoàn toàn tái lập được từ đầu trên máy tính khác.
- Không phát hiện lỗi logic hay sai số tiềm ẩn trong mã nguồn của B.
