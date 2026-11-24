# Báo cáo Tuyến B (Prosodic & Chất giọng) và Hợp nhất điểm (Giai đoạn 2B)

> Tài liệu này tổng hợp kết quả thực nghiệm của **Thành viên B** cho **Mục 3, 4 (T1–T6)** trên tuyến Prosodic (`feat_prosodic`, 11 đặc trưng) và phần **Hợp nhất điểm 3 tuyến (`search_fused`)** ở Giai đoạn 2B.

---

## 1. Trích rút đặc trưng & Kiểm tra lặp lại (T1)

Module: [tracks/track_b_prosodic.py](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/tracks/track_b_prosodic.py) (`extract(wav_path)`).

- **11 đặc trưng Prosodic & Chất giọng:** `f0_mean`, `f0_std`, `f0_min`, `f0_max` (Pitch autocorrelation 75–300 Hz, chỉ tính khung có thanh); `energy_mean`, `energy_std` (Intensity dB); `speaking_rate` (đỉnh năng lượng độc lập ngôn ngữ); `jitter_local` (%), `shimmer_local` (dB); `hnr_mean` (dB); `zcr_mean`.
- **Kiểm tra trên 20 file cũ** (phân tầng `Nam` / `UK` / `US` trong `data/wav/`, lưu chi tiết tại [t1_prosodic_verify_20files.csv](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/t1_prosodic_verify_20files.csv)):
  - Khi bỏ qua bước tiền xử lý lặp lại (`skip_preprocess=True`, vì 1.000 file gốc đã chuẩn hoá sẵn): sai số tuyệt đối lớn nhất `0.00e+00`, sai số tương đối `0.00e+00` (khớp tuyệt đối 100% với `docs/B/features_prosodic.csv`).
  - Khi chạy lại qua `common.preprocess.preprocess()` (`skip_preprocess=False`): trung vị sai số tương đối `0.0000`, lớn nhất `0.1821` trên một số file bị lệch biên cắt lặng vài chục mili-giây (đúng như ghi chú hợp đồng Bước 0).

---

## 2. Cơ sở dữ liệu PostgreSQL (T2)

Script nạp dữ liệu: [tracks/load_prosodic.py](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/tracks/load_prosodic.py).

- Nạp 1.000 dòng vào bảng `voice.feat_prosodic` (khóa chính `file_id` tham chiếu `voice.audio_files(file_id)`).
- Tính và lưu 11 cặp `(mean, std)` trên toàn bộ 1.000 file vào `voice.norm_params` với `feature_set = 'prosodic'`.
- Hàm `load_matrix(conn)` đọc ma trận `(1000, 11)` kiểu `float64` kèm `norm_params` để phục vụ chuẩn hoá z-score khi tìm kiếm.

---

## 3. Lựa chọn độ đo tìm kiếm trên tập `dev` (T3)

Chạy trên 20 truy vấn `dev` (10 `seen`, 10 `unseen`) bằng `common.evaluate.evaluate`, sau khi chuẩn hoá z-score theo `norm_params` ([t3_prosodic_metrics_dev.csv](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/t3_prosodic_metrics_dev.csv)):

| Độ đo | Seen Top-1 | Seen P@5 | Seen MAP@5 | Unseen group@5 | Unseen lang@5 |
|---|---:|---:|---:|---:|---:|
| **Mahalanobis** (reg `1e-3·I`) | **0.100** | **0.120** | **0.0797** | **0.880** | 0.940 |
| Euclid | 0.100 | 0.100 | 0.0573 | 0.880 | **0.960** |
| Cosine | 0.100 | 0.080 | 0.0700 | 0.840 | 0.900 |

- **Kết luận T3:** Chọn **Mahalanobis** (`DEFAULT_METRIC = "mahalanobis"`) do đạt `Seen P@5` (`0.120`) và `Seen MAP@5` (`0.0797`) cao nhất trên `dev`. Lý do: một số đặc trưng prosodic có tương quan mạnh với nhau (ví dụ `f0_mean`–`f0_min`–`f0_max` hay `jitter`–`hnr`), nên Mahalanobis khử được phần tương quan lặp lại tốt hơn Euclid.

---

## 4. Minh hoạ kết quả trung gian (T4)

Chi tiết vector thô, vector z-score, khoảng cách và xếp hạng top-5 được lưu tại [t4_prosodic_intermediate.csv](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/t4_prosodic_intermediate.csv) và [t4_prosodic_intermediate.json](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/t4_prosodic_intermediate.json).

1. **Truy vấn `seen`: `Q21` (`A_0744`, `SPK_VIVOSDEV03`, `Nam`)**
   - `f0_mean = 142.46 Hz` (`z = +0.74`), `f0_min = 111.57 Hz` (`z = +1.16`), `speaking_rate = 4.91` (`z = +1.26`).
   - Top-5 trả về **5/5 file cùng speaker `SPK_VIVOSDEV03`**: `#1 A_0722` (`sim = 0.490`, `d = 1.040`), `#2 A_0745` (`0.469`), `#3 A_0747` (`0.443`), `#4 A_0731` (`0.422`), `#5 A_0718` (`0.408`).
   - Biểu đồ: [fig_t4_prosodic_seen_compare.png](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/fig_t4_prosodic_seen_compare.png).

2. **Truy vấn `unseen`: `Q01` (`A_0652`, `SPK_VIVOSDEV01`, `Nam`)** (toàn bộ file của `SPK_VIVOSDEV01` bị loại khỏi CSDL)
   - `f0_mean = 147.72 Hz` (`z = +0.98`), `hnr_mean = 13.13 dB` (`z = +0.89`), `jitter_local = 1.30%` (`z = -1.14`).
   - Top-5 trả về **5/5 file tiếng Việt giọng `Nam`** (`SPK_VIVOSDEV05`, `06`, `07`, `03`): `#1 A_0797` (`sim = 0.422`, `d = 1.369`), `#2 A_0832` (`0.390`), `#3 A_0871` (`0.365`), `#4 A_0880` (`0.363`), `#5 A_0717` (`0.356`).
   - Biểu đồ: [fig_t4_prosodic_unseen_compare.png](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/fig_t4_prosodic_unseen_compare.png).

---

## 5. Đánh giá trên tập `test` & Phân tích lỗi (T5)

Chạy `track_b_prosodic` với độ đo `mahalanobis` ([eval_track_b_prosodic_test.csv](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/eval_track_b_prosodic_test.csv)):

- **`seen` (`n = 20`):** `top1 = 0.250`, `P@5 = 0.180`, `MAP@5 = 0.167`
- **`unseen` (`n = 10`):** `group@5 = 0.800`, `lang@5 = 0.940`

**Phân tích các trường hợp sai ([t5_prosodic_error_analysis.csv](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/t5_prosodic_error_analysis.csv)):**
- Với `seen`, 15/20 truy vấn không đúng Top-1 ở tuyến prosodic đơn lẻ. Phần lớn câu sai rơi vào các clip ngắn (`1.0–1.5s`, ví dụ `Q24 A_0771 = 1.38s`, `Q27 A_0378 = 1.02s`, `Q28 A_0408 = 1.18s`), nơi `jitter`, `shimmer`, `speaking_rate` dao động mạnh, hoặc giữa các speaker có khoảng `f0_mean` chồng lấn (ví dụ giữa các speaker `US` quanh 110–125 Hz).
- Với `unseen`, 8/10 truy vấn đạt `group@5 >= 0.6`; 2 câu có `group@5 = 0.4` đều là `UK` ngắn (`Q03 A_0368 = 1.22s` nhầm sang `US`, `Q06 A_0582 = 1.34s` nhầm sang `Nam`/`US`) vì giọng `UK` và `US` có phân bố `f0_mean` gần nhau (113 Hz vs 118 Hz).
- Bảng mẫu chấm nghe thử 10 truy vấn (để cả 3 thành viên điền điểm 1–5) đã tạo tại [t5_listening_eval_template.csv](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/t5_listening_eval_template.csv).

---

## 6. Tab Demo Streamlit (T6)

- [app/tab_b_prosodic.py](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/app/tab_b_prosodic.py) cung cấp `render_tab(conn=None)` (tự động đọc PostgreSQL nếu có kết nối, hoặc fallback sang CSV offline): cho phép tải file mới hoặc chọn mẫu từ `data/wav/`, nghe lại truy vấn và từng kết quả trong top-5, xem bảng 11 đặc trưng + z-score, và vẽ biểu đồ so sánh với kết quả hạng 1.
- Có thể chạy độc lập bằng `streamlit run app/app.py` ([app/app.py](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/app/app.py)).

---

## 7. Hợp nhất điểm 3 tuyến — Giai đoạn 2B (`search_fused`)

Module: [tracks/fusion.py](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/tracks/fusion.py).

- **Chuẩn hoá thang điểm:** Cả 3 tuyến đưa về `similarity ∈ [0, 1]` (Prosodic: Mahalanobis `1/(1+d)`; Spectral: z-score + Cosine `1/(1+d)` khớp với `tracks/track_c_spectral.py`; Embedding: Cosine `(cos+1)/2` khớp với `tracks/track_a_embedding.py`).
- **Chọn trọng số trên `dev`:** Grid search bước `0.05` ([eval_fusion_weights_dev.csv](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/eval_fusion_weights_dev.csv)) chọn **`w_prosodic = 0.05`, `w_spectral = 0.15`, `w_embedding = 0.80`**.
- **Bảng so sánh 4 hệ thống ([eval_fusion_comparison.csv](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/eval_fusion_comparison.csv), [fig_2b_fusion_comparison.png](file:///home/dunggawr/Documents/voice_search_buoc0/voice_search/report/B/fig_2b_fusion_comparison.png)):**

| Tập | Hệ thống | Seen Top-1 | Seen P@5 | Seen MAP@5 | Unseen group@5 | Unseen lang@5 |
|---|---|---:|---:|---:|---:|---:|
| `dev` | Prosodic (B) | 0.100 | 0.120 | 0.0797 | 0.880 | 0.940 |
| `dev` | Spectral (A) | 0.700 | 0.460 | 0.4170 | 0.900 | 0.900 |
| `dev` | Embedding (C) | 1.000 | 0.980 | 0.9950 | 0.800 | 1.000 |
| `dev` | **Fused (0.05/0.15/0.80)** | **1.000** | **0.980** | **1.0000** | **0.980** | **1.000** |
| `test` | Prosodic (B) | 0.250 | 0.180 | 0.1670 | 0.800 | 0.940 |
| `test` | Spectral (A) | 0.600 | 0.430 | 0.3675 | 0.980 | 0.980 |
| `test` | Embedding (C) | 1.000 | 1.000 | 1.0000 | 0.820 | 0.980 |
| `test` | **Fused (0.05/0.15/0.80)** | **1.000** | **1.000** | **1.0000** | **0.960** | **1.000** |

**Nhận xét 2B:** Kết hợp trọng số nhỏ từ Prosodic (`0.05`) và Spectral (`0.15`) với Embedding (`0.80`) giữ nguyên **tuyệt đối** `Top-1 = 1.000`, `P@5 = 1.000`, `MAP@5 = 1.0000` cho `seen` trên `test`, đồng thời nâng rõ rệt `unseen group@5` từ `0.820` lên `0.960` và `unseen lang@5` từ `0.980` lên `1.000` (bù đúng điểm yếu "embedding ít giữ thông tin vùng/accent giữa UK và US" đã dự báo ở Bước 0).
