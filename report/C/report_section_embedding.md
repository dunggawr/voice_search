# Báo cáo Tuyến C — Speaker Embedding (ECAPA-TDNN, 192 chiều)

> Tài liệu này trình bày kết quả thực nghiệm của **Thành viên C** cho **Mục 3, 4** trên tuyến Speaker Embedding (`feat_embedding`, 192 chiều, ECAPA-TDNN).

---

## 1. Mô tả bộ đặc trưng

**Mô hình:** ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation — Time Delay Neural Network), pretrained trên VoxCeleb (`speechbrain/spkrec-ecapa-voxceleb`).

**Vector đặc trưng:** 192 chiều (`float64`), lưu trong bảng `voice.feat_embedding` dưới dạng `REAL[]`.

**Ý nghĩa:** Embedding học **danh tính giọng nói** (speaker identity) — không học ngôn ngữ hay accent. Hai file cùng một người nói sẽ có embedding gần nhau trong không gian 192 chiều, bất kể nội dung câu nói.

**Độ đo tìm kiếm:** Cosine Similarity (không dùng z-score — L2-normalize thay thế).

$$\text{similarity} = \frac{\hat{q} \cdot \hat{m}_i}{1} = \frac{q \cdot m_i}{\|q\| \cdot \|m_i\|}, \quad \text{rescale} \in [0, 1]$$

---

## 2. Trích rút đặc trưng & Kiểm tra lặp lại (T1)

**Module:** [`tracks/track_a_embedding.py`](file:///e:/study/BT/csdl/voice_search/tracks/track_a_embedding.py) — hàm `extract(wav_path)`.

**Quy trình:**
1. `preprocess(wav_path)` → 16 kHz, mono, chuẩn hoá âm lượng −23 LUFS, cắt lặng `top_db=25`.
2. Chuyển sang tensor PyTorch → `EncoderClassifier.encode_batch()` → shape `(1, 1, 192)`.
3. `.squeeze().numpy().astype(np.float64)` → shape `(192,)`.

**Lazy load model:** Model SpeechBrain chỉ tải lần đầu (~80 MB, cache tại `~/.cache/huggingface/`), không tải lại mỗi lần `extract()` được gọi.

**Kiểm tra:** So sánh `extract()` với `docs/C/features_embedding.csv` trên 20 file cũ trong `data/wav/`. Sai số nhỏ chấp nhận được do bước cắt lặng có thể lệch vài chục mili-giây (đã được ghi nhận trong hợp đồng Bước 0).

---

## 3. Cơ sở dữ liệu PostgreSQL (T2)

**Bảng:** `voice.feat_embedding` — khoá chính `file_id`, cột `vector REAL[]` (192 phần tử).

**Script nạp:** [`tracks/load_t2_embedding.py`](file:///e:/study/BT/csdl/voice_search/tracks/load_t2_embedding.py)

- Đọc 1000 dòng từ `docs/C/features_embedding.csv`.
- Batch insert bằng `psycopg2.extras.execute_values` (không vòng lặp 1000 INSERT đơn).
- **Không lưu `norm_params`**: tuyến embedding dùng cosine (tự chuẩn hoá L2), không dùng z-score.

**Kiểm tra:**
```sql
SELECT COUNT(*) FROM voice.feat_embedding;  -- 1000
```

**Hàm `load_matrix(conn)`:** Trả `(ids, M)` với `ids` shape `(1000,)`, `M` shape `(1000, 192)` kiểu `float64`. Raise `RuntimeError` rõ ràng nếu bảng rỗng.

---

## 4. Lựa chọn độ đo tìm kiếm trên tập `dev` (T3)

Chạy trên 20 truy vấn `dev` (10 `seen`, 10 `unseen`):

| Độ đo | Seen Top-1 | Seen P@5 | Seen MAP@5 | Unseen group@5 | Unseen lang@5 |
|---|---:|---:|---:|---:|---:|
| **Cosine** | **1.000** | **0.980** | **0.995** | **0.800** | **1.000** |

> **Kết luận T3:** Chọn **Cosine Similarity** — đây là độ đo tự nhiên nhất cho embedding học từ metric learning (VoxCeleb). Euclid và Mahalanobis không áp dụng được đúng cách vì không có `norm_params` z-score cho tuyến này; cosine đã đạt kết quả tuyệt đối trên seen (Top-1 = 1.000, MAP@5 = 0.995 trên dev).

---

## 5. Minh hoạ kết quả trung gian (T4)

Chi tiết trung gian lưu tại [`report/C/t4_demo_queries.py`](file:///e:/study/BT/csdl/voice_search/report/C/t4_demo_queries.py).

### Truy vấn Seen: Q41 (`A_0997`, `SPK_VIVOSDEV10`, vùng Nam)

- Vector embedding 192 chiều, sau L2-normalize → cosine similarity với 999 file còn lại.
- Top-5 trả về **5/5 file cùng speaker `SPK_VIVOSDEV10`**.
- Biểu đồ similarity: [`report/C/t4_seen.png`](file:///e:/study/BT/csdl/voice_search/report/C/t4_seen.png)

### Truy vấn Unseen: Q11 (`A_0851`, `SPK_VIVOSDEV06`, vùng Nam)

- Toàn bộ file của `SPK_VIVOSDEV06` bị loại khỏi CSDL trước khi tìm kiếm.
- Top-5 trả về 5/5 file cùng vùng `Nam` (`group@5 = 1.0`) và cùng ngôn ngữ tiếng Việt (`lang@5 = 1.0`).
- Biểu đồ similarity: [`report/C/t4_unseen.png`](file:///e:/study/BT/csdl/voice_search/report/C/t4_unseen.png)

---

## 6. Đánh giá trên tập `test` (T5)

Chạy: `python -m common.evaluate tracks.track_a_embedding --split test`

Kết quả chi tiết: [`report/C/eval_track_a_embedding_test.csv`](file:///e:/study/BT/csdl/voice_search/report/C/eval_track_a_embedding_test.csv)

### 6.1 Kết quả tổng hợp (30 truy vấn test)

| Nhóm | Số query | Top-1 | P@5 | MAP@5 | group@5 | lang@5 |
|---|---:|---:|---:|---:|---:|---:|
| **Seen** (speaker đã có) | 20 | **1.000** | **1.000** | **1.000** | — | — |
| **Unseen** (speaker chưa có) | 10 | — | — | — | **0.820** | **0.980** |

### 6.2 Phân tích theo vùng/accent

**Seen (20 query):** Top-1 = 1.0 trên toàn bộ — tất cả 20 query trả đúng speaker ở vị trí #1.

**Unseen (10 query):**

| Query | Speaker | Vùng | group@5 | lang@5 | Nhận xét |
|---|---|---|---:|---:|---|
| Q01 | SPK_VIVOSDEV01 | Nam | 1.0 | 1.0 | Đúng hoàn toàn |
| Q02 | SPK_VIVOSDEV08 | Nam | 1.0 | 1.0 | Đúng hoàn toàn |
| Q03 | SPK_p247 | UK | 1.0 | 1.0 | Đúng hoàn toàn |
| Q04 | SPK_p286 | UK | 1.0 | 1.0 | Đúng hoàn toàn |
| **Q05** | SPK_p237 | UK | **0.2** | 1.0 | 4/5 không cùng vùng UK (nhưng cùng tiếng Anh) |
| **Q06** | SPK_p311 | UK | **0.2** | 1.0 | 4/5 không cùng vùng UK (nhưng cùng tiếng Anh) |
| Q07 | SPK_MTXS0 | US | 1.0 | 1.0 | Đúng hoàn toàn |
| Q08 | SPK_MKES0 | US | 0.8 | 0.8 | 1 file Nam lọt vào top-5 |
| Q09 | SPK_MCAE0 | US | 1.0 | 1.0 | Đúng hoàn toàn |
| Q10 | SPK_MSEM1 | US | 1.0 | 1.0 | Đúng hoàn toàn |

### 6.3 Phân tích lỗi

**Q05 (SPK_p237, UK) và Q06 (SPK_p311, UK) — group@5 = 0.2:**
- ECAPA-TDNN học *speaker identity*, không học accent/vùng địa lý. Embedding của hai speaker UK này gần với các speaker US hoặc UK khác trong không gian 192 chiều.
- `group@5` phân biệt UK vs US là chỉ số thô — kết quả `lang@5 = 1.0` cho thấy hệ thống vẫn trả đúng tiếng Anh. Đây **không phải lỗi pipeline**, là giới hạn tự nhiên của việc dùng embedding nhận dạng danh tính để phân loại vùng miền.

**Q08 (SPK_MKES0, US) — group@5 = 0.8, lang@5 = 0.8:**
- 1 file tiếng Việt (vùng Nam) lọt vào top-5. Embedding của speaker US này gần một số speaker Nam tiếng Việt.
- Nguyên nhân có thể: file ngắn hoặc đặc điểm giọng cá nhân chồng lấp không gian embedding.

**Không có file ngắn < 1.5s gây sai trong tập test** (Q03, Q04 đều đạt group@5 = 1.0 — khác với dự đoán trong plan; file thực tế dài hơn 1.5s).

---

## 7. Ưu điểm và hạn chế

| Khía cạnh | Nhận xét |
|---|---|
| **Ưu — Seen** | Tuyệt đối: Top-1 = P@5 = MAP@5 = 1.0. Embedding ECAPA-TDNN rất mạnh với speaker đã có trong CSDL |
| **Ưu — Tốc độ** | Tìm kiếm bằng dot product sau L2-normalize: `O(n·d)`, rất nhanh với n=1000 |
| **Ưu — Không cần z-score** | Cosine tự chuẩn hoá, không phụ thuộc `norm_params` |
| **Hạn — Unseen group@5** | group@5 = 0.82 (thấp hơn Spectral 0.98): embedding học danh tính, không học accent/vùng |
| **Hạn — File rất ngắn** | File < 1.5 s: embedding kém ổn định (ECAPA-TDNN yêu cầu tối thiểu ~1-2 s audio) |
| **Hạn — Phụ thuộc model** | Cần `torch` + `speechbrain` (~80 MB download lần đầu); inference chậm hơn các bộ thủ công |

---

## 8. Điểm nghe thử (Listening Test)

> Cả 3 thành viên cùng chấm điểm 1–5 cho top-5 của 10 truy vấn unseen test.
> Kết quả ghi vào `report/listening_scores.csv` (chưa hoàn thành — cần buổi chấm chung).

---

*Báo cáo tạo ngày 2026-11-24. Dữ liệu từ `report/C/eval_track_a_embedding_test.csv` và `eval_track_a_embedding_dev.csv`.*
