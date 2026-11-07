# Hệ CSDL tìm kiếm giọng nói đàn ông: gói Bước 0 (giai đoạn 2)

Gói này là "hợp đồng chung" của nhóm cho Yêu cầu 3, 4, 5. **Không tự đổi tên bảng, tên cột, tên hàm, tên file** nếu chưa báo cả nhóm; đổi một chỗ là 3 tuyến không ghép được.

## 1. Cấu trúc thư mục

```
voice_search/
  schema.sql               hợp đồng CSDL (SQLite)
  init_db.py               tạo db/voice.db + nạp bảng speakers, audio_files
  make_query_set.py        đã chạy sẵn, ra data/query_set.csv (không chạy lại)
  common/
    preprocess.py          chuẩn hoá FILE MỚI (16 kHz mono, cắt lặng top_db=25, -23 LUFS)
    db.py                  đọc/ghi voice.db (load_matrix, norm_params, lưu embedding)
    evaluate.py            khung chấm điểm chung (top1, P@5, MAP@5, group@5, lang@5)
  tracks/
    _template.py           mẫu module cho mỗi tuyến
  data/
    metadata_master.csv    metadata 1000 file của A
    query_set.csv          50 truy vấn cố định (30 test, 20 dev)
    wav/                   (tự giải nén 1000 file A_0001.wav ... A_1000.wav vào đây; không đưa lên git)
    new_recordings/        file mới tự thu bằng điện thoại (xem README trong thư mục)
  db/voice.db              (sinh ra bởi init_db.py; không đưa lên git)
  app/                     ứng dụng demo (Streamlit)
  report/                  bảng và hình cho báo cáo
```

## 2. Cài đặt và dựng CSDL (mỗi người làm 1 lần)

```
pip install -r requirements.txt
# giải nén 1000 file wav vào data/wav/
python init_db.py data/metadata_master.csv data/wav
```

Sau lệnh trên `db/voice.db` có 1000 file, 84 speaker; các bảng `feat_*` và `norm_params` còn trống: mỗi người tự nạp bảng của mình ở việc T2.

## 3. Phân tuyến

| Người | Bộ đặc trưng | Bảng ghi vào | Module |
|---|---|---|---|
| A | Phổ (46 cột) | `feat_spectral` | `tracks/track_c_spectral.py` |
| B | Prosodic (11 cột) | `feat_prosodic` | `tracks/track_b_prosodic.py` |
| C | Speaker embedding (192 chiều, ECAPA-TDNN) | `feat_embedding` | `tracks/track_a_embedding.py` |

Mỗi người chỉ ghi vào bảng của mình. Dữ liệu giai đoạn 1 nạp nguyên từ CSV đã nộp (`features_embedding.csv`, `features_prosodic.csv`, `features_spectral.csv`), cột khớp tên trong `schema.sql`.

## 4. Hợp đồng

**CSDL.** Xem `schema.sql`. Embedding lưu `float32` little-endian dạng BLOB (192 × 4 byte) bằng `db.save_embedding`. `norm_params` lưu trung bình và độ lệch chuẩn từng cột tính trên 1000 file (`db.save_norm_params`); tìm kiếm z-score bằng đúng các tham số này.

**Mỗi module tuyến có 3 hàm** (xem `tracks/_template.py`):

| Hàm | Vào | Ra |
|---|---|---|
| `extract(wav_path)` | đường dẫn file mới | vector đặc trưng 1 chiều (`np.ndarray`), sau khi qua `preprocess()` |
| `load_matrix(conn)` | kết nối CSDL | `(ids, M)`; có thể bỏ nếu dùng `db.load_matrix` mặc định |
| `search(qvec, ids, M, k=5)` | vector truy vấn, danh sách file và ma trận cần tìm | `[(file_id, similarity, distance), ...]`, đúng k phần tử, **sắp xếp giảm dần theo similarity** |

`similarity` nằm trong [0, 1], càng lớn càng giống (để hợp nhất điểm ở giai đoạn 2B). `search` không được trả về chính file truy vấn.

**File mới.** Mọi file chưa nằm trong 1000 file (file demo, file tự thu) phải qua `common.preprocess.preprocess()` trước khi trích đặc trưng. 1000 file cũ đã chuẩn hoá, không cần chạy lại. Hàm tái tạo đúng quy trình của A trên khoảng 83% file; một số file cũ lệch vài chục mili giây ở khâu cắt lặng, nên khi kiểm tra `extract()` trên file cũ hãy đọc thẳng file trong `data/wav/` và cho phép sai số nhỏ.

**Bộ truy vấn và cách chấm** (`data/query_set.csv`, `common/evaluate.py`):

| Loại | Số lượng | Cách tìm | Chỉ số |
|---|---|---|---|
| seen | 20 test + 10 dev | loại đúng file truy vấn khỏi CSDL | top1, P@5, MAP@5 |
| unseen | 10 test + 10 dev | loại **toàn bộ** file của speaker đó | group@5, lang@5, và nghe thử |

- `dev`: dùng để chọn độ đo, chọn trọng số hợp nhất. `test`: chỉ dùng để báo cáo, **không chỉnh tham số theo test**.
- Chạy: `python -m common.evaluate tracks.track_b_prosodic --split test`
- Điểm nghe thử: cả 3 người cùng chấm 1 đến 5 cho top-5 của 10 truy vấn, ghi vào `report/`.

**Lưu ý khi đọc kết quả.**
- P@5 bị chặn trên bởi số file còn lại của speaker: speaker UK chỉ có 3 đến 8 file, nên tối đa P@5 có thể dưới 1 dù tìm đúng hoàn toàn. Hãy xem top1 và MAP@5 là chính.
- group@5 (cùng vùng/accent) và lang@5 chỉ là kiểm tra thô cho speaker chưa có. Embedding học danh tính giọng, không học accent, nên group@5 giữa UK và US có thể thấp mà vẫn tốt. Đánh giá chính cho nhóm này là nghe thử.
- Mỗi nhóm truy vấn chỉ 10 đến 20 câu, nên số liệu dao động mạnh; báo cáo kèm số lượng.

**Git.** Mỗi người một nhánh (`track-a`, `track-b`, `track-c`), gộp vào `main` qua pull request để người khác xem. Không commit `data/wav/` và `db/voice.db`.

## 5. Việc của mỗi tuyến (T1 đến T6)

- [ ] **T1** `extract(wav_path)` cho bộ của mình; test trên 20 file cũ.
- [ ] **T2** Nạp bảng `feat_*` của mình, lưu `norm_params`, `load_matrix`; thử vài câu SQL (lấy đặc trưng một file, đếm file theo speaker).
- [ ] **T3** `search()` top-5; thử các độ đo trên `dev` (Euclid, cosine, Mahalanobis) và chọn một.
- [ ] **T4** Minh hoạ 2 truy vấn (một seen, một unseen): vector đặc trưng, sau chuẩn hoá, khoảng cách, bảng xếp hạng, biểu đồ.
- [ ] **T5** Chạy `common.evaluate` trên `test`, phân tích các câu sai, chấm nghe thử.
- [ ] **T6** Một tab trong `app/` và mục báo cáo cho bộ của mình.

## 6. Giai đoạn 2B (sau khi xong T1 đến T6)

- A: sơ đồ khối, quy trình truy vấn, ghép 3 bảng, mục báo cáo về CSDL. Kiểm tra tuyến B.
- B: hợp nhất điểm 3 bộ (`search_fused`), chọn trọng số **trên `dev`**. Kiểm tra tuyến C.
- C: đánh giá tổng, ứng dụng demo hoàn chỉnh, kịch bản demo. Kiểm tra tuyến A.
- Kiểm tra chéo: chạy code người kia từ đầu trên máy mình, so với báo cáo, ghi lỗi.
