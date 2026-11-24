# Đánh giá hệ thống tìm kiếm giọng nói

Chạy `python -m tracks.fusion --from-db` để đọc vector từ PostgreSQL.
Trọng số được chọn trên tập dev gồm 5% Prosodic, 15% Spectral và
80% Embedding. Bảng dưới là kết quả trên tập test gồm 20 truy vấn
seen và 10 truy vấn unseen.

| Phương pháp | Seen Top-1 | Seen P@5 | Seen MAP@5 | Unseen cùng vùng | Unseen cùng ngôn ngữ |
|---|---:|---:|---:|---:|---:|
| Prosodic | 25% | 18% | 16,70% | 80% | 94% |
| Spectral | 60% | 43% | 36,75% | 98% | 98% |
| Embedding | 100% | 100% | 100% | 82% | 98% |
| Fusion | 100% | 100% | 100% | 96% | 100% |

Trên tập test, Fusion giữ mức 100% ở cả ba chỉ số tìm đúng người nói
của nhóm seen, ngang với Embedding. Ở nhóm unseen, Fusion tăng tỷ lệ
kết quả cùng vùng từ 82% lên 96% so với Embedding. Spectral đạt tỷ lệ
cùng vùng cao nhất là 98%. Với nhóm unseen, toàn bộ file của người nói
truy vấn đã bị loại, nên hai chỉ số chỉ đo mức khớp vùng và ngôn ngữ.