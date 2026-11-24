# Kịch bản demo hệ thống tìm kiếm giọng nói

1. Mở app bằng `python -m streamlit run app/main.py`.
2. Ở tab Spectral và Prosodic, tải `new_A_1.wav`, mở bảng top 5
   và phát thử bản thu cùng một kết quả.
3. Ở tab Embedding, thử cùng bản WAV trên máy chạy được PyTorch.
   Máy Đạt đã đối chiếu top 5 trên 50/50 truy vấn từ vector có sẵn.
4. Ở tab Fused, chọn mẫu `A_0001`, xem top 5 và phát kết quả
   hạng 1 `A_0005`. Tab hiện dùng vector có sẵn của file mẫu.
5. Mở bảng đánh giá tập test. Fusion đạt Top-1 100% trên nhóm seen
   và tăng tỷ lệ cùng vùng của nhóm unseen từ 82% lên 96% so với Embedding.