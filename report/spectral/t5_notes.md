# Đánh giá Spectral trên tập test

Tuyến Spectral dùng cosine sau chuẩn hóa z-score. Trên 20 truy vấn seen,
12/20 truy vấn có kết quả hạng 1 cùng người nói. P@5 đạt 43% và MAP@5
đạt 36,7%. Trên 10 truy vấn unseen, group@5 và lang@5 cùng đạt 98%.

Các truy vấn seen sai ở hạng 1 gồm Q25, Q27, Q28, Q29, Q33, Q34, Q37
và Q40. Với Q25, cả 5 kết quả cùng vùng UK nhưng khác người nói. Với
Q37, hạng 1 sai nhưng 3/5 kết quả vẫn cùng người nói. Các trường hợp này
cho thấy những người cùng vùng có thể có đặc trưng phổ gần nhau.

Ở nhóm unseen, Q02 đưa A_0560 (UK, tiếng Anh) lên hạng 1 cho truy vấn
A_0930 (giọng Nam, tiếng Việt). Bốn kết quả còn lại cùng vùng và ngôn
ngữ với truy vấn. Các chỉ số unseen ở đây đo mức khớp vùng và ngôn ngữ.

Tuyến Spectral xử lý được cả hai bản thu mới từ iPhone. Tuy nhiên, top 5 chủ yếu trả về giọng tiếng Anh dù đầu vào là tiếng Việt. Với hai mẫu này, khả năng tìm giọng cùng vùng và ngôn ngữ thấp hơn kết quả quan sát được trên tập test.
## Nghe thử

Đạt đã chấm đủ 10 truy vấn. Điểm trung bình là 3,4/5 cho nhóm seen
và 2,6/5 cho nhóm unseen. Điểm của Dũng và Tùng đang chờ bổ sung.s