# File mới tự thu (kiểm tra kênh thu khác)

Mỗi người thu 2 file bằng điện thoại, tổng 6 file (cần tối thiểu 5), để thử hệ thống với file thật chưa từng nằm trong 1000 file.

- Người nói: nam. Mỗi file chỉ một người nói, 2 đến 6 giây, câu tự chọn.
- Thu ở phòng yên tĩnh, để điện thoại cách miệng khoảng 20 cm. Định dạng gốc của điện thoại (m4a/mp3/wav) đều được.
- Đặt tên: `new_<người thu>_<số>.<đuôi>`, ví dụ `new_B_1.m4a`, kèm một dòng trong `new_recordings.csv`: `file,speaker_note,thiet_bi`.
- **Chỉ thu giọng của người đã đồng ý** cho dùng giọng trong bài tập. Không dùng file của người khác khi chưa hỏi.
- Chạy qua `python -m common.preprocess <file_vào> <file_ra.wav>` rồi mới trích đặc trưng.
- Không đưa file thu lên git nếu có ghi tên thật; giữ trong thư mục này trên máy từng người.
