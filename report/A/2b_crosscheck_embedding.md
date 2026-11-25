Đối chiếu từ vector có sẵn: top 5 khớp 50/50 truy vấn với kết quả của tuyến C. Kiểm thử bằng WAV mới trên máy Đạt đang bị chặn ở bước nạp PyTorch (WinError 4551).
# Kiểm tra chéo tuyến Embedding

Top 5 khớp kết quả tuyến C trên 50/50 truy vấn, gồm 20 truy vấn dev
và 30 truy vấn test. Phép đối chiếu dùng vector có sẵn trong CSV.

Đã sửa lỗi import trong app chung. Kiểm thử trích embedding từ WAV mới
trên máy Đạt còn chờ vì Windows Application Control chặn PyTorch
tải DLL với WinError 4551.
