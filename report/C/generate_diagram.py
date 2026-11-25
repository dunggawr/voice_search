import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(15, 10.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 10.5)
ax.axis("off")

PAD = 0.2

def add_box(x, y, w, h, fc, ec, lw=1.8):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={PAD},rounding_size=0.25",
        fc=fc, ec=ec, lw=lw,
    )
    ax.add_patch(box)

# 1. Tang nhap lieu
add_box(0.5, 8.2, 3.0, 1.4, fc="#E3F2FD", ec="#1976D2", lw=2)
ax.text(
    2.0, 8.9,
    "1. File Truy Vấn (WAV Mới)\n- Tải lên từ giao diện\n- File tự thu / nguồn ngoài",
    ha="center", va="center", fontsize=10, weight="bold", color="#0D47A1",
)

# 2. Tang tien xu ly
add_box(4.5, 8.2, 3.4, 1.4, fc="#E8F5E9", ec="#388E3C", lw=2)
ax.text(
    6.2, 8.9,
    "2. Tiền Xử Lý (common.preprocess)\n- 16 kHz Mono\n- Cắt lặng (top_db=25)\n- Chuẩn hóa -23 LUFS",
    ha="center", va="center", fontsize=9.5, weight="bold", color="#1B5E20",
)

# 3. Tang trich rut dac trung (song song)
add_box(9.0, 8.7, 5.4, 1.0, fc="#EDE7F6", ec="#5E35B1", lw=1.8)
ax.text(
    11.7, 9.2,
    "Speaker Embedding — track_a_embedding.py\nECAPA-TDNN (192 chiều) | L2 Normalize (Cosine)",
    ha="center", va="center", fontsize=9.2, weight="bold", color="#311B92",
)

add_box(9.0, 7.0, 5.4, 1.0, fc="#FFF3E0", ec="#E65100", lw=1.8)
ax.text(
    11.7, 7.5,
    "Đặc trưng Phổ (Spectral) — track_c_spectral.py\n46 đặc trưng phổ & MFCC | z-score + Cosine",
    ha="center", va="center", fontsize=9.2, weight="bold", color="#BF360C",
)

add_box(9.0, 5.3, 5.4, 1.0, fc="#FCE4EC", ec="#C2185B", lw=1.8)
ax.text(
    11.7, 5.8,
    "Đặc trưng Prosodic — track_b_prosodic.py\n11 đặc trưng F0/Energy/Quality | z-score + Mahalanobis",
    ha="center", va="center", fontsize=9.2, weight="bold", color="#880E4F",
)

# 4. Tang co so du lieu PostgreSQL
add_box(0.5, 3.7, 7.4, 3.1, fc="#ECEFF1", ec="#455A64", lw=2)
ax.text(
    4.2, 6.3,
    "CƠ SỞ DỮ LIỆU POSTGRESQL 16 (Schema 'voice')",
    ha="center", va="center", fontsize=11, weight="bold", color="#263238",
)
ax.text(
    4.2, 4.9,
    "• speakers (84) & audio_files (1.000)\n"
    "• feat_embedding (192 chiều REAL[])\n"
    "• feat_spectral (46 cột DOUBLE PRECISION)\n"
    "• feat_prosodic (11 cột DOUBLE PRECISION)\n"
    "• norm_params (57 cặp mean, std dùng cho z-score)",
    ha="center", va="center", fontsize=9.5, color="#37474F", linespacing=1.4,
)

# 5. Tang tim kiem & hop nhat
add_box(9.0, 2.6, 5.4, 1.8, fc="#FFF9C4", ec="#FBC02D", lw=2)
ax.text(
    11.7, 4.0,
    "HỆ THỐNG TÌM KIẾM & HỢP NHẤT\ntracks.fusion.search_fused()",
    ha="center", va="center", fontsize=10, weight="bold", color="#E65100",
)
ax.text(
    11.7, 3.1,
    "sim_fused = 0.80·sim_emb + 0.15·sim_spec + 0.05·sim_pros\n"
    "dist_fused = 1.0 - sim_fused  |  Tối ưu trọng số trên tập dev",
    ha="center", va="center", fontsize=9, color="#573A08", linespacing=1.35,
)

# 6. Tang giao dien Streamlit
add_box(0.5, 0.5, 13.9, 1.3, fc="#E0F7FA", ec="#00838F", lw=2)
ax.text(
    7.45, 1.15,
    "ỨNG DỤNG DEMO STREAMLIT (app/main.py)\n"
    "Tab 1: Embedding  |  Tab 2: Spectral  |  Tab 3: Prosodic  |  Tab 4: Hợp nhất\n"
    "Top-5 Kết Quả: File ID  |  Speaker  |  Vùng/Accent  |  Điểm Tương Đồng  |  Nghe Audio  |  Biểu đồ",
    ha="center", va="center", fontsize=10, weight="bold", color="#006064", linespacing=1.35,
)

# Mui ten noi dung mep ngoai cac khoi
arrow = dict(arrowstyle="->", lw=1.8, color="#37474F")
# 1 -> 2
ax.annotate("", xy=(4.3, 8.9), xytext=(3.7, 8.9), arrowprops=arrow)
# 2 -> 3
ax.annotate("", xy=(8.8, 9.2), xytext=(8.1, 9.1), arrowprops=arrow)
ax.annotate("", xy=(8.8, 7.5), xytext=(8.1, 8.9), arrowprops=arrow)
ax.annotate("", xy=(8.8, 5.8), xytext=(8.1, 8.6), arrowprops=arrow)
# DB <-> Fusion
ax.annotate("", xy=(8.8, 3.7), xytext=(8.1, 4.5), arrowprops=dict(arrowstyle="<->", lw=1.8, color="#455A64"))
# 3 -> Fusion
ax.annotate("", xy=(11.7, 4.6), xytext=(11.7, 5.1), arrowprops=arrow)
# Fusion -> UI
ax.annotate("", xy=(11.7, 2.0), xytext=(11.7, 2.4), arrowprops=arrow)
# DB -> UI
ax.annotate("", xy=(4.2, 2.0), xytext=(4.2, 3.5), arrowprops=arrow)

plt.tight_layout()
plt.savefig("report/block_diagram.png", dpi=300, bbox_inches="tight")
print("Saved report/block_diagram.png")
