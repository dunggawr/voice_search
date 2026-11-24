import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# Styles
box_style = dict(boxstyle="round,pad=0.5", linewidth=1.5)

# 1. Input Layer
p1 = patches.FancyBboxPatch((0.5, 7.8), 2.8, 1.4, boxstyle="round,pad=0.4", fc="#E3F2FD", ec="#1976D2", lw=2)
ax.add_patch(p1)
ax.text(1.9, 8.6, "1. File Truy Vấn (WAV Mới)\n- Upload Streamlit\n- File tự thu điện thoại", 
        ha="center", va="center", fontsize=10, weight="bold", color="#0D47A1")

# 2. Preprocess Layer
p2 = patches.FancyBboxPatch((4.2, 7.8), 3.0, 1.4, boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#388E3C", lw=2)
ax.add_patch(p2)
ax.text(5.7, 8.6, "2. Tiền Xử Lý (common.preprocess)\n- 16 kHz Mono\n- Cắt lặng (top_db=25)\n- Chuẩn hóa -23 LUFS", 
        ha="center", va="center", fontsize=9.5, weight="bold", color="#1B5E20")

# 3. Feature Extraction Tracks (Parallel)
# Track C (Embedding)
p3c = patches.FancyBboxPatch((8.2, 8.4), 5.2, 1.2, boxstyle="round,pad=0.4", fc="#EDE7F6", ec="#5E35B1", lw=1.8)
ax.add_patch(p3c)
ax.text(10.8, 9.0, "Tuyến C (Embedding) — track_a_embedding.py\nECAPA-TDNN (192-d) | L2 Normalize (Cosine)", 
        ha="center", va="center", fontsize=9, weight="bold", color="#311B92")

# Track A (Spectral)
p3a = patches.FancyBboxPatch((8.2, 6.9), 5.2, 1.2, boxstyle="round,pad=0.4", fc="#FFF3E0", ec="#E65100", lw=1.8)
ax.add_patch(p3a)
ax.text(10.8, 7.5, "Tuyến A (Spectral) — track_c_spectral.py\n46 đặc trưng phổ & MFCC | z-score chuẩn hóa", 
        ha="center", va="center", fontsize=9, weight="bold", color="#BF360C")

# Track B (Prosodic)
p3b = patches.FancyBboxPatch((8.2, 5.4), 5.2, 1.2, boxstyle="round,pad=0.4", fc="#FCE4EC", ec="#C2185B", lw=1.8)
ax.add_patch(p3b)
ax.text(10.8, 6.0, "Tuyến B (Prosodic) — track_b_prosodic.py\n11 đặc trưng F0/Energy/Quality | Mahalanobis / z-score", 
        ha="center", va="center", fontsize=9, weight="bold", color="#880E4F")

# 4. Database Layer (PostgreSQL)
p_db = patches.FancyBboxPatch((0.5, 4.0), 6.7, 2.8, boxstyle="round,pad=0.5", fc="#ECEFF1", ec="#455A64", lw=2)
ax.add_patch(p_db)
ax.text(3.85, 6.4, "CƠ SỞ DỮ LIỆU POSTGRESQL 16 (Schema 'voice')", 
        ha="center", va="center", fontsize=11, weight="bold", color="#263238")
ax.text(3.85, 5.2, "• metadata_master: speakers (84) & audio_files (1000)\n• feat_embedding (192-d REAL[]) — C\n• feat_spectral (46 cột float) — A\n• feat_prosodic (11 cột float) — B\n• norm_params: (mean, std) dùng cho z-score", 
        ha="center", va="center", fontsize=9, color="#37474F")

# 5. Search & Fusion Layer
p_fusion = patches.FancyBboxPatch((8.2, 2.5), 5.2, 2.2, boxstyle="round,pad=0.5", fc="#FFF9C4", ec="#FBC02D", lw=2)
ax.add_patch(p_fusion)
ax.text(10.8, 4.2, "HỆ THỐNG TÌM KIẾM & HỢP NHẤT (2B)\ntracks.fusion.search_fused()", 
        ha="center", va="center", fontsize=10, weight="bold", color="#F57F17")
ax.text(10.8, 3.2, "sim_fused = 0.80·sim_emb + 0.15·sim_spec + 0.05·sim_pros\ndist_fused = 1.0 - sim_fused\n(Tối ưu trọng số trên tập dev)", 
        ha="center", va="center", fontsize=9, color="#573A08")

# 6. Presentation / Demo App Layer
p_ui = patches.FancyBboxPatch((1.5, 0.6), 11.0, 1.4, boxstyle="round,pad=0.4", fc="#E0F7FA", ec="#00838F", lw=2)
ax.add_patch(p_ui)
ax.text(7.0, 1.3, "ỨNG DỤNG DEMO STREAMLIT (app/main.py)\nTab 1: Embedding (C) | Tab 2: Spectral (A) | Tab 3: Prosodic (B) | Tab 4: Fused (A+B+C)\nTop-5 Kết Quả: File ID | Speaker | Vùng/Accent | Điểm Tương Đồng | Nghe Audio | Bar Chart", 
        ha="center", va="center", fontsize=10, weight="bold", color="#006064")

# Arrows
arrow = dict(arrowstyle="->", lw=1.8, color="#37474F")
# 1 -> 2
ax.annotate("", xy=(4.2, 8.5), xytext=(3.3, 8.5), arrowprops=arrow)
# 2 -> 3
ax.annotate("", xy=(8.2, 9.0), xytext=(7.2, 8.8), arrowprops=arrow)
ax.annotate("", xy=(8.2, 7.5), xytext=(7.2, 8.5), arrowprops=arrow)
ax.annotate("", xy=(8.2, 6.0), xytext=(7.2, 8.2), arrowprops=arrow)

# DB -> 3/Search
ax.annotate("", xy=(8.2, 3.6), xytext=(7.2, 4.8), arrowprops=dict(arrowstyle="<->", lw=1.8, color="#455A64"))
# 3 -> Fusion
ax.annotate("", xy=(10.8, 4.7), xytext=(10.8, 5.4), arrowprops=arrow)
# Fusion -> UI
ax.annotate("", xy=(10.8, 2.0), xytext=(10.8, 2.5), arrowprops=arrow)
# Direct Track -> UI (for single tab demo)
ax.annotate("", xy=(3.0, 2.0), xytext=(3.0, 4.0), arrowprops=arrow)

plt.tight_layout()
plt.savefig("report/block_diagram.png", dpi=300, bbox_inches="tight")
print("Saved report/block_diagram.png")
