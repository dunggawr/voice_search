"""T6 - Tab Embedding (Tuyen C) cho Streamlit app.

Chuc nang:
  1. Upload file WAV moi -> preprocess -> extract (ECAPA-TDNN) -> cosine search top-5
  2. Hien thi bang ket qua: file_id, speaker, region, similarity
  3. Nghe lai file truy van + tung ket qua (st.audio)
  4. Bieu do bar chart similarity

Cach dung: goi render() tu app/main.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile, os

import streamlit as st
from common import db
from tracks.track_a_embedding import load_matrix, search, extract

# ---------------------------------------------------------------------------
# Cache ket noi va ma tran (chi tai 1 lan)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang tải ma trận embedding từ CSDL...")
def _load_data():
    conn = db.connect()
    ids, M = load_matrix(conn)
    meta = db.read_df(conn, "SELECT file_id, speaker_id, region_accent, language FROM audio_files")
    conn.close()
    return ids, M, meta.set_index("file_id")

def render():
    st.header("🔊 Tìm kiếm theo giọng nói — Embedding ECAPA-TDNN (192 chiều)")
    st.markdown(
        "Tải lên 1 file WAV (giọng đọc mới). Hệ thống trích **speaker embedding** "
        "bằng ECAPA-TDNN rồi tìm **top 5 file giống nhất** trong CSDL 1.000 file."
    )

    ids, M, meta = _load_data()
    st.caption(f"CSDL: {len(ids)} file | {meta['speaker_id'].nunique()} người nói")

    # -----------------------------------------------------------------------
    # Upload
    # -----------------------------------------------------------------------
    uploaded = st.file_uploader("Chọn file âm thanh", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded is None:
        st.info("Hãy tải lên 1 file âm thanh để bắt đầu tìm kiếm.")
        return

    # Luu tam ra disk de preprocess doc duoc
    suffix = "." + uploaded.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    st.audio(tmp_path, format="audio/wav")
    st.caption(f"File truy vấn: `{uploaded.name}`")

    with st.spinner("Đang trích embedding..."):
        try:
            qvec = extract(tmp_path)
        except Exception as e:
            st.error(f"Lỗi trích đặc trưng: {e}")
            os.unlink(tmp_path)
            return

    st.success(f"Embedding hợp lệ: 192 chiều | chuẩn={np.linalg.norm(qvec):.4f}")

    # -----------------------------------------------------------------------
    # Tim kiem
    # -----------------------------------------------------------------------
    k = st.slider("Số kết quả trả về (k)", min_value=1, max_value=10, value=5)
    results = search(qvec, ids, M, k=k)

    rows = []
    for rank, (fid, sim, dist) in enumerate(results, 1):
        spk = meta.loc[fid, "speaker_id"]  if fid in meta.index else "?"
        reg = meta.loc[fid, "region_accent"] if fid in meta.index else "?"
        lng = meta.loc[fid, "language"]    if fid in meta.index else "?"
        rows.append({"Hạng": rank, "file_id": fid, "speaker_id": spk,
                     "Vùng/Accent": reg, "Ngôn ngữ": lng,
                     "Độ tương đồng": round(sim, 4), "Khoảng cách": round(dist, 4)})

    df_res = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Bang ket qua
    # -----------------------------------------------------------------------
    st.subheader("Kết quả Top-k")
    st.dataframe(df_res, use_container_width=True)

    # -----------------------------------------------------------------------
    # Bieu do bar chart
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(df_res["file_id"][::-1], df_res["Độ tương đồng"][::-1], color="#3498db")
    ax.set_xlabel("Độ tương đồng Cosine [0, 1]")
    ax.set_title("Độ tương đồng Top-k")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, df_res["Độ tương đồng"][::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -----------------------------------------------------------------------
    # Nghe lai tung ket qua (neu co file wav tren disk)
    # -----------------------------------------------------------------------
    st.subheader("Nghe lại kết quả")
    for _, row in df_res.iterrows():
        wav_path = os.path.join("data", "wav", f"{row['file_id']}.wav")
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(f"**#{row['Hạng']} {row['file_id']}**  \n"
                        f"Người nói: `{row['speaker_id']}`  Vùng: `{row['Vùng/Accent']}`  "
                        f"Điểm: `{row['Độ tương đồng']}`")
        with col2:
            if os.path.exists(wav_path):
                st.audio(wav_path, format="audio/wav")
            else:
                st.caption("(Không tìm thấy file WAV trên đĩa — giải nén data/wav/ để nghe)")

    os.unlink(tmp_path)
