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
@st.cache_resource(show_spinner="Dang tai ma tran embedding tu CSDL...")
def _load_data():
    conn = db.connect()
    ids, M = load_matrix(conn)
    meta = db.read_df(conn, "SELECT file_id, speaker_id, region_accent, language FROM audio_files")
    conn.close()
    return ids, M, meta.set_index("file_id")

def render():
    st.header("🔊 Tim kiem theo giong noi — Embedding ECAPA-TDNN (192 chieu)")
    st.markdown(
        "Upload 1 file WAV (giong doc moi). He thong trich **speaker embedding** "
        "bang ECAPA-TDNN roi tim **top-5 file giong nhat** trong CSDL 1000 file."
    )

    ids, M, meta = _load_data()
    st.caption(f"CSDL: {len(ids)} file | {meta['speaker_id'].nunique()} speaker")

    # -----------------------------------------------------------------------
    # Upload
    # -----------------------------------------------------------------------
    uploaded = st.file_uploader("Chon file WAV", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded is None:
        st.info("Hay upload 1 file am thanh de bat dau tim kiem.")
        return

    # Luu tam ra disk de preprocess doc duoc
    suffix = "." + uploaded.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    st.audio(tmp_path, format="audio/wav")
    st.caption(f"File truy van: `{uploaded.name}`")

    with st.spinner("Dang trich embedding..."):
        try:
            qvec = extract(tmp_path)
        except Exception as e:
            st.error(f"Loi trich dac trung: {e}")
            os.unlink(tmp_path)
            return

    st.success(f"Embedding OK: 192 chieu | norm={np.linalg.norm(qvec):.4f}")

    # -----------------------------------------------------------------------
    # Tim kiem
    # -----------------------------------------------------------------------
    k = st.slider("So ket qua tra ve (k)", min_value=1, max_value=10, value=5)
    results = search(qvec, ids, M, k=k)

    rows = []
    for rank, (fid, sim, dist) in enumerate(results, 1):
        spk = meta.loc[fid, "speaker_id"]  if fid in meta.index else "?"
        reg = meta.loc[fid, "region_accent"] if fid in meta.index else "?"
        lng = meta.loc[fid, "language"]    if fid in meta.index else "?"
        rows.append({"Hang": rank, "file_id": fid, "speaker_id": spk,
                     "region": reg, "language": lng,
                     "similarity": round(sim, 4), "distance": round(dist, 4)})

    df_res = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Bang ket qua
    # -----------------------------------------------------------------------
    st.subheader("Ket qua top-k")
    st.dataframe(df_res, use_container_width=True)

    # -----------------------------------------------------------------------
    # Bieu do bar chart
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(df_res["file_id"][::-1], df_res["similarity"][::-1], color="#3498db")
    ax.set_xlabel("Cosine Similarity [0, 1]")
    ax.set_title("Top-k similarity")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, df_res["similarity"][::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # -----------------------------------------------------------------------
    # Nghe lai tung ket qua (neu co file wav tren disk)
    # -----------------------------------------------------------------------
    st.subheader("Nghe lai ket qua")
    for _, row in df_res.iterrows():
        wav_path = os.path.join("data", "wav", f"{row['file_id']}.wav")
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(f"**#{row['Hang']} {row['file_id']}**  \n"
                        f"spk: `{row['speaker_id']}`  region: `{row['region']}`  "
                        f"sim: `{row['similarity']}`")
        with col2:
            if os.path.exists(wav_path):
                st.audio(wav_path, format="audio/wav")
            else:
                st.caption("(File WAV khong co tren disk — giai nen data/wav/ de nghe)")

    os.unlink(tmp_path)
