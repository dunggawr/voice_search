"""App demo Voice Search - Streamlit.

Cau truc: 4 tab
  Tab 1 (C): Embedding ECAPA-TDNN
  Tab 2 (A): Spectral (46 chieu)
  Tab 3 (B): Prosodic (11 chieu)
  Tab 4   : Fused (Hop nhat 3 bo dac trung)

Chay: streamlit run app/main.py
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="Tìm kiếm giọng nói",
    page_icon="🔊",
    layout="wide",
)


@st.cache_resource
def _try_connect():
    try:
        from common import db
        return db.connect()
    except Exception:
        return None


conn = _try_connect()

tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Embedding",
    "📊 Phổ (Spectral)",
    "🎵 Prosodic",
    "🔀 Hợp nhất",
])

with tab1:
    from psycopg2 import Error as DatabaseError

    try:
        from app.tab_embedding import render as render_embedding

        render_embedding()
    except (ImportError, OSError, RuntimeError, DatabaseError) as exc:
        st.warning(f"Tab Embedding chưa sẵn sàng: {exc}")

with tab2:
    from app.tab_spectral import render as render_spectral

    render_spectral()

with tab3:
    from app.tab_b_prosodic import render_tab as render_prosodic

    render_prosodic(conn)

with tab4:
    from tracks import fusion

    st.header("🔀 Hợp nhất 3 bộ đặc trưng")
    comp_path = os.path.join(ROOT, "report", "B", "eval_fusion_comparison.csv")
    fig_path = os.path.join(ROOT, "report", "B", "fig_2b_fusion_comparison.png")
    if os.path.exists(comp_path):
        st.subheader("So sánh 4 hệ thống trên tập dev & test")
        st.dataframe(pd.read_csv(comp_path), use_container_width=True, hide_index=True)
    if os.path.exists(fig_path):
        st.image(fig_path, caption="So sánh 3 bộ đặc trưng đơn lẻ và hệ thống hợp nhất (tập TEST)")

    st.subheader("Thử tìm kiếm hợp nhất trên tập 1.000 file")
    ids_f, matrices_f, norms_f = fusion.load_all_from_csv()
    meta_f = pd.read_csv(os.path.join(ROOT, "data", "metadata_master.csv")).set_index("file_id")
    sample_id = st.selectbox("Chọn file truy vấn mẫu", ids_f.tolist(), index=0, key="fused_sample_id")
    q_wav = os.path.join(ROOT, "data", "wav", f"{sample_id}.wav")
    if os.path.exists(q_wav):
        st.audio(q_wav)

    q_idx = int(np.where(ids_f == sample_id)[0][0])
    keep = ids_f != sample_id
    qvecs = {fs: matrices_f[fs][q_idx] for fs in fusion.FEATURE_SETS}
    sub_mats = {fs: matrices_f[fs][keep] for fs in fusion.FEATURE_SETS}
    res_f = fusion.search_fused(qvecs, ids_f[keep], sub_mats, k=5, norm_params=norms_f)

    rows = []
    for rank, (fid, sim, dist) in enumerate(res_f, start=1):
        m = meta_f.loc[fid]
        rows.append({
            "Hạng": rank,
            "file_id": fid,
            "speaker_id": m["speaker_id"],
            "Vùng/Accent": m["region_accent"],
            "Ngôn ngữ": m["language"],
            "Độ tương đồng": round(sim, 4),
            "Khoảng cách": round(dist, 4),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for rank, (fid, sim, _) in enumerate(res_f, start=1):
        wp = os.path.join(ROOT, "data", "wav", f"{fid}.wav")
        if os.path.exists(wp):
            st.caption(f"#{rank} {fid} ({meta_f.loc[fid, 'speaker_id']}) — sim={sim:.3f}")
            st.audio(wp)
