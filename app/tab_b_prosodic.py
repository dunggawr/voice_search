"""Tab demo cho Tuyen B (Prosodic & Chat giong - 11 dac trung).

Co the nhung vao app chung o giai doan 2B bang cach goi:
    from app.tab_b_prosodic import render_tab
    render_tab(conn)
Neu `conn=None` (hoac DB chua bat), tab tu dong doc du lieu tu CSV giai doan 1.
"""
import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from tracks import track_b_prosodic as tb


@st.cache_data(show_spinner=False)
def _load_offline_data():
    ids, M = tb.load_matrix_csv("docs/B/features_prosodic.csv")
    meta = pd.read_csv("data/metadata_master.csv").set_index("file_id")
    mean = M.mean(axis=0)
    std = np.where(M.std(axis=0) > 1e-12, M.std(axis=0), 1.0)
    return ids, M, meta, mean, std


def _load_db_data(conn):
    from common import db
    ids, M = tb.load_matrix(conn)
    meta = db.read_df(
        conn,
        "SELECT file_id, speaker_id, region_accent, language, path, duration_sec FROM audio_files",
    ).set_index("file_id")
    try:
        mean, std = db.load_norm_params(conn, tb.FEATURE_SET)
    except Exception:
        mean = M.mean(axis=0)
        std = np.where(M.std(axis=0) > 1e-12, M.std(axis=0), 1.0)
    return ids, M, meta, mean, std


def _resolve_wav_path(file_id, meta):
    if "path" in meta.columns:
        p = meta.loc[file_id, "path"]
        if isinstance(p, str) and os.path.exists(p):
            return p
    fallback = os.path.join("data", "wav", f"{file_id}.wav")
    return fallback if os.path.exists(fallback) else None


def _plot_compare_rank1(q_label, q_z, r1_label, r1_z):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(tb.COLS))
    w = 0.38
    ax.bar(x - w / 2, q_z, width=w, color="#2a78d6", label=f"Truy vấn ({q_label})")
    ax.bar(x + w / 2, r1_z, width=w, color="#eb6834", label=f"Hạng 1 ({r1_label})")
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tb.COLS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("z-score")
    ax.set_title("So sánh vector z-score giữa file truy vấn và kết quả hạng 1", loc="left", fontsize=10.5)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e3e2de", linewidth=0.6)
    fig.tight_layout()
    return fig


def render_tab(conn=None):
    st.subheader("Tìm kiếm theo đặc trưng Prosodic & Chất giọng (11 chiều)")
    if conn is not None:
        try:
            ids, M, meta, mean, std = _load_db_data(conn)
            st.caption("Nguồn dữ liệu: PostgreSQL (`voice.feat_prosodic`)")
        except Exception as exc:
            st.warning(f"Không đọc được PostgreSQL ({exc}); chuyển sang đọc CSV offline.")
            ids, M, meta, mean, std = _load_offline_data()
    else:
        ids, M, meta, mean, std = _load_offline_data()
        st.caption("Nguồn dữ liệu: CSV offline (`docs/B/features_prosodic.csv`)")

    col1, col2 = st.columns([2, 1])
    with col1:
        source_mode = st.radio(
            "Chọn nguồn file truy vấn",
            ["Tải file âm thanh mới (.wav/.mp3/.m4a)", "Chọn mẫu từ data/wav/"],
            horizontal=True,
            key="b_source_mode",
        )
    with col2:
        metric = st.selectbox(
            "Độ đo khoảng cách",
            ["euclidean", "cosine", "mahalanobis"],
            index=["euclidean", "cosine", "mahalanobis"].index(tb.DEFAULT_METRIC),
            key="b_metric",
        )

    qvec = None
    q_label = None
    exclude_fid = None

    if source_mode.startswith("Tải"):
        uploaded = st.file_uploader("Tải lên giọng nói nam cần tìm", type=["wav", "mp3", "m4a", "ogg", "flac"], key="b_upload")
        if uploaded is not None:
            st.audio(uploaded)
            suffix = os.path.splitext(uploaded.name)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                with st.spinner("Đang tiền xử lý (-23 LUFS, cắt lặng) và trích 11 đặc trưng prosodic..."):
                    qvec = tb.extract(tmp_path, skip_preprocess=False)
                q_label = uploaded.name
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    else:
        sample_id = st.selectbox("Chọn file trong tập 1.000 file", ids.tolist(), index=0, key="b_sample_id")
        wav_path = _resolve_wav_path(sample_id, meta)
        if wav_path:
            st.audio(wav_path)
        exclude_fid = sample_id
        q_label = sample_id
        idx = int(np.where(ids == sample_id)[0][0])
        qvec = M[idx]

    if qvec is None:
        st.info("Hãy tải lên 1 file âm thanh hoặc chọn 1 file mẫu để tìm top 5 giọng nói tương đồng.")
        return

    if exclude_fid is not None:
        keep = ids != exclude_fid
        search_ids, search_M = ids[keep], M[keep]
    else:
        search_ids, search_M = ids, M

    results = tb.search(qvec, search_ids, search_M, k=5, metric=metric, mean=mean, std=std)
    q_z = (qvec - mean) / np.where(std > 1e-12, std, 1.0)

    with st.expander("Xem vector đặc trưng đã trích (11 chiều & z-score)", expanded=False):
        feat_df = pd.DataFrame({
            "Đặc trưng": tb.COLS,
            "Giá trị gốc": np.round(qvec, 4),
            "Z-score": np.round(q_z, 4),
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("### Top 5 kết quả giống nhất")
    res_rows = []
    for rank, (fid, sim, dist) in enumerate(results, start=1):
        row_meta = meta.loc[fid]
        res_rows.append({
            "Hạng": rank,
            "file_id": fid,
            "speaker_id": row_meta["speaker_id"],
            "Vùng/Accent": row_meta["region_accent"],
            "Ngôn ngữ": row_meta["language"],
            "Độ tương đồng": round(sim, 4),
            "Khoảng cách": round(dist, 4),
        })
    st.dataframe(pd.DataFrame(res_rows), use_container_width=True, hide_index=True)

    for rank, (fid, sim, dist) in enumerate(results, start=1):
        row_meta = meta.loc[fid]
        c_left, c_right = st.columns([1.4, 2.0])
        with c_left:
            st.write(f"**#{rank} {fid}** — `{row_meta['speaker_id']}` ({row_meta['region_accent']}/{row_meta['language']}) | độ tương đồng=`{sim:.3f}`")
        with c_right:
            wp = _resolve_wav_path(fid, meta)
            if wp:
                st.audio(wp)
            else:
                st.caption("Không tìm thấy file WAV trên đĩa.")

    r1_id = results[0][0]
    r1_idx = int(np.where(ids == r1_id)[0][0])
    r1_z = (M[r1_idx] - mean) / np.where(std > 1e-12, std, 1.0)
    fig = _plot_compare_rank1(q_label, q_z, r1_id, r1_z)
    st.pyplot(fig)
    plt.close(fig)
