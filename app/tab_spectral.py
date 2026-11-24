import os
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from psycopg2 import Error as DatabaseError

from common import db
from tracks.track_c_spectral import FEATURE_SET, extract, load_matrix, search


def render():
    st.header("Tìm giọng nói theo đặc trưng phổ")
    uploaded = st.file_uploader("Tải file WAV mới", type=["wav"], key="spectral_upload")
    if uploaded is None:
        return

    audio = uploaded.getvalue()
    st.audio(audio, format="audio/wav")

    try:
        with st.spinner("Đang trích đặc trưng và tìm top 5..."):
            with TemporaryDirectory() as folder:
                query_path = Path(folder) / "query.wav"
                query_path.write_bytes(audio)
                vector = extract(query_path)

            with closing(db.connect()) as conn:
                ids, matrix = load_matrix(conn)
                if len(ids) < 5:
                    raise ValueError("Chạy python -m tracks.load_t2_spectral để nạp dữ liệu.")
                norm_params = db.load_norm_params(conn, FEATURE_SET)
                meta = db.read_df(
                    conn,
                    "SELECT file_id, speaker_id, region_accent, language FROM audio_files",
                ).set_index("file_id")

            hits = search(vector, ids, matrix, k=5, norm_params=norm_params)
    except (DatabaseError, OSError, RuntimeError, ValueError) as exc:
        st.error(f"Không xử lý được truy vấn: {exc}")
        return

    results = pd.DataFrame(hits, columns=["file_id", "similarity", "distance"])
    results.insert(0, "rank", range(1, len(results) + 1))
    results = results.join(meta, on="file_id")
    st.subheader("Top 5 kết quả")
    st.dataframe(results.round(4), hide_index=True)

    wav_dir = Path(os.environ.get("VOICE_WAV_DIR", "data/wav"))
    for row in results.itertuples(index=False):
        st.write(f"**Hạng {row.rank} · {row.file_id}**")
        wav_path = wav_dir / f"{row.file_id}.wav"
        if wav_path.is_file():
            st.audio(str(wav_path), format="audio/wav")
        else:
            st.caption(f"Chưa tìm thấy {wav_path.name} trong thư mục WAV.")