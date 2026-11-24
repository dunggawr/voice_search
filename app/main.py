"""App demo Voice Search - Streamlit.

Cau truc: 4 tab
  Tab 1 (C): Embedding ECAPA-TDNN
  Tab 2 (A): Spectral (chua co - placeholder)
  Tab 3 (B): Prosodic (chua co - placeholder)
  Tab 4   : Fused (chua co - placeholder)

Chay: streamlit run app/main.py
"""
import streamlit as st

st.set_page_config(
    page_title="Voice Search Demo",
    page_icon="🔊",
    layout="wide",
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Embedding (C)",
    "📊 Spectral (A)",
    "🎵 Prosodic (B)",
    "🔀 Fused",
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
    st.header("🎵 Prosodic — F0, Energy, Jitter, Shimmer")
    st.info("Chua co — cho thanh vien B (track_b_prosodic.py) hoan thanh T6 cua ho.")

with tab4:
    st.header("🔀 Fused — Hop nhat 3 bo dac trung")
    st.info("Chua co — cho tat ca 3 tuyen hoan thanh va thanh vien B tich hop.")
