"""Ung dung demo Streamlit cho phan viec Thanh vien B (Tuyen B Prosodic + So sanh Hop nhat 2B).

Chay:
    streamlit run app/app.py
"""
import os
import sys
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common import db
from app.tab_b_prosodic import render_tab as render_prosodic_tab

st.set_page_config(page_title="Voice Search — Track B & Fusion", layout="wide")
st.title("He thong tim kiem giong noi nam — Tuyen B (Prosodic) & Hop nhat 2B")


@st.cache_resource
def _try_connect():
    try:
        return db.connect()
    except Exception:
        return None


conn = _try_connect()
tab1, tab2 = st.tabs(["Tuyen B: Prosodic (11 chieu)", "Giai doan 2B: Hop nhat diem (Fusion)"])

with tab1:
    render_prosodic_tab(conn)

with tab2:
    st.subheader("So sanh 3 tuyen don le va he thong hop nhat (search_fused)")
    comp_path = os.path.join(ROOT, "report", "B", "eval_fusion_comparison.csv")
    fig_path = os.path.join(ROOT, "report", "B", "fig_2b_fusion_comparison.png")
    if os.path.exists(comp_path):
        st.dataframe(pd.read_csv(comp_path), use_container_width=True, hide_index=True)
    else:
        st.info("Chua co file report/B/eval_fusion_comparison.csv. Hay chay: python -m tracks.fusion")
    if os.path.exists(fig_path):
        st.image(fig_path, caption="So sanh 3 tuyen va he thong hop nhat tren tap TEST")
