"""MAU module cho moi tuyen. Copy thanh tracks/track_<ten>.py (VD track_a_embedding.py) roi dien vao.
Giu NGUYEN ten hai ham extract va search va kieu tra ve de ghep duoc o giai doan 2B.
"""
import numpy as np
from common import db
from common.preprocess import preprocess

FEATURE_SET = "prosodic"        # prosodic (B) | spectral (C) | embedding (A)


def extract(wav_path):
    """T1. Nhan duong dan 1 file am thanh MOI -> vector dac trung 1 chieu (np.ndarray float64).
    Phai goi preprocess() truoc, dung dung tham so da chot o giai doan 1.
    Kiem tra: chay lai tren 20 file cu phai ra gan nhu dung gia tri trong CSV/CSDL."""
    y, sr = preprocess(wav_path)
    raise NotImplementedError


def load_matrix(conn):
    """T2. (ids, M) tu bang cua tuyen minh. Co the bo ham nay neu dung db.load_matrix mac dinh."""
    return db.load_matrix(conn, FEATURE_SET)


def search(qvec, ids, M, k=5):
    """T3. Tra ve k ket qua giong nhat, SAP XEP GIAM DAN theo similarity:
        [(file_id, similarity, distance), ...]
    similarity cang lon cang giong, nam trong [0, 1] (de hop nhat diem o 2B). Khong tra ve file truy van.
    Goi y: z-score bang db.load_norm_params roi khoang cach Euclid; embedding dung cosine."""
    raise NotImplementedError
