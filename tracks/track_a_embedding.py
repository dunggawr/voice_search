"""Tuyen C (Thanh vien C) - Speaker Embedding voi ECAPA-TDNN (192 chieu).

FEATURE_SET = "embedding"
  - extract(wav_path)       : file wav moi -> np.ndarray (192,) float64
  - load_matrix(conn)       : (ids, M) tu bang feat_embedding
  - search(qvec, ids, M, k) : cosine similarity, tra ve k ket qua sap xep giam dan

Luu y:
  - Khong dung z-score. Cosine tu chuan hoa L2.
  - extract() chi goi cho file MOI (file demo, file tu thu). 1000 file giai doan 1 da co san trong CSDL.
  - Model SpeechBrain tai lan dau ~80 MB, cache tai ~/.cache/huggingface/
"""
import numpy as np
from common import db
from common.preprocess import preprocess

FEATURE_SET = "embedding"

# ---------------------------------------------------------------------------
# Load model mot lan o muc module (tranh tai lai moi lan goi extract)
# SpeechBrain 1.0+ doi ten import; tuong thich ca hai phien ban
# ---------------------------------------------------------------------------
try:
    from speechbrain.inference.classifiers import EncoderClassifier as _EC
except ImportError:
    from speechbrain.pretrained import EncoderClassifier as _EC  # type: ignore

import torch as _torch

_classifier = None  # lazy load, chi tai khi extract() duoc goi lan dau

def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = _EC.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
    return _classifier


# ---------------------------------------------------------------------------
# T1 - Trich dac trung file moi
# ---------------------------------------------------------------------------
def extract(wav_path):
    """Nhan duong dan 1 file am thanh MOI -> np.ndarray shape (192,) float64.

    Bat buoc goi preprocess() truoc khi trich dac trung.
    Kiem tra: chay tren file cu trong data/wav/ phai ra gan dung gia tri trong feat_embedding.
    """
    y, sr = preprocess(wav_path)                        # 16 kHz, mono, -23 LUFS
    wav_tensor = _torch.tensor(y).unsqueeze(0)          # (1, T)
    with _torch.no_grad():
        emb = _get_classifier().encode_batch(wav_tensor)  # (1, 1, 192)
    return emb.squeeze().numpy().astype(np.float64)     # (192,)


# ---------------------------------------------------------------------------
# T2 - Load ma tran tu CSDL
# ---------------------------------------------------------------------------
def load_matrix(conn):
    """(ids, M) tu bang feat_embedding. ids: (1000,), M: (1000, 192) float64."""
    ids, M = db.load_matrix(conn, FEATURE_SET)
    if len(ids) == 0:
        raise RuntimeError(
            "feat_embedding trong - hay chay tracks/load_t2_embedding.py truoc"
        )
    return ids, M


# ---------------------------------------------------------------------------
# T3 - Tim kiem bang Cosine Similarity
# ---------------------------------------------------------------------------
def search(qvec, ids, M, k=5):
    """Tim k file giong nhat theo cosine similarity.

    Tham so:
        qvec : np.ndarray (192,) - vector embedding file truy van
        ids  : np.ndarray (n,)   - mang file_id (da loai cac file excluded)
        M    : np.ndarray (n, 192) - ma tran embedding tuong ung voi ids
        k    : so ket qua tra ve (mac dinh 5)

    Tra ve:
        list[(file_id, similarity, distance)] sap xep GIAM DAN theo similarity
        similarity in [0, 1], cang lon cang giong
        distance = 1 - cosine (cosine distance)
    """
    # L2 normalize de dot product = cosine similarity
    qn = qvec.astype(np.float64)
    qn = qn / (np.linalg.norm(qn) + 1e-12)

    Mn = M.astype(np.float64)
    norms = np.linalg.norm(Mn, axis=1, keepdims=True)
    Mn = Mn / (norms + 1e-12)

    cosine_scores = Mn @ qn                     # (n,) in [-1, 1]
    similarity = (cosine_scores + 1.0) / 2.0    # rescale -> [0, 1]
    distance = 1.0 - cosine_scores              # cosine distance

    # Lay top k (sap xep giam dan theo similarity)
    top_idx = np.argsort(-similarity)[:k]

    return [(str(ids[i]), float(similarity[i]), float(distance[i])) for i in top_idx]
