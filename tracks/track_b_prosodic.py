"""Tuyen B: Dac trung Prosodic & Chat giong (11 chieu).

Thuc hien hop dong chung (T1, T2, T3):
  - extract(wav_path) -> np.ndarray (11,) float64
  - load_matrix(conn) -> (ids, M)
  - search(qvec, ids, M, k=5) -> [(file_id, similarity, distance), ...]
"""
import warnings
import numpy as np
import pandas as pd
import parselmouth
import librosa
import soundfile as sf
from parselmouth.praat import call
from scipy.signal import find_peaks

from common import db
from common.preprocess import preprocess

warnings.filterwarnings("ignore")

FEATURE_SET = "prosodic"
COLS = [
    "f0_mean",
    "f0_std",
    "f0_min",
    "f0_max",
    "energy_mean",
    "energy_std",
    "speaking_rate",
    "jitter_local",
    "shimmer_local",
    "hnr_mean",
    "zcr_mean",
]
DEFAULT_METRIC = "mahalanobis"

# Cache norm_params (mean, std) khi load_matrix(conn) duoc goi
_NORM_PARAMS = None


def extract(wav_path, skip_preprocess=False):
    """T1. Trich 11 dac trung prosodic tu 1 file am thanh.

    Mac dinh chay common.preprocess.preprocess(wav_path) cho file moi (16 kHz mono,
    cat lang top_db=25, -23 LUFS). Dat skip_preprocess=True khi kiem tra lai tren
    1000 file cu da chuan hoa trong data/wav/.
    """
    if skip_preprocess:
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float64)
    else:
        y_f32, sr = preprocess(wav_path)
        y = y_f32.astype(np.float64)

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    dur = len(y) / sr
    out = {c: 0.0 for c in COLS}

    # 1. F0 (Hz) - Pitch autocorrelation, chi lay khung co thanh (voiced)
    try:
        pitch = snd.to_pitch_ac(pitch_floor=75, pitch_ceiling=300)
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        if len(f0) >= 3:
            out["f0_mean"] = float(f0.mean())
            out["f0_std"] = float(f0.std())
            out["f0_min"] = float(f0.min())
            out["f0_max"] = float(f0.max())
    except Exception:
        pass

    # 2. Energy (dB) + Speaking rate (dinh nang luong, doc lap ngon ngu)
    try:
        it = snd.to_intensity(minimum_pitch=75, time_step=0.01)
        v = it.values[0]
        out["energy_mean"] = float(v.mean())
        out["energy_std"] = float(v.std())
        thr = np.percentile(v, 50) if len(v) > 5 else v.mean()
        pk, _ = find_peaks(v, height=thr, prominence=2.0, distance=8)
        out["speaking_rate"] = float(len(pk) / max(dur, 1e-6))
    except Exception:
        pass

    # 3. Jitter (%) & Shimmer (dB)
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 300)
        jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        shim = call([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        if np.isfinite(jit):
            out["jitter_local"] = float(jit)
        if np.isfinite(shim):
            out["shimmer_local"] = float(shim)
    except Exception:
        pass

    # 4. HNR (dB)
    try:
        h = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
        hv = h.values[0]
        hv = hv[hv > -200]
        if len(hv):
            out["hnr_mean"] = float(hv.mean())
    except Exception:
        pass

    # 5. ZCR
    out["zcr_mean"] = float(librosa.feature.zero_crossing_rate(y.astype(np.float32))[0].mean())

    vec = np.array([out[c] for c in COLS], dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def load_matrix(conn):
    """T2. Doc (ids, M) tu bang feat_prosodic va nap norm_params neu co."""
    global _NORM_PARAMS
    ids, M = db.load_matrix(conn, FEATURE_SET)
    try:
        _NORM_PARAMS = db.load_norm_params(conn, FEATURE_SET)
    except Exception:
        _NORM_PARAMS = None
    return ids, M


def load_matrix_csv(csv_path="docs/B/features_prosodic.csv"):
    """Doc (ids, M) tu CSV giai doan 1 (dung khi chay offline khong co PostgreSQL)."""
    df = pd.read_csv(csv_path).sort_values("file_id").reset_index(drop=True)
    ids = df["file_id"].to_numpy(dtype=object)
    M = df[COLS].to_numpy(dtype=np.float64)
    return ids, M


def _resolve_norm(M, mean=None, std=None):
    if mean is not None and std is not None:
        m = np.asarray(mean, dtype=np.float64)
        s = np.asarray(std, dtype=np.float64)
    elif _NORM_PARAMS is not None and _NORM_PARAMS[0].shape == (M.shape[1],):
        m, s = _NORM_PARAMS
    else:
        m = M.mean(axis=0)
        s = M.std(axis=0)
    s = np.where(s > 1e-12, s, 1.0)
    return m, s


def compute_scores(qvec, M, metric=None, mean=None, std=None):
    """Tinh (similarities, distances) giua qvec va toan bo cac hang cua M."""
    metric = metric or DEFAULT_METRIC
    q = np.asarray(qvec, dtype=np.float64).ravel()
    M = np.asarray(M, dtype=np.float64)
    m, s = _resolve_norm(M, mean, std)

    zq = (q - m) / s
    ZM = (M - m) / s

    if metric == "euclidean":
        dists = np.linalg.norm(ZM - zq, axis=1)
        sims = 1.0 / (1.0 + dists)
    elif metric == "cosine":
        qn = np.linalg.norm(zq)
        Mn = np.linalg.norm(ZM, axis=1)
        denom = np.maximum(qn * Mn, 1e-12)
        cos = np.clip((ZM @ zq) / denom, -1.0, 1.0)
        dists = 1.0 - cos
        sims = np.clip((cos + 1.0) / 2.0, 0.0, 1.0)
    elif metric == "mahalanobis":
        cov = np.cov(ZM, rowvar=False) + 1e-3 * np.eye(ZM.shape[1])
        VI = np.linalg.inv(cov)
        diff = ZM - zq
        quad = np.einsum("ij,jk,ik->i", diff, VI, diff)
        dists = np.sqrt(np.maximum(quad, 0.0))
        sims = 1.0 / (1.0 + dists)
    else:
        raise ValueError(f"Do do khong hop le: {metric}")

    return sims.astype(np.float64), dists.astype(np.float64)


def search(qvec, ids, M, k=5, metric=None, mean=None, std=None):
    """T3. Tra ve k ket qua giong nhat, sap xep giam dan theo similarity."""
    sims, dists = compute_scores(qvec, M, metric=metric, mean=mean, std=std)
    # Sap xep giam dan theo similarity, neu bang nhau thi uu tien distance nho hon
    order = np.lexsort((dists, -sims))[:k]
    return [(ids[i], float(sims[i]), float(dists[i])) for i in order]
