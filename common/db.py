"""Ham dung chung de doc/ghi voice.db. Moi tuyen dung cac ham nay, khong tu viet lai."""
import sqlite3
import numpy as np

DB_PATH = "db/voice.db"
TABLES = {"prosodic": "feat_prosodic", "spectral": "feat_spectral", "embedding": "feat_embedding"}


def connect(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def load_matrix(conn, feature_set):
    """Tra ve (ids, M): ids la mang file_id (da sap xep), M la ma tran float64 (n_file x n_dac_trung)."""
    table = TABLES[feature_set]
    if feature_set == "embedding":
        rows = conn.execute(f"SELECT file_id, vector FROM {table} ORDER BY file_id").fetchall()
        ids = np.array([r[0] for r in rows], dtype=object)
        M = np.vstack([np.frombuffer(r[1], dtype="<f4") for r in rows]).astype(np.float64)
        return ids, M
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})") if c[1] != "file_id"]
    rows = conn.execute(f"SELECT file_id, {', '.join(cols)} FROM {table} ORDER BY file_id").fetchall()
    ids = np.array([r[0] for r in rows], dtype=object)
    M = np.array([r[1:] for r in rows], dtype=np.float64)
    return ids, M


def feature_names(conn, feature_set):
    if feature_set == "embedding":
        return [f"emb_{i}" for i in range(1, 193)]
    table = TABLES[feature_set]
    return [c[1] for c in conn.execute(f"PRAGMA table_info({table})") if c[1] != "file_id"]


def save_embedding(conn, file_id, vec):
    vec = np.asarray(vec, dtype="<f4")
    assert vec.shape == (192,), "embedding phai co 192 chieu"
    conn.execute("INSERT OR REPLACE INTO feat_embedding(file_id, dim, vector) VALUES (?,?,?)", (file_id, 192, vec.tobytes()))


def save_norm_params(conn, feature_set, names, M):
    """Luu mean/std tung cot (tinh tren toan bo 1000 file) de dung z-score luc truy van."""
    mean, std = M.mean(axis=0), M.std(axis=0)
    conn.execute("DELETE FROM norm_params WHERE feature_set=?", (feature_set,))
    conn.executemany("INSERT INTO norm_params VALUES (?,?,?,?)",
                     [(feature_set, n, float(m), float(s if s > 1e-12 else 1.0)) for n, m, s in zip(names, mean, std)])


def load_norm_params(conn, feature_set):
    """Tra ve (mean, std) theo thu tu cot cua bang."""
    names = feature_names(conn, feature_set)
    d = {n: (m, s) for n, m, s in conn.execute(
        "SELECT feature_name, mean, std FROM norm_params WHERE feature_set=?", (feature_set,))}
    return np.array([d[n][0] for n in names]), np.array([d[n][1] for n in names])
