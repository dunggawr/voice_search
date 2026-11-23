"""Ham dung chung de doc/ghi PostgreSQL (schema "voice"). Moi tuyen dung cac ham nay, khong tu viet lai.

Ket noi: bien moi truong DATABASE_URL, mac dinh khop voi docker-compose.yml:
    postgresql://postgres:postgres@localhost:5432/voice_search
"""
import os
import numpy as np
import pandas as pd
import psycopg2

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/voice_search"
SCHEMA = "voice"
TABLES = {"prosodic": "feat_prosodic", "spectral": "feat_spectral", "embedding": "feat_embedding"}


def connect(dsn=None):
    """Ket noi PostgreSQL va dat search_path = voice (khong can go ten schema trong cau SQL)."""
    conn = psycopg2.connect(dsn or os.environ.get("DATABASE_URL", DEFAULT_DSN))
    conn.set_client_encoding("UTF8")      # tranh loi chu co dau / dau gach "–" tren may Windows hoac he thong locale C
    with conn.cursor() as cur:
        cur.execute(f"SET search_path TO {SCHEMA}")
    conn.commit()
    return conn


def read_df(conn, sql, params=None):
    """Chay SELECT, tra ve DataFrame."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def feature_names(conn, feature_set):
    """Ten cot dac trung theo thu tu trong bang (embedding: emb_1 ... emb_192)."""
    if feature_set == "embedding":
        return [f"emb_{i}" for i in range(1, 193)]
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema=%s AND table_name=%s AND column_name <> 'file_id' ORDER BY ordinal_position",
                    (SCHEMA, TABLES[feature_set]))
        return [r[0] for r in cur.fetchall()]


def load_matrix(conn, feature_set):
    """Tra ve (ids, M): ids la mang file_id (sap xep tang dan), M la ma tran float64 (n_file x n_dac_trung)."""
    table = TABLES[feature_set]
    with conn.cursor() as cur:
        if feature_set == "embedding":
            cur.execute(f"SELECT file_id, vector FROM {table} ORDER BY file_id")
            rows = cur.fetchall()
            if not rows:
                return np.array([], dtype=object), np.empty((0, 192), dtype=np.float64)
            ids = np.array([r[0] for r in rows], dtype=object)
            return ids, np.array([r[1] for r in rows], dtype=np.float64)
        cols = feature_names(conn, feature_set)
        cur.execute(f"SELECT file_id, {', '.join(cols)} FROM {table} ORDER BY file_id")
        rows = cur.fetchall()
        if not rows:
            return np.array([], dtype=object), np.empty((0, len(cols)), dtype=np.float64)
    ids = np.array([r[0] for r in rows], dtype=object)
    return ids, np.array([r[1:] for r in rows], dtype=np.float64)


def save_embedding(conn, file_id, vec):
    """Ghi (hoac cap nhat) embedding 192 chieu cua 1 file. Nho conn.commit() sau khi ghi xong."""
    vec = [float(x) for x in np.asarray(vec).ravel()]
    assert len(vec) == 192, "embedding phai co 192 chieu"
    with conn.cursor() as cur:
        cur.execute("INSERT INTO feat_embedding(file_id, vector) VALUES (%s, %s) "
                    "ON CONFLICT (file_id) DO UPDATE SET vector = EXCLUDED.vector", (file_id, vec))


def save_norm_params(conn, feature_set, names, M):
    """Luu mean/std tung cot (tinh tren toan bo 1000 file) de dung z-score luc truy van. Nho commit()."""
    mean, std = M.mean(axis=0), M.std(axis=0)
    rows = [(feature_set, n, float(m), float(s if s > 1e-12 else 1.0)) for n, m, s in zip(names, mean, std)]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM norm_params WHERE feature_set=%s", (feature_set,))
        cur.executemany("INSERT INTO norm_params VALUES (%s,%s,%s,%s)", rows)


def load_norm_params(conn, feature_set):
    """Tra ve (mean, std) theo thu tu cot cua bang."""
    if feature_set == "embedding":
        raise ValueError("Tuyen embedding khong dung norm_params (z-score), dung cosine similarity.")
    names = feature_names(conn, feature_set)
    with conn.cursor() as cur:
        cur.execute("SELECT feature_name, mean, std FROM norm_params WHERE feature_set=%s", (feature_set,))
        d = {n: (m, s) for n, m, s in cur.fetchall()}
    return np.array([d[n][0] for n in names]), np.array([d[n][1] for n in names])
