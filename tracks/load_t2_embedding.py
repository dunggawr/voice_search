"""Nap 1000 vector embedding tu docs/C/features_embedding.csv vao bang feat_embedding.
Viec T2 cua tuyen C.
"""
import pandas as pd
from psycopg2.extras import execute_values
from common import db

def load_t2(csv_path="docs/C/features_embedding.csv"):
    conn = db.connect()
    df = pd.read_csv(csv_path)
    emb_cols = [f"emb_{i}" for i in range(1, 193)]
    
    rows = [(str(row["file_id"]), [float(x) for x in row[emb_cols]]) for _, row in df.iterrows()]
    
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO feat_embedding (file_id, vector) VALUES %s "
            "ON CONFLICT (file_id) DO UPDATE SET vector = EXCLUDED.vector",
            rows,
            template="(%s, %s::real[])",
            page_size=100
        )
    conn.commit()
    
    # Kiem tra lai
    ids, M = db.load_matrix(conn, "embedding")
    print(f"Nap thanh cong feat_embedding: {len(ids)} files, ma tran shape: {M.shape}")
    conn.close()

if __name__ == "__main__":
    load_t2()
