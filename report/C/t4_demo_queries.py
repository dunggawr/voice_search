"""T4 - Minh hoa 2 truy van: 1 seen (Q41) va 1 unseen (Q11).

Chay: python -m report.C.t4_demo_queries
Ket qua: in bang xep hang va luu bieu do report/C/t4_seen.png, report/C/t4_unseen.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from common import db
from tracks.track_a_embedding import load_matrix, search

# ---------------------------------------------------------------------------
# Ket noi & load du lieu
# ---------------------------------------------------------------------------
conn = db.connect()
ids, M = load_matrix(conn)
pos = {f: i for i, f in enumerate(ids)}
meta = db.read_df(conn, "SELECT file_id, speaker_id, region_accent, language FROM audio_files")
meta = meta.set_index("file_id")
query_set = pd.read_csv("data/query_set.csv")
conn.close()

def run_query(q_row, label):
    """Chay 1 truy van, in bang, ve bieu do."""
    file_id  = q_row["file_id"]
    spk_id   = q_row["speaker_id"]
    region   = q_row["region_accent"]
    kind     = q_row["kind"]
    duration = q_row["duration_sec"]

    print(f"\n{'='*60}")
    print(f"[{label}] query_id={q_row['query_id']}  kind={kind}")
    print(f"  file_id      : {file_id}")
    print(f"  speaker_id   : {spk_id}")
    print(f"  region_accent: {region}")
    print(f"  duration_sec : {duration:.2f} s")

    # Vector query tu CSDL
    qvec = M[pos[file_id]]
    print(f"\n  Vector (192 chieu): min={qvec.min():.4f}  max={qvec.max():.4f}"
          f"  mean={qvec.mean():.4f}  norm={np.linalg.norm(qvec):.4f}")

    # Sau L2-normalize
    qn = qvec / (np.linalg.norm(qvec) + 1e-12)
    print(f"  Sau L2-norm  : norm={np.linalg.norm(qn):.6f} (phai ~1.0)")

    # Loai exclude_ids
    excl = set(str(q_row["exclude_ids"]).split(";"))
    keep = np.array([f not in excl for f in ids])
    ids_k, M_k = ids[keep], M[keep]

    # Tim kiem
    results = search(qvec, ids_k, M_k, k=5)

    # Bang ket qua
    print(f"\n  Top-5 ket qua:")
    rows = []
    for rank, (fid, sim, dist) in enumerate(results, 1):
        spk   = meta.loc[fid, "speaker_id"]
        reg   = meta.loc[fid, "region_accent"]
        match = "DUNG" if spk == spk_id else "sai "
        print(f"    {rank}. {fid}  spk={spk}  region={reg}  sim={sim:.4f}  dist={dist:.4f}  [{match}]")
        rows.append({"rank": rank, "file_id": fid, "speaker_id": spk,
                     "region_accent": reg, "similarity": sim, "distance": dist,
                     "match": spk == spk_id})

    # Bieu do bar chart similarity
    df_res = pd.DataFrame(rows)
    colors = ["#2ecc71" if m else "#e74c3c" for m in df_res["match"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(df_res["file_id"][::-1], df_res["similarity"][::-1], color=colors[::-1])
    ax.set_xlabel("Cosine Similarity [0, 1]")
    ax.set_title(f"[{label}] {file_id} ({spk_id}, {region}, {duration:.1f}s)\n"
                 f"kind={kind}  Xanh=cung speaker, Do=khac speaker")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, df_res["similarity"][::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    out_path = f"report/C/t4_{kind}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  Bieu do da luu: {out_path}")
    return df_res

# ---------------------------------------------------------------------------
# Truy van 1: SEEN - Q41 (SPK_VIVOSDEV10, Nam, 2.91 s)
# ---------------------------------------------------------------------------
q_seen   = query_set[query_set["query_id"] == "Q41"].iloc[0]
df_seen  = run_query(q_seen, "SEEN Q41")

# ---------------------------------------------------------------------------
# Truy van 2: UNSEEN - Q11 (SPK_VIVOSDEV06, Nam, 3.62 s)
# ---------------------------------------------------------------------------
q_unseen  = query_set[query_set["query_id"] == "Q11"].iloc[0]
df_unseen = run_query(q_unseen, "UNSEEN Q11")

print("\n\nTom tat:")
print(f"  SEEN   top1={'DUNG' if df_seen.iloc[0]['match'] else 'sai'}"
      f"  sim={df_seen.iloc[0]['similarity']:.4f}")
print(f"  UNSEEN group@5={df_unseen['match'].mean():.2f} (speaker ko co trong CSDL, chi xem region/lang)")
