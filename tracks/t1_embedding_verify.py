"""T1 (Tuyen C - embedding): Kiem tra extract() tren 20 file cu so voi CSV giai doan 1.

Doc 20 file (phan tang Nam/UK/US) tu data/wav/, trich embedding ECAPA-TDNN,
doi so vector 192 chieu voi docs/C/features_embedding.csv (cosine sim + sai so trung binh).

Ket qua: report/C/t1_embedding_verify_20files.csv

Chay:
    python -m tracks.t1_embedding_verify [--wav-dir data/wav] [--csv docs/C/features_embedding.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd

from tracks import track_a_embedding as ta


def pick_samples(meta, n_per_region=(7, 7, 6)):
    """Chon 20 file phan tang Nam/UK/US, lay deu theo thu tu file_id."""
    rows = []
    for acc, n in zip(("Nam", "UK", "US"), n_per_region):
        sub = meta[meta.region_accent == acc].sort_values("file_id")
        idx = np.linspace(0, len(sub) - 1, n, dtype=int)
        rows.extend(sub.iloc[idx][["file_id", "region_accent"]].to_dict("records"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav-dir", default="data/wav/processed")
    ap.add_argument("--csv", default="docs/C/features_embedding.csv")
    ap.add_argument("--out", default="report/C/t1_embedding_verify_20files.csv")
    a = ap.parse_args()

    ref = pd.read_csv(a.csv).set_index("file_id")
    emb_cols = [c for c in ref.columns if c.startswith("emb_")]
    meta = pd.read_csv("data/metadata_master.csv")[["file_id", "region_accent"]]
    samples = pick_samples(meta)

    print(f"T1 embedding: trich {len(samples)} file cu o {a.wav_dir} ...")
    rows = []
    for s in samples:
        fid = s["file_id"]
        path = os.path.join(a.wav_dir, f"{fid}.wav")
        if not os.path.exists(path):
            print(f"  [SKIP] khong co {path}")
            continue
        vec = ta.extract(path)                       # (192,) - chay preprocess + ECAPA
        refv = ref.loc[fid, emb_cols].to_numpy(dtype=np.float64)
        d = np.abs(vec - refv)
        cos = float(np.dot(vec, refv) / (np.linalg.norm(vec) * np.linalg.norm(refv) + 1e-12))
        rows.append({
            "file_id": fid,
            "region_accent": s["region_accent"],
            "cosine_sim": cos,
            "mean_abs_diff": float(d.mean()),
            "max_abs_diff": float(d.max()),
            "rel_err": float(d.mean() / (np.abs(refv).mean() + 1e-12)),
        })

    out = pd.DataFrame(rows).sort_values("cosine_sim", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)

    print(f"\n=== T1 embedding: {len(out)}/20 file da trich ===")
    print(f"cosine_sim   : mean={out.cosine_sim.mean():.4f}  min={out.cosine_sim.min():.4f}  max={out.cosine_sim.max():.4f}")
    print(f"mean_abs_diff: mean={out.mean_abs_diff.mean():.3e}  max={out.max_abs_diff.max():.3e}")
    print(f"rel_err      : mean={out.rel_err.mean():.4f}  max={out.rel_err.max():.4f}")
    print(out.round(5).to_string(index=False))
    print(f"\nDa luu: {a.out}")


if __name__ == "__main__":
    main()
