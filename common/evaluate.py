"""Khung danh gia CHUNG cho moi tuyen va cho he thong hop nhat (viec T5 va 2B).

Cach chay:   python -m common.evaluate tracks.track_b_prosodic --split test
Module cua tuyen can co:  FEATURE_SET (prosodic|spectral|embedding)  va  search(qvec, ids, M, k=5).
(co the them load_matrix(conn); neu khong co se dung common.db.load_matrix)

Quy uoc:
  - Moi truy van lay vec dac trung cua file truy van tu ma tran trong CSDL, sau do LOAI cac file trong exclude_ids
    khoi ids/M roi moi goi search() -> khong bao gio tim thay chinh file truy van.
  - search() tra ve list[(file_id, similarity, distance)] sap xep GIAM DAN theo similarity, dung k=5 phan tu.
Chi so:
  seen   (speaker co trong CSDL): top1, P@5, AP@5 (relevant = cac file con lai cung speaker)
  unseen (speaker chua co)      : group@5 (ti le 5 ket qua cung vung/accent), lang@5 (cung ngon ngu)
  Luu y: neu loi ra file dai hon/ngan hon 5 thi khong tinh; ket qua duoc bao cao tach dev / test.
"""
import argparse
import importlib
import numpy as np
import pandas as pd
from . import db

K = 5


def _ap_at_k(rels, n_relevant, k=K):
    if n_relevant == 0:
        return 0.0
    hits, s = 0, 0.0
    for i, r in enumerate(rels[:k], 1):
        if r:
            hits += 1
            s += hits / i
    return s / min(k, n_relevant)


def evaluate(search_fn, ids, M, meta, query_set, split="test", qvec_fn=None):
    """meta: DataFrame audio_files (file_id, speaker_id, region_accent, language).
    qvec_fn(file_id) -> vec (tuy chon): dung khi muon trich dac trung tu file wav that thay vi lay tu CSDL."""
    meta = meta.set_index("file_id")
    ids = np.asarray(ids, dtype=object)
    pos = {f: i for i, f in enumerate(ids)}
    spk_files = meta.groupby("speaker_id").size()
    out = []
    for q in query_set[query_set.split == split].itertuples():
        qvec = qvec_fn(q.file_id) if qvec_fn else M[pos[q.file_id]]
        excl = set(str(q.exclude_ids).split(";"))
        keep = np.array([f not in excl for f in ids])
        res = search_fn(qvec, ids[keep], M[keep], K)
        top = [r[0] for r in res][:K]
        same_spk = [meta.loc[f, "speaker_id"] == q.speaker_id for f in top]
        row = dict(query_id=q.query_id, kind=q.kind, region_accent=q.region_accent, file_id=q.file_id,
                   speaker_id=q.speaker_id, top5=" ".join(top))
        if q.kind == "seen":
            n_rel = int(spk_files[q.speaker_id]) - 1
            row.update(top1=float(same_spk[0]), p5=float(np.mean(same_spk)), ap5=_ap_at_k(same_spk, n_rel))
        else:
            row.update(group5=float(np.mean([meta.loc[f, "region_accent"] == q.region_accent for f in top])),
                       lang5=float(np.mean([meta.loc[f, "language"] == meta.loc[q.file_id, "language"] for f in top])))
        out.append(row)
    return pd.DataFrame(out)


def summarize(df):
    seen, unseen = df[df.kind == "seen"], df[df.kind == "unseen"]
    lines = []
    if len(seen):
        lines.append(f"seen   (n={len(seen):2d}): top1={seen.top1.mean():.3f}  P@5={seen.p5.mean():.3f}  MAP@5={seen.ap5.mean():.3f}")
    if len(unseen):
        lines.append(f"unseen (n={len(unseen):2d}): group@5={unseen.group5.mean():.3f}  lang@5={unseen.lang5.mean():.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--queries", default="data/query_set.csv")
    a = ap.parse_args()
    mod = importlib.import_module(a.module)
    conn = db.connect(a.db)
    ids, M = (mod.load_matrix(conn) if hasattr(mod, "load_matrix") else db.load_matrix(conn, mod.FEATURE_SET))
    meta = pd.read_sql("SELECT file_id, speaker_id, region_accent, language FROM audio_files", conn)
    df = evaluate(mod.search, ids, M, meta, pd.read_csv(a.queries), a.split)
    print(f"== {a.module}  split={a.split}\n" + summarize(df))
    df.to_csv(f"report/eval_{a.module.split('.')[-1]}_{a.split}.csv", index=False)


if __name__ == "__main__":
    main()
