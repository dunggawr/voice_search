"""Giai doan 2B (Thanh vien B): Hop nhat diem 3 bo dac trung (search_fused).

Chuan hoa diem tuong dong cua tung tuyen ve [0, 1], tinh diem tong hop co trong so:
    sim_fused = w_prosodic * sim_prosodic + w_spectral * sim_spectral + w_embedding * sim_embedding
    dist_fused = 1.0 - sim_fused

Chon trong so bang grid search tren tap `dev` (khong chinh theo `test`), sau do
danh gia so sanh 4 he thong (prosodic, spectral, embedding, fused) tren `test`.

Cach chay:
    python -m tracks.fusion
"""
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import db, evaluate
from tracks import track_b_prosodic as tb

FEATURE_SETS = ("prosodic", "spectral", "embedding")
DEFAULT_METRICS = {
    "prosodic": tb.DEFAULT_METRIC,
    "spectral": "cosine",
    "embedding": "cosine",
}
DEFAULT_WEIGHTS = {
    "prosodic": 0.05,
    "spectral": 0.15,
    "embedding": 0.80,
}


def load_all_from_db(conn):
    """Doc ma tran va tham so z-score cua ca 3 bo tu PostgreSQL."""
    matrices = {}
    norm_params = {}
    ids_ref = None
    for fs in FEATURE_SETS:
        ids, M = db.load_matrix(conn, fs)
        if ids_ref is None:
            ids_ref = ids
        elif not np.array_equal(ids_ref, ids):
            order = {fid: idx for idx, fid in enumerate(ids)}
            M = M[[order[fid] for fid in ids_ref]]
        matrices[fs] = M
        try:
            norm_params[fs] = db.load_norm_params(conn, fs)
        except Exception:
            m = M.mean(axis=0)
            s = np.where(M.std(axis=0) > 1e-12, M.std(axis=0), 1.0)
            norm_params[fs] = (m, s)
    return ids_ref, matrices, norm_params


def load_all_from_csv(
    prosodic_csv="docs/B/features_prosodic.csv",
    spectral_csv="docs/C/features_spectral.csv",
    embedding_csv="docs/C/features_embedding.csv",
):
    """Doc ma tran va tham so z-score cua ca 3 bo tu CSV giai doan 1 (offline)."""
    paths = {
        "prosodic": prosodic_csv,
        "spectral": spectral_csv,
        "embedding": embedding_csv,
    }
    matrices = {}
    norm_params = {}
    ids_ref = None
    for fs in FEATURE_SETS:
        df = pd.read_csv(paths[fs]).sort_values("file_id").reset_index(drop=True)
        ids = df["file_id"].to_numpy(dtype=object)
        if ids_ref is None:
            ids_ref = ids
        elif not np.array_equal(ids_ref, ids):
            df = df.set_index("file_id").loc[ids_ref].reset_index()
        feat_cols = [c for c in df.columns if c != "file_id"]
        M = df[feat_cols].to_numpy(dtype=np.float64)
        matrices[fs] = M
        m = M.mean(axis=0)
        s = np.where(M.std(axis=0) > 1e-12, M.std(axis=0), 1.0)
        norm_params[fs] = (m, s)
    return ids_ref, matrices, norm_params


def score_single_modality(feature_set, qvec, M, metric=None, mean=None, std=None):
    """Tinh (similarities, distances) trong [0, 1] cho 1 tuyen."""
    metric = metric or DEFAULT_METRICS[feature_set]
    q = np.asarray(qvec, dtype=np.float64).ravel()
    M = np.asarray(M, dtype=np.float64)

    if feature_set == "prosodic":
        return tb.compute_scores(q, M, metric=metric, mean=mean, std=std)

    if feature_set == "spectral":
        if mean is None or std is None:
            m = M.mean(axis=0)
            s = np.where(M.std(axis=0) > 1e-12, M.std(axis=0), 1.0)
        else:
            m, s = np.asarray(mean, dtype=np.float64), np.where(np.asarray(std) > 1e-12, std, 1.0)
        zq = (q - m) / s
        ZM = (M - m) / s
    elif feature_set == "embedding":
        # Embedding ECAPA-TDNN da nam tren khong gian chuan hoa; cosine truc tiep la chuan
        zq = q
        ZM = M
    else:
        raise ValueError(f"feature_set khong hop le: {feature_set}")

    if metric == "cosine":
        qn = np.linalg.norm(zq)
        Mn = np.linalg.norm(ZM, axis=1)
        cos = np.clip((ZM @ zq) / np.maximum(qn * Mn, 1e-12), -1.0, 1.0)
        dists = 1.0 - cos
        # Spectral (A) dung 1/(1+d), Embedding (C) dung (cos+1)/2
        sims = 1.0 / (1.0 + dists) if feature_set == "spectral" else np.clip((cos + 1.0) / 2.0, 0.0, 1.0)
    elif metric == "euclidean":
        dists = np.linalg.norm(ZM - zq, axis=1)
        sims = 1.0 / (1.0 + dists)
    else:
        raise ValueError(f"Do do khong ho tro cho {feature_set}: {metric}")

    return sims.astype(np.float64), dists.astype(np.float64)


def search_fused(qvecs, ids, matrices, weights=None, k=5, norm_params=None, metrics=None):
    """2B. Hop nhat diem tuong dong cua 3 bo dac trung, tra ve top-k giam dan.

    qvecs: dict {feature_set: vector 1D}
    matrices: dict {feature_set: ma tran (n_files x n_dims)} cung thu tu voi `ids`
    """
    weights = weights or DEFAULT_WEIGHTS
    metrics = metrics or DEFAULT_METRICS
    norm_params = norm_params or {}

    w_sum = sum(weights[fs] for fs in FEATURE_SETS)
    if w_sum <= 0:
        raise ValueError("Tong trong so phai > 0")

    fused_sim = np.zeros(len(ids), dtype=np.float64)
    for fs in FEATURE_SETS:
        w = weights[fs] / w_sum
        if w == 0:
            continue
        m, s = norm_params.get(fs, (None, None))
        sims, _ = score_single_modality(fs, qvecs[fs], matrices[fs], metric=metrics[fs], mean=m, std=s)
        fused_sim += w * sims

    fused_dist = 1.0 - fused_sim
    order = np.lexsort((fused_dist, -fused_sim))[:k]
    return [(ids[i], float(fused_sim[i]), float(fused_dist[i])) for i in order]


def _evaluate_fused(ids, matrices, norm_params, meta, queries, split, weights, metrics):
    pos = {f: i for i, f in enumerate(ids)}

    def qvec_fn(fid):
        idx = pos[fid]
        return {fs: matrices[fs][idx] for fs in FEATURE_SETS}

    def search_fn(qvecs, sub_ids, sub_idx, k=5):
        sub_mats = {fs: matrices[fs][sub_idx] for fs in FEATURE_SETS}
        return search_fused(qvecs, sub_ids, sub_mats, weights=weights, k=k, norm_params=norm_params, metrics=metrics)

    # Dung evaluate() bang cach truyen chi so hang lam ma tran gia
    idx_col = np.arange(len(ids))
    return evaluate.evaluate(search_fn, ids, idx_col, meta, queries, split=split, qvec_fn=qvec_fn)


def grid_search_weights_dev(ids, matrices, norm_params, meta, queries, metrics, step=0.05):
    """Chon trong so toi uu tren tap `dev` (khong cham vao `test`)."""
    # Precompute similarity matrix cho tung bo tren tap dev de grid search nhanh va chinh xac
    dev_q = queries[queries.split == "dev"].copy()
    pos = {f: i for i, f in enumerate(ids)}
    meta_idx = meta.set_index("file_id")
    spk_files = meta_idx.groupby("speaker_id").size()

    precomp = []
    for q in dev_q.itertuples():
        excl = set(str(q.exclude_ids).split(";"))
        keep = np.array([f not in excl for f in ids])
        sub_ids = ids[keep]
        q_idx = pos[q.file_id]
        sims_by_fs = {}
        for fs in FEATURE_SETS:
            m, s = norm_params.get(fs, (None, None))
            sims_by_fs[fs], _ = score_single_modality(
                fs, matrices[fs][q_idx], matrices[fs][keep], metric=metrics[fs], mean=m, std=s
            )
        precomp.append((q, sub_ids, sims_by_fs))

    steps = int(round(1.0 / step))
    grid_rows = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k_emb = steps - i - j
            wp, ws, we = round(i * step, 4), round(j * step, 4), round(k_emb * step, 4)
            seen_top1, seen_p5, seen_ap5 = [], [], []
            unseen_g5, unseen_l5 = [], []
            for q, sub_ids, sims_by_fs in precomp:
                fused = wp * sims_by_fs["prosodic"] + ws * sims_by_fs["spectral"] + we * sims_by_fs["embedding"]
                top = [sub_ids[idx] for idx in np.argsort(-fused)[:5]]
                if q.kind == "seen":
                    rel = [meta_idx.loc[f, "speaker_id"] == q.speaker_id for f in top]
                    seen_top1.append(float(rel[0]))
                    seen_p5.append(float(np.mean(rel)))
                    seen_ap5.append(evaluate._ap_at_k(rel, int(spk_files[q.speaker_id]) - 1))
                else:
                    unseen_g5.append(float(np.mean([meta_idx.loc[f, "region_accent"] == q.region_accent for f in top])))
                    unseen_l5.append(float(np.mean([meta_idx.loc[f, "language"] == meta_idx.loc[q.file_id, "language"] for f in top])))
            grid_rows.append({
                "w_prosodic": wp, "w_spectral": ws, "w_embedding": we,
                "dev_seen_top1": float(np.mean(seen_top1)),
                "dev_seen_p5": float(np.mean(seen_p5)),
                "dev_seen_map5": float(np.mean(seen_ap5)),
                "dev_unseen_group5": float(np.mean(unseen_g5)),
                "dev_unseen_lang5": float(np.mean(unseen_l5)),
                # Tieu chi chon: uu tien seen MAP@5, seen P@5, seen top1, va giu ca 3 tuyen co dong gop (>0) neu hoa
                "all_positive": int(wp > 0 and ws > 0 and we > 0),
            })

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["dev_seen_map5", "dev_seen_p5", "dev_seen_top1", "dev_unseen_group5", "all_positive", "w_embedding"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    return grid_df


def run_fusion_benchmark(from_db=False, dsn=None):
    os.makedirs("report/B", exist_ok=True)
    if from_db:
        conn = db.connect(dsn)
        try:
            ids, matrices, norm_params = load_all_from_db(conn)
            meta = db.read_df(conn, "SELECT file_id, speaker_id, region_accent, language FROM audio_files ORDER BY file_id")
        finally:
            conn.close()
    else:
        ids, matrices, norm_params = load_all_from_csv()
        meta = pd.read_csv("data/metadata_master.csv")[["file_id", "speaker_id", "region_accent", "language"]]

    queries = pd.read_csv("data/query_set.csv")
    metrics = dict(DEFAULT_METRICS)

    grid_df = grid_search_weights_dev(ids, matrices, norm_params, meta, queries, metrics, step=0.05)
    grid_df.to_csv("report/B/eval_fusion_weights_dev.csv", index=False)

    # Chon bo trong so co ca 3 tuyen dong gop tot nhat, dong thoi in top-5 grid
    best_all = grid_df[grid_df.all_positive == 1].iloc[0]
    best_overall = grid_df.iloc[0]
    chosen = best_all if (
        best_all.dev_seen_map5 >= best_overall.dev_seen_map5 - 1e-9
        and best_all.dev_seen_p5 >= best_overall.dev_seen_p5 - 1e-9
        and best_all.dev_seen_top1 >= best_overall.dev_seen_top1 - 1e-9
    ) else best_overall

    best_weights = {
        "prosodic": float(chosen.w_prosodic),
        "spectral": float(chosen.w_spectral),
        "embedding": float(chosen.w_embedding),
    }

    print("=== 2B: Top-5 trong so tren tap DEV ===")
    print(grid_df.head(5).round(4).to_string(index=False))
    print(f"=> Trong so duoc chon: {best_weights}")

    # So sanh 4 he thong tren dev & test
    comp_rows = []
    for split in ["dev", "test"]:
        for fs in FEATURE_SETS:
            m, s = norm_params[fs]
            def single_search(qvec, sub_ids, sub_M, k=5, _fs=fs, _m=m, _s=s):
                sims, dists = score_single_modality(_fs, qvec, sub_M, metric=metrics[_fs], mean=_m, std=_s)
                order = np.lexsort((dists, -sims))[:k]
                return [(sub_ids[idx], float(sims[idx]), float(dists[idx])) for idx in order]

            df_s = evaluate.evaluate(single_search, ids, matrices[fs], meta, queries, split=split)
            seen, unseen = df_s[df_s.kind == "seen"], df_s[df_s.kind == "unseen"]
            comp_rows.append({
                "split": split, "system": fs,
                "seen_top1": float(seen.top1.mean()), "seen_p5": float(seen.p5.mean()), "seen_map5": float(seen.ap5.mean()),
                "unseen_group5": float(unseen.group5.mean()), "unseen_lang5": float(unseen.lang5.mean()),
            })

        df_f = _evaluate_fused(ids, matrices, norm_params, meta, queries, split, best_weights, metrics)
        seen, unseen = df_f[df_f.kind == "seen"], df_f[df_f.kind == "unseen"]
        comp_rows.append({
            "split": split, "system": "fused",
            "seen_top1": float(seen.top1.mean()), "seen_p5": float(seen.p5.mean()), "seen_map5": float(seen.ap5.mean()),
            "unseen_group5": float(unseen.group5.mean()), "unseen_lang5": float(unseen.lang5.mean()),
        })

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv("report/B/eval_fusion_comparison.csv", index=False)
    print("\n=== 2B: So sanh 4 he thong tren DEV va TEST ===")
    print(comp_df.round(4).to_string(index=False))

    # Bieu do so sanh tren TEST
    test_df = comp_df[comp_df.split == "test"].set_index("system").loc[["prosodic", "spectral", "embedding", "fused"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(test_df))
    w = 0.24
    ax.bar(x - w, test_df["seen_top1"], width=w, label="Seen Top-1", color="#2a78d6")
    ax.bar(x, test_df["seen_p5"], width=w, label="Seen P@5", color="#1baf7a")
    ax.bar(x + w, test_df["seen_map5"], width=w, label="Seen MAP@5", color="#eb6834")
    ax.set_xticks(x)
    ax.set_xticklabels(["Prosodic", "Spectral", "Embedding", f"Hợp nhất ({best_weights['prosodic']:.2f}/{best_weights['spectral']:.2f}/{best_weights['embedding']:.2f})"])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Điểm đánh giá")
    ax.set_title("So sánh 3 bộ đặc trưng đơn lẻ và hệ thống hợp nhất (tập TEST)", loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", color="#e3e2de", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for idx, row in enumerate(test_df.itertuples()):
        ax.text(idx - w, row.seen_top1 + 0.02, f"{row.seen_top1:.2f}", ha="center", fontsize=8)
        ax.text(idx, row.seen_p5 + 0.02, f"{row.seen_p5:.2f}", ha="center", fontsize=8)
        ax.text(idx + w, row.seen_map5 + 0.02, f"{row.seen_map5:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig("report/B/fig_2b_fusion_comparison.png", dpi=160)
    plt.close(fig)
    return best_weights, comp_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-db", action="store_true")
    ap.add_argument("--dsn", default=None)
    args = ap.parse_args()
    run_fusion_benchmark(from_db=args.from_db, dsn=args.dsn)
