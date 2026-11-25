"""Chay thuc nghiem T1, T3, T4, T5 cho Tuyen B (Prosodic) va luu ket qua vao report/.

Co the chay truc tiep khong can PostgreSQL (doc tu data/metadata_master.csv va
docs/B/features_prosodic.csv), hoac tu dong dung PostgreSQL neu --from-db duoc bat.

Cach chay:
    python -m tracks.run_track_b_experiments
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import db, evaluate
from tracks import track_b_prosodic as tb

BG, TX, SEC, GR = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
ACCENT_COL = {"Nam": "#2a78d6", "UK": "#eb6834", "US": "#1baf7a"}
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GR,
    "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC,
    "text.color": TX, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GR, "grid.linewidth": 0.6, "font.size": 10,
})


def load_data(from_db=False, dsn=None):
    if from_db:
        conn = db.connect(dsn)
        try:
            ids, M = tb.load_matrix(conn)
            meta = db.read_df(conn, "SELECT file_id, speaker_id, region_accent, language FROM audio_files ORDER BY file_id")
            return ids, M, meta
        finally:
            conn.close()
    ids, M = tb.load_matrix_csv("docs/B/features_prosodic.csv")
    meta = pd.read_csv("data/metadata_master.csv")[["file_id", "speaker_id", "region_accent", "language"]]
    return ids, M, meta


def run_t1(meta, csv_path="docs/B/features_prosodic.csv", wav_dir="data/wav"):
    """T1: Kiem tra extract() tren 20 file cu (phan tang Nam/UK/US) so voi CSV."""
    df_csv = pd.read_csv(csv_path).set_index("file_id")
    samples = []
    for acc, n in [("Nam", 7), ("UK", 7), ("US", 6)]:
        sub = meta[meta.region_accent == acc].sort_values("file_id")
        idx = np.linspace(0, len(sub) - 1, n, dtype=int)
        samples.extend(sub.iloc[idx]["file_id"].tolist())

    rows = []
    for fid in samples:
        wav_path = os.path.join(wav_dir, f"{fid}.wav")
        ref = df_csv.loc[fid, tb.COLS].to_numpy(dtype=np.float64)
        v_raw = tb.extract(wav_path, skip_preprocess=True)
        v_pre = tb.extract(wav_path, skip_preprocess=False)
        abs_raw = np.abs(v_raw - ref)
        rel_raw = abs_raw / np.maximum(np.abs(ref), 1e-9)
        abs_pre = np.abs(v_pre - ref)
        rel_pre = abs_pre / np.maximum(np.abs(ref), 1e-9)
        rows.append({
            "file_id": fid,
            "region_accent": meta.set_index("file_id").loc[fid, "region_accent"],
            "max_abs_diff_skip_pre": float(abs_raw.max()),
            "max_rel_diff_skip_pre": float(rel_raw.max()),
            "max_abs_diff_with_pre": float(abs_pre.max()),
            "max_rel_diff_with_pre": float(rel_pre.max()),
        })
    t1_df = pd.DataFrame(rows)
    t1_df.to_csv("report/B/t1_prosodic_verify_20files.csv", index=False)
    print("=== T1: Kiem tra extract() tren 20 file cu ===")
    print(
        f"skip_preprocess=True : max_abs={t1_df.max_abs_diff_skip_pre.max():.2e}, "
        f"max_rel={t1_df.max_rel_diff_skip_pre.max():.2e}"
    )
    print(
        f"skip_preprocess=False: median_rel={t1_df.max_rel_diff_with_pre.median():.4f}, "
        f"max_rel={t1_df.max_rel_diff_with_pre.max():.4f}"
    )
    return t1_df


def run_t3(ids, M, meta, queries):
    """T3: So sanh Euclid, Cosine, Mahalanobis tren tap dev."""
    mean, std = M.mean(axis=0), np.where(M.std(axis=0) > 1e-12, M.std(axis=0), 1.0)
    rows = []
    for metric in ["euclidean", "cosine", "mahalanobis"]:
        def search_fn(qvec, sub_ids, sub_M, k=5, _m=metric):
            return tb.search(qvec, sub_ids, sub_M, k=k, metric=_m, mean=mean, std=std)

        df_dev = evaluate.evaluate(search_fn, ids, M, meta, queries, split="dev")
        seen = df_dev[df_dev.kind == "seen"]
        unseen = df_dev[df_dev.kind == "unseen"]
        rows.append({
            "metric": metric,
            "seen_n": len(seen),
            "seen_top1": float(seen.top1.mean()),
            "seen_p5": float(seen.p5.mean()),
            "seen_map5": float(seen.ap5.mean()),
            "unseen_n": len(unseen),
            "unseen_group5": float(unseen.group5.mean()),
            "unseen_lang5": float(unseen.lang5.mean()),
        })
    res = pd.DataFrame(rows).sort_values(["seen_p5", "seen_map5", "seen_top1"], ascending=False).reset_index(drop=True)
    res.to_csv("report/B/t3_prosodic_metrics_dev.csv", index=False)
    print("\n=== T3: So sanh do do tren dev ===")
    print(res.round(4).to_string(index=False))
    best_metric = res.iloc[0]["metric"]
    print(f"=> Chon do do tot nhat theo P@5 (tie-break MAP@5): {best_metric}")
    return res, best_metric


def _plot_intermediate(q_row, q_vec, q_z, top5_df, M_map, mean, std, out_png, title):
    r1_id = top5_df.iloc[0]["file_id"]
    r1_z = (M_map[r1_id] - mean) / std

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.35, 1.0]})

    x = np.arange(len(tb.COLS))
    w = 0.38
    axes[0].bar(x - w / 2, q_z, width=w, color="#2a78d6", label=f"Truy van ({q_row.file_id})")
    axes[0].bar(x + w / 2, r1_z, width=w, color="#eb6834", label=f"Hang 1 ({r1_id})")
    axes[0].axhline(0, color=SEC, linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tb.COLS, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Gia tri z-score")
    axes[0].set_title("Vector z-score: Truy van vs Hang 1", loc="left", fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=8.5)

    y_pos = np.arange(len(top5_df))[::-1]
    colors = [
        "#1baf7a" if (q_row.kind == "seen" and r.speaker_id == q_row.speaker_id) else "#eb6834"
        if r.region_accent != q_row.region_accent else "#2a78d6"
        for r in top5_df.itertuples()
    ]
    axes[1].barh(y_pos, top5_df["similarity"], color=colors, height=0.55)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"#{r.rank} {r.file_id} ({r.speaker_id}, {r.region_accent})" for r in top5_df.itertuples()], fontsize=8.5)
    axes[1].set_xlim(0, 1.08)
    axes[1].set_xlabel("Similarity [0, 1]")
    axes[1].set_title("Top-5 ket qua gan nhat", loc="left", fontsize=10.5)
    for yp, r in zip(y_pos, top5_df.itertuples()):
        axes[1].text(r.similarity + 0.015, yp, f"sim={r.similarity:.3f} (d={r.distance:.2f})", va="center", fontsize=8, color=SEC)

    fig.suptitle(title, fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run_t4(ids, M, meta, queries, best_metric):
    """T4: Minh hoa ket qua trung gian cho 1 truy van seen (Q21) va 1 unseen (Q01)."""
    mean, std = M.mean(axis=0), np.where(M.std(axis=0) > 1e-12, M.std(axis=0), 1.0)
    meta_idx = meta.set_index("file_id")
    M_map = {fid: M[i] for i, fid in enumerate(ids)}

    interm_rows = []
    interm_json = {}

    specs = [
        ("Q21", "report/B/fig_t4_prosodic_seen_compare.png", "Bộ Prosodic — Minh họa truy vấn SEEN (Q21 - A_0744, SPK_VIVOSDEV03, Nam)"),
        ("Q01", "report/B/fig_t4_prosodic_unseen_compare.png", "Bộ Prosodic — Minh họa truy vấn UNSEEN (Q01 - A_0652, SPK_VIVOSDEV01, Nam)"),
    ]
    for qid, png_path, title in specs:
        q_row = queries[queries.query_id == qid].iloc[0]
        excl = set(str(q_row.exclude_ids).split(";"))
        keep = np.array([f not in excl for f in ids])
        q_vec = M_map[q_row.file_id]
        q_z = (q_vec - mean) / std
        top5 = tb.search(q_vec, ids[keep], M[keep], k=5, metric=best_metric, mean=mean, std=std)

        top5_records = []
        for rank, (fid, sim, dist) in enumerate(top5, start=1):
            sp = meta_idx.loc[fid, "speaker_id"]
            acc = meta_idx.loc[fid, "region_accent"]
            lang = meta_idx.loc[fid, "language"]
            top5_records.append({
                "query_id": qid, "query_file": q_row.file_id, "query_kind": q_row.kind,
                "query_speaker": q_row.speaker_id, "rank": rank, "file_id": fid,
                "speaker_id": sp, "region_accent": acc, "language": lang,
                "same_speaker": sp == q_row.speaker_id,
                "similarity": sim, "distance": dist,
            })
        top5_df = pd.DataFrame(top5_records)
        interm_rows.extend(top5_records)
        interm_json[qid] = {
            "query_file": q_row.file_id,
            "kind": q_row.kind,
            "speaker_id": q_row.speaker_id,
            "region_accent": q_row.region_accent,
            "metric": best_metric,
            "raw_vector": {c: float(v) for c, v in zip(tb.COLS, q_vec)},
            "zscore_vector": {c: float(v) for c, v in zip(tb.COLS, q_z)},
            "top5": top5_records,
        }
        _plot_intermediate(q_row, q_vec, q_z, top5_df, M_map, mean, std, png_path, title)

    pd.DataFrame(interm_rows).to_csv("report/B/t4_prosodic_intermediate.csv", index=False)
    with open("report/B/t4_prosodic_intermediate.json", "w", encoding="utf8") as f:
        json.dump(interm_json, f, ensure_ascii=False, indent=2)
    print("\n=== T4: Da luu minh hoa trung gian vao report/t4_prosodic_intermediate.{csv,json} va 2 bieu do ===")


def run_t5(ids, M, meta, queries, best_metric):
    """T5: Danh gia dev & test, phan tich loi, tao mau cham nghe thu."""
    mean, std = M.mean(axis=0), np.where(M.std(axis=0) > 1e-12, M.std(axis=0), 1.0)
    meta_idx = meta.set_index("file_id")

    def search_fn(qvec, sub_ids, sub_M, k=5):
        return tb.search(qvec, sub_ids, sub_M, k=k, metric=best_metric, mean=mean, std=std)

    print(f"\n=== T5: Danh gia voi metric={best_metric} ===")
    df_dev = evaluate.evaluate(search_fn, ids, M, meta, queries, split="dev")
    df_test = evaluate.evaluate(search_fn, ids, M, meta, queries, split="test")
    df_dev.to_csv("report/B/eval_track_b_prosodic_dev.csv", index=False)
    df_test.to_csv("report/B/eval_track_b_prosodic_test.csv", index=False)
    print("DEV:\n" + evaluate.summarize(df_dev))
    print("TEST:\n" + evaluate.summarize(df_test))

    # Phan tich loi tren test
    err_rows = []
    for r in df_test.itertuples():
        top_ids = r.top5.split()
        r1_id = top_ids[0]
        r1_spk = meta_idx.loc[r1_id, "speaker_id"]
        r1_acc = meta_idx.loc[r1_id, "region_accent"]
        is_err = (r.kind == "seen" and r.top1 < 1.0) or (r.kind == "unseen" and r.group5 < 0.6)
        if is_err:
            err_rows.append({
                "query_id": r.query_id,
                "kind": r.kind,
                "file_id": r.file_id,
                "speaker_id": r.speaker_id,
                "region_accent": r.region_accent,
                "rank1_file": r1_id,
                "rank1_speaker": r1_spk,
                "rank1_accent": r1_acc,
                "top1_or_group5": r.top1 if r.kind == "seen" else r.group5,
                "p5_or_lang5": r.p5 if r.kind == "seen" else r.lang5,
                "top5": r.top5,
            })
    pd.DataFrame(err_rows).to_csv("report/B/t5_prosodic_error_analysis.csv", index=False)

    # Tao mau 10 truy van cham nghe thu (5 seen + 5 unseen tu test)
    listen = pd.concat([
        df_test[df_test.kind == "seen"].head(5),
        df_test[df_test.kind == "unseen"].head(5),
    ], ignore_index=True)[["query_id", "kind", "region_accent", "file_id", "speaker_id", "top5"]]
    listen["score_A_1to5"] = ""
    listen["score_B_1to5"] = ""
    listen["score_C_1to5"] = ""
    listen["note"] = ""
    listen.to_csv("report/B/t5_listening_eval_template.csv", index=False)
    print(f"Da luu {len(err_rows)} truong hop can phan tich loi va mau cham nghe thu 10 truy van.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-db", action="store_true")
    ap.add_argument("--dsn", default=None)
    args = ap.parse_args()

    os.makedirs("report/B", exist_ok=True)
    ids, M, meta = load_data(from_db=args.from_db, dsn=args.dsn)
    queries = pd.read_csv("data/query_set.csv")

    run_t1(meta)
    _, best_metric = run_t3(ids, M, meta, queries)
    run_t4(ids, M, meta, queries, best_metric)
    run_t5(ids, M, meta, queries, best_metric)


if __name__ == "__main__":
    main()
