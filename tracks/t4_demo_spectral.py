"""Minh họa một truy vấn seen và một truy vấn unseen của Spectral."""

import argparse
from contextlib import closing
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import db
from tracks.track_c_spectral import FEATURE_SET, extract, load_matrix, search

OUTPUT = Path("report/A")


def run_query(query, wav_dir, ids, matrix, meta, names, norm_params):
    file_id = query["file_id"]
    kind = query["kind"]
    query_vec = extract(wav_dir / f"{file_id}.wav", preprocessed=True)

    # Seen: loại file truy vấn. Unseen: loại mọi file của người nói đó.
    excluded = set(query["exclude_ids"].split(";"))
    keep = ~np.isin(ids, list(excluded))
    results = search(query_vec, ids[keep], matrix[keep], k=5, norm_params=norm_params)
    if len(results) != 5:
        raise ValueError(f"{file_id} không có đủ 5 ứng viên.")

    ranking = pd.DataFrame(results, columns=["file_id", "similarity", "distance"])
    ranking.insert(0, "rank", range(1, 6))
    ranking = ranking.join(meta, on="file_id")
    ranking["same_speaker"] = ranking["speaker_id"].eq(query["speaker_id"])
    ranking["same_region"] = ranking["region_accent"].eq(query["region_accent"])
    ranking.to_csv(OUTPUT / f"t4_{kind}_top5.csv", index=False, float_format="%.6f")

    top_id = results[0][0]
    top_vec = matrix[np.flatnonzero(ids == top_id)[0]]
    mean, std = norm_params
    vectors = pd.DataFrame({
        "feature": names,
        "query_raw": query_vec,
        "query_z": (query_vec - mean) / std,
        "top1_raw": top_vec,
        "top1_z": (top_vec - mean) / std,
    })
    vectors.to_csv(OUTPUT / f"t4_{kind}_vectors.csv", index=False, float_format="%.9g")

    x = np.arange(1, len(names) + 1)
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    ax.plot(x, vectors["query_z"], "o-", markersize=3, label=f"Truy vấn {file_id}")
    ax.plot(x, vectors["top1_z"], "s--", markersize=3, label=f"Hạng 1 {top_id}")
    ax.axhline(0, color="0.75", linewidth=0.8)
    for edge in (13.5, 26.5, 30.5, 34.5):
        ax.axvline(edge, color="0.85", linewidth=0.8)
    ax.set_xticks(
        [7, 20, 28.5, 32.5, 40.5],
        ["MFCC mean", "MFCC std", "Formant", "Phổ", "LPC"],
    )
    ax.set(
        xlim=(0.5, 46.5),
        ylabel="Giá trị z-score",
        title=f"Spectral {kind}: {file_id} và kết quả hạng 1",
    )
    ax.legend()
    fig.savefig(OUTPUT / f"t4_{kind}.png", dpi=150)
    plt.close(fig)

    print(f"\n{query.name} ({kind}) — {file_id} — {query['speaker_id']}")
    print(ranking.round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=Path, default=Path("data/wav"))
    args = parser.parse_args()

    with closing(db.connect()) as conn:
        ids, matrix = load_matrix(conn)
        names = db.feature_names(conn, FEATURE_SET)
        norm_params = db.load_norm_params(conn, FEATURE_SET)
        meta = db.read_df(
            conn,
            "SELECT file_id, speaker_id, region_accent, language FROM audio_files",
        ).set_index("file_id")

    queries = pd.read_csv("data/query_set.csv").set_index("query_id")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for query_id in ("Q41", "Q11"):
        run_query(queries.loc[query_id], args.wav_dir, ids, matrix, meta, names, norm_params)


if __name__ == "__main__":
    main()