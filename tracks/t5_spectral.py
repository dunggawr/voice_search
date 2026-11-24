import argparse
from contextlib import closing
from functools import partial
from pathlib import Path

import pandas as pd

from common import db
from common.evaluate import evaluate, summarize
from tracks.track_c_spectral import FEATURE_SET, extract, load_matrix, search

OUTPUT = Path("report/spectral")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="*", type=Path)
    args = parser.parse_args()

    with closing(db.connect()) as conn:
        ids, matrix = load_matrix(conn)
        norm_params = db.load_norm_params(conn, FEATURE_SET)
        meta = db.read_df(
            conn, "SELECT file_id, speaker_id, region_accent, language FROM audio_files",
        )

    search_fn = partial(search, norm_params=norm_params)
    queries = pd.read_csv("data/query_set.csv")
    result = evaluate(search_fn, ids, matrix, meta, queries, split="test")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "t5_test_results.csv", index=False, float_format="%.6f")
    print(summarize(result))

    listening_path = OUTPUT / "t5_listening.csv"
    if not listening_path.exists():
        listening = result.groupby("kind", sort=False).head(5)[
            ["query_id", "kind", "file_id", "top5"]
        ].copy()
        for column in ("diem_dat", "diem_dung", "diem_tung", "ghi_chu"):
            listening[column] = ""
        listening.to_csv(listening_path, index=False)

    if args.recordings:
        rows = []
        for wav in args.recordings:
            vector = extract(wav)
            hits = search_fn(vector, ids, matrix, k=5)
            for rank, (file_id, similarity, distance) in enumerate(hits, start=1):
                rows.append((wav.name, rank, file_id, similarity, distance))

        phone = pd.DataFrame(
            rows, columns=["query_file", "rank", "file_id", "similarity", "distance"]
        ).merge(meta, on="file_id", validate="many_to_one")
        phone.to_csv(OUTPUT / "t5_phone_top5.csv", index=False, float_format="%.6f")
        print("\nTop 5 cho file tự thu")
        print(phone.round(4).to_string(index=False))


if __name__ == "__main__":
    main()