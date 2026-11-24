from contextlib import closing
from functools import partial
from pathlib import Path

import pandas as pd

from common import db
from common.evaluate import evaluate
from tracks.track_c_spectral import FEATURE_SET, load_matrix, search


def main():
    with closing(db.connect()) as conn:
        ids, M = load_matrix(conn)
        norm_params = db.load_norm_params(conn, FEATURE_SET)
        meta = db.read_df(
            conn,
            "SELECT file_id, speaker_id, region_accent, language FROM audio_files",
        )

    queries = pd.read_csv("data/query_set.csv")
    rows = []

    for metric in ("euclidean", "cosine", "mahalanobis"):
        search_fn = partial(search, metric=metric, norm_params=norm_params)
        result = evaluate(search_fn, ids, M, meta, queries, split="dev")
        seen = result[result["kind"] == "seen"]
        unseen = result[result["kind"] == "unseen"]

        rows.append({
            "metric": metric,
            "p5": seen["p5"].mean(),
            "map5": seen["ap5"].mean(),
            "top1": seen["top1"].mean(),
            "group5": unseen["group5"].mean(),
            "lang5": unseen["lang5"].mean(),
        })

    summary = pd.DataFrame(rows).sort_values(
        ["p5", "map5", "top1"], ascending=False
    )
    output = Path("report/spectral/t3_metrics_dev.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, float_format="%.6f")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()