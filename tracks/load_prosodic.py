"""T2 (Tuyen B): Nap 1000 dong dac trung prosodic vao PostgreSQL (voice.feat_prosodic),
luu tham so chuan hoa z-score (voice.norm_params) va chay cac cau SQL kiem tra.

Cach chay:
    python -m tracks.load_prosodic [--csv docs/B/features_prosodic.csv] [--dsn ...]
"""
import argparse
import pandas as pd
from psycopg2.extras import execute_values

from common import db
from tracks.track_b_prosodic import COLS, FEATURE_SET


def main(csv_path="docs/B/features_prosodic.csv", dsn=None):
    df = pd.read_csv(csv_path).sort_values("file_id").reset_index(drop=True)
    cols = ["file_id"] + COLS
    rows = df[cols].astype(object).where(df[cols].notna(), None).values.tolist()

    conn = db.connect(dsn)
    try:
        with conn.cursor() as cur:
            update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLS)
            sql = (
                f"INSERT INTO feat_prosodic ({', '.join(cols)}) VALUES %s "
                f"ON CONFLICT (file_id) DO UPDATE SET {update_sql}"
            )
            execute_values(cur, sql, rows)

        M = df[COLS].to_numpy(dtype=float)
        db.save_norm_params(conn, FEATURE_SET, COLS, M)
        conn.commit()

        # Kiem tra SQL T2
        cnt_df = db.read_df(conn, "SELECT COUNT(*) AS so_dong FROM feat_prosodic")
        sample_df = db.read_df(
            conn,
            "SELECT a.file_id, a.speaker_id, a.region_accent, p.f0_mean, p.hnr_mean, p.jitter_local "
            "FROM audio_files a JOIN feat_prosodic p USING (file_id) WHERE a.file_id = 'A_0001'",
        )
        spk_df = db.read_df(
            conn,
            "SELECT a.speaker_id, a.region_accent, COUNT(*) AS so_file "
            "FROM audio_files a JOIN feat_prosodic p USING (file_id) "
            "GROUP BY a.speaker_id, a.region_accent ORDER BY so_file DESC, a.speaker_id LIMIT 5",
        )
        region_df = db.read_df(
            conn,
            "SELECT a.region_accent, COUNT(*) AS so_file, ROUND(AVG(p.f0_mean)::numeric, 2) AS f0_tb "
            "FROM audio_files a JOIN feat_prosodic p USING (file_id) "
            "GROUP BY a.region_accent ORDER BY f0_tb",
        )

        ids, M_db = db.load_matrix(conn, FEATURE_SET)
        mean_db, std_db = db.load_norm_params(conn, FEATURE_SET)

        print(f"Da nap feat_prosodic: {int(cnt_df.iloc[0]['so_dong'])} dong; load_matrix shape={M_db.shape}")
        print(f"Da luu norm_params ({FEATURE_SET}): {len(mean_db)} cot (f0_mean={mean_db[0]:.3f} +- {std_db[0]:.3f})")
        print("\n[SQL 1] Dac trung file A_0001:\n", sample_df.to_string(index=False))
        print("\n[SQL 2] Top 5 speaker theo so file:\n", spk_df.to_string(index=False))
        print("\n[SQL 3] F0 trung binh theo vung:\n", region_df.to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="docs/B/features_prosodic.csv")
    ap.add_argument("--dsn", default=None)
    args = ap.parse_args()
    main(args.csv, args.dsn)
