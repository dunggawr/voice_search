"""Tao voice.db tu schema.sql va nap 2 bang chung: speakers, audio_files (tu metadata_master.csv).
Cac bang dac trung (feat_*, norm_params) de TRONG: moi nguoi tu nap bang cua minh o viec T2.

Cach chay:  python init_db.py data/metadata_master.csv [thu_muc_wav]
"""
import os
import sys
import sqlite3
import pandas as pd


def main(meta_csv, wav_dir="data/wav", db_path="db/voice.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.executescript(open("schema.sql", encoding="utf8").read())
    m = pd.read_csv(meta_csv)
    spk = m.groupby("speaker_id").agg(region_accent=("region_accent", "first"), language=("language", "first"),
                                       n_files=("file_id", "count")).reset_index()
    spk.to_sql("speakers", conn, if_exists="append", index=False)
    m["path"] = m["file_id"].map(lambda f: os.path.join(wav_dir, f + ".wav"))
    cols = ["file_id", "speaker_id", "path", "duration_sec", "sample_rate", "age_group", "region_accent",
            "language", "recording_device", "environment", "voice_note", "collector"]
    m[cols].to_sql("audio_files", conn, if_exists="append", index=False)
    conn.commit()
    n_f = conn.execute("SELECT COUNT(*) FROM audio_files").fetchone()[0]
    n_s = conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
    print(f"Da tao {db_path}: {n_f} file, {n_s} speaker. Cac bang feat_* dang trong (viec T2 cua tung tuyen).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
