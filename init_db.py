"""Tao schema "voice" trong PostgreSQL tu schema.sql va nap 2 bang chung: speakers, audio_files (tu metadata_master.csv).
Cac bang dac trung (feat_*, norm_params) de TRONG: moi nguoi tu nap bang cua minh o viec T2.

Cach chay (database voice_search phai co san: xem docker-compose.yml):
    python init_db.py data/metadata_master.csv [thu_muc_wav] [--reset]
  --reset : XOA schema "voice" (moi bang trong do) roi tao lai. Khong co --reset ma schema da ton tai thi dung lai,
            de khong lam mat du lieu dac trung dang nap.
Ket noi qua bien moi truong DATABASE_URL (mac dinh xem common/db.py).
"""
import argparse
import os
import sys
import pandas as pd
from psycopg2.extras import execute_values
from common import db

COLS = ["file_id", "speaker_id", "path", "duration_sec", "sample_rate", "age_group", "region_accent",
        "language", "recording_device", "environment", "voice_note", "collector"]


def rows(df):
    """DataFrame -> list cac tuple Python thuan (NaN thanh NULL) de psycopg2 nap duoc."""
    return df.astype(object).where(df.notna(), None).values.tolist()


def main(meta_csv, wav_dir="data/wav", reset=False):
    conn = db.connect()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name=%s", (db.SCHEMA,))
    if cur.fetchone():
        if not reset:
            sys.exit(f'Schema "{db.SCHEMA}" da ton tai. Chay lai voi --reset neu that su muon XOA va tao lai tu dau.')
        cur.execute(f"DROP SCHEMA {db.SCHEMA} CASCADE")
    cur.execute(open("schema.sql", encoding="utf8").read())
    m = pd.read_csv(meta_csv)
    spk = m.groupby("speaker_id").agg(region_accent=("region_accent", "first"), language=("language", "first"),
                                       n_files=("file_id", "count")).reset_index()
    execute_values(cur, "INSERT INTO speakers (speaker_id, region_accent, language, n_files) VALUES %s", rows(spk))
    m["path"] = m["file_id"].map(lambda f: os.path.join(wav_dir, f + ".wav"))
    execute_values(cur, f"INSERT INTO audio_files ({', '.join(COLS)}) VALUES %s", rows(m[COLS]))
    conn.commit()
    cur.execute("SELECT (SELECT COUNT(*) FROM audio_files), (SELECT COUNT(*) FROM speakers)")
    n_f, n_s = cur.fetchone()
    print(f'Da tao schema "{db.SCHEMA}": {n_f} file, {n_s} speaker. Cac bang feat_* dang trong (viec T2 cua tung tuyen).')


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meta_csv")
    ap.add_argument("wav_dir", nargs="?", default="data/wav")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    main(a.meta_csv, a.wav_dir, a.reset)
