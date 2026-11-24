"""Nạp đặc trưng Spectral giai đoạn 1 vào PostgreSQL."""

from contextlib import closing

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from common import db

CSV_PATH = "docs/C/features_spectral.csv"
FEATURE_SET = "spectral"


def main():
    df = pd.read_csv(CSV_PATH, dtype={"file_id": str})

    # PostgreSQL sẽ commit khi thành công và rollback nếu có lỗi.
    with closing(db.connect()) as conn, conn:
        names = db.feature_names(conn, FEATURE_SET)
        columns = ["file_id", *names]

        if len(names) != 46 or set(df.columns) != set(columns):
            raise ValueError("Các cột trong CSV không khớp bảng feat_spectral.")

        if len(df) != 1000 or df["file_id"].isna().any() or not df["file_id"].is_unique:
            raise ValueError("CSV phải có đúng 1.000 file_id hợp lệ, không trùng.")

        matrix = df[names].to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("CSV chứa giá trị Spectral thiếu hoặc không hữu hạn.")

        rows = [
            (file_id, *values)
            for file_id, values in zip(df["file_id"], matrix.tolist())
        ]

        updates = ", ".join(f"{name} = EXCLUDED.{name}" for name in names)
        statement = (
            f"INSERT INTO feat_spectral ({', '.join(columns)}) VALUES %s "
            f"ON CONFLICT (file_id) DO UPDATE SET {updates}"
        )

        with conn.cursor() as cur:
            execute_values(cur, statement, rows, page_size=100)

        db.save_norm_params(conn, FEATURE_SET, names, matrix)

        ids, loaded = db.load_matrix(conn, FEATURE_SET)
        if loaded.shape != matrix.shape:
            raise ValueError(f"Ma trận đọc lại có kích thước sai: {loaded.shape}")

        means, _ = db.load_norm_params(conn, FEATURE_SET)

    print(f"Nạp thành công {len(ids)} file vào voice.feat_spectral.")
    print(f"Ma trận: {loaded.shape}")
    print(f"Số cặp mean/std: {len(means)}")


if __name__ == "__main__":
    main()