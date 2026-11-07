"""Tao bo truy van kiem thu CO DINH (seed 42) -> data/query_set.csv. Chay 1 lan, ca nhom dung chung file ket qua.

Moi dong: query_id, split (dev|test), kind (seen|unseen), file_id, speaker_id, region_accent, exclude_ids
  - seen  : speaker co trong CSDL; chi loai chinh file truy van khoi CSDL khi tim (exclude_ids = file_id)
  - unseen: coi nhu speaker chua co; loai TOAN BO file cua speaker do khoi CSDL khi tim
  - dev : dung de chon do do / trong so hop nhat diem (tuyen B). test : chi dung de bao cao ket qua, khong de chinh tham so.
Cach chay:  python make_query_set.py data/metadata_master.csv
"""
import sys
import numpy as np
import pandas as pd

SEED = 42
MIN_DUR = 1.0                    # truy van khong ngan hon 1 giay
PLAN = {                         # so truy van moi nhom (Nam, UK, US)
    ("test", "unseen"): {"Nam": 2, "UK": 4, "US": 4},
    ("dev", "unseen"):  {"Nam": 2, "UK": 4, "US": 4},
    ("test", "seen"):   {"Nam": 4, "UK": 8, "US": 8},
    ("dev", "seen"):    {"Nam": 2, "UK": 4, "US": 4},
}


def main(meta_csv, out="data/query_set.csv"):
    m = pd.read_csv(meta_csv)
    rng = np.random.default_rng(SEED)
    cnt = m.groupby("speaker_id").size()
    grp = m.drop_duplicates("speaker_id").set_index("speaker_id")["region_accent"]
    used_unseen, used_seen_spk, used_files, rows = set(), {"test": set(), "dev": set()}, set(), []

    for (split, kind) in [("test", "unseen"), ("dev", "unseen"), ("test", "seen"), ("dev", "seen")]:
        for g, n in PLAN[(split, kind)].items():
            spk_pool = [s for s in cnt.index if grp[s] == g]
            if kind == "unseen":
                spk_pool = [s for s in spk_pool if s not in used_unseen]
            else:   # seen: can >= 3 file de co file cung speaker con lai; speaker khong trung trong cung split; khong la speaker unseen
                spk_pool = [s for s in spk_pool if cnt[s] >= 3 and s not in used_seen_spk[split] and s not in used_unseen]
            chosen = list(rng.choice(sorted(spk_pool), size=n, replace=False))
            for s in chosen:
                cand = m[(m.speaker_id == s) & (m.duration_sec >= MIN_DUR) & (~m.file_id.isin(used_files))]
                f = cand.iloc[int(rng.integers(len(cand)))]
                used_files.add(f.file_id)
                if kind == "unseen":
                    used_unseen.add(s)
                    excl = ";".join(m[m.speaker_id == s].file_id)
                else:
                    used_seen_spk[split].add(s)
                    excl = f.file_id
                rows.append(dict(split=split, kind=kind, file_id=f.file_id, speaker_id=s, region_accent=g,
                                 duration_sec=round(f.duration_sec, 2), exclude_ids=excl))
    q = pd.DataFrame(rows)
    q.insert(0, "query_id", [f"Q{i+1:02d}" for i in range(len(q))])
    q.to_csv(out, index=False)
    print(q.groupby(["split", "kind", "region_accent"]).size().unstack(fill_value=0))
    print(f"Tong {len(q)} truy van -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:2])
