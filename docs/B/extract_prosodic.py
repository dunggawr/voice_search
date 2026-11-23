"""Trich 10 dac trung prosodic / chat giong. Usage: python extract_prosodic.py <wav_dirs...> -o features_prosodic.csv"""
import sys, glob, os, argparse, warnings
import numpy as np, pandas as pd, parselmouth, librosa, soundfile as sf
from parselmouth.praat import call
from scipy.signal import find_peaks
from multiprocessing import Pool
warnings.filterwarnings("ignore")
COLS = ["f0_mean","f0_std","f0_min","f0_max","energy_mean","energy_std",
        "speaking_rate","jitter_local","shimmer_local","hnr_mean","zcr_mean"]

def feat(path):
    fid = os.path.basename(path)[:-4]
    out = {"file_id": fid}; err = []
    y, sr = sf.read(path); y = y.astype(float)
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    dur = len(y)/sr
    try:  # F0
        pitch = snd.to_pitch_ac(pitch_floor=75, pitch_ceiling=300)
        f0 = pitch.selected_array["frequency"]; f0 = f0[f0 > 0]
        if len(f0) >= 3:
            out.update(f0_mean=f0.mean(), f0_std=f0.std(), f0_min=f0.min(), f0_max=f0.max())
        else: err.append("f0")
    except Exception as e: err.append("f0")
    try:  # energy (dB) + speaking rate (energy peaks, language independent)
        it = snd.to_intensity(minimum_pitch=75, time_step=0.01)
        v = it.values[0]
        out.update(energy_mean=v.mean(), energy_std=v.std())
        thr = np.percentile(v, 50) if len(v) > 5 else v.mean()
        pk, _ = find_peaks(v, height=thr, prominence=2.0, distance=8)  # >=80ms apart, dip>=2dB
        out["speaking_rate"] = len(pk)/dur
    except Exception: err.append("energy")
    try:  # jitter / shimmer
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 300)
        out["jitter_local"] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)*100
        out["shimmer_local"] = call([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception: err.append("jit_shim")
    try:  # HNR
        h = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
        hv = h.values[0]; hv = hv[hv > -200]
        if len(hv): out["hnr_mean"] = hv.mean()
        else: err.append("hnr")
    except Exception: err.append("hnr")
    out["zcr_mean"] = float(librosa.feature.zero_crossing_rate(y.astype(np.float32))[0].mean())
    out["_err"] = ",".join(err)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("-o", default="features_prosodic.csv")
    ap.add_argument("--limit", type=int); a = ap.parse_args()
    files = sorted(p for d in a.dirs for p in glob.glob(os.path.join(d, "*.wav")))
    if a.limit: files = files[::max(1, len(files)//a.limit)][:a.limit]
    with Pool() as pool: rows = pool.map(feat, files, chunksize=10)
    df = pd.DataFrame(rows).sort_values("file_id")
    df[["file_id"]+COLS].to_csv(a.o, index=False)
    df[df._err != ""][["file_id","_err"]].to_csv(a.o.replace(".csv","_errors.csv"), index=False)
    print(len(df), "files; NaN per col:\n", df[COLS].isna().sum())
