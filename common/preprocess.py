"""Tien xu ly file am thanh MOI giong het quy trinh da chuan hoa 1000 file (giai doan 1).

Tham so suy ra tu 1000 file da chuan hoa: WAV PCM16, mono, 16 kHz, cat lang top_db=25,
chuan hoa am luong -23 LUFS (pyloudnorm). Moi file moi (file test, file tu thu) PHAI di qua ham nay
truoc khi trich dac trung.
"""
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln

SR = 16000
TOP_DB = 25
TARGET_LUFS = -23.0
MIN_LUFS_SEC = 0.5      # file ngan hon: pyloudnorm khong do duoc, dung RMS tuong ung
PEAK_LIMIT = 0.99       # khong de bi clip khi tang am luong


def preprocess(path_or_array, sr=None):
    """Tra ve (y, 16000): y la mang float32 mono da cat lang va chuan hoa am luong.

    path_or_array: duong dan file (wav/mp3/m4a... librosa doc duoc) hoac mang numpy kem sr goc.
    """
    if isinstance(path_or_array, (str, bytes)) or hasattr(path_or_array, "__fspath__"):
        y, _ = librosa.load(path_or_array, sr=SR, mono=True)
    else:
        y = np.asarray(path_or_array, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr and sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = y.astype(np.float32)

    trimmed, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(trimmed) > int(0.1 * SR):          # tranh cat het file qua ngan
        y = trimmed
    if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
        raise ValueError("File rong hoac toan im lang")

    if len(y) / SR >= MIN_LUFS_SEC:
        loud = pyln.Meter(SR).integrated_loudness(y.astype(np.float64))
        gain_db = TARGET_LUFS - loud
    else:                                     # RMS -22.9 dB tuong duong ~ -23 LUFS tren du lieu cu
        rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)
        gain_db = -22.97 - rms
    y = y * (10 ** (gain_db / 20))
    peak = np.max(np.abs(y))
    if peak > PEAK_LIMIT:
        y = y * (PEAK_LIMIT / peak)
    return y.astype(np.float32), SR


def preprocess_file(src, dst):
    """Chuan hoa file src va luu thanh WAV PCM16 tai dst."""
    y, sr = preprocess(src)
    sf.write(dst, y, sr, subtype="PCM_16")
    return dst


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("Cach dung: python -m common.preprocess <file_vao> <file_ra.wav>")
    preprocess_file(sys.argv[1], sys.argv[2])
    print("Da luu", sys.argv[2])
