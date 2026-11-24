"""Trích 46 đặc trưng phổ cho thành viên A."""

import librosa
import numpy as np
import parselmouth

from common.preprocess import SR, preprocess

FEATURE_SET = "spectral"
N_FFT = 2048
HOP_LENGTH = 512


def extract(wav_path, *, preprocessed=False):
    """Trả về 46 đặc trưng. Dùng preprocessed=True cho WAV đã chuẩn hóa."""
    if preprocessed:
        y, sr = librosa.load(wav_path, sr=None, mono=False)
        if sr != SR or y.ndim != 1:
            raise ValueError("WAV đã chuẩn hóa phải là âm thanh mono 16 kHz.")
    else:
        y, sr = preprocess(wav_path)

    # 26 giá trị MFCC gồm 13 trung bình và 13 độ lệch chuẩn.
    spectrum = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mel = librosa.feature.melspectrogram(S=spectrum**2, sr=sr, n_fft=N_FFT)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)

    # Tính trung bình của 4 formant F1 đến F4.
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    formant = sound.to_formant_burg(
        max_number_of_formants=5.0,
        maximum_formant=5000.0,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )
    times = formant.xs()
    formant_means = []
    for number in range(1, 5):
        values = np.array([formant.get_value_at_time(number, t) for t in times])
        defined = values[np.isfinite(values)]
        if defined.size == 0:
            raise ValueError(f"Không đo được formant F{number} từ file âm thanh.")
        formant_means.append(defined.mean())

    # 4 đặc trưng hình dạng phổ.
    changes = np.diff(spectrum, axis=1)
    flux = np.linalg.norm(changes, axis=0).mean() if changes.shape[1] else 0.0
    spectral_shape = [
        librosa.feature.spectral_centroid(S=spectrum, sr=sr, n_fft=N_FFT).mean(),
        librosa.feature.spectral_bandwidth(S=spectrum, sr=sr, n_fft=N_FFT).mean(),
        librosa.feature.spectral_rolloff(S=spectrum, sr=sr, n_fft=N_FFT).mean(),
        flux,
    ]

    # 12 hệ số LPC, bỏ hệ số đầu luôn bằng 1.
    lpc = librosa.lpc(y, order=12)[1:]
    features = np.concatenate(
        [mfcc.mean(axis=1), mfcc.std(axis=1), formant_means, spectral_shape, lpc],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(features)):
        raise ValueError("File âm thanh tạo ra đặc trưng phổ không hữu hạn.")
    return features


def load_matrix(conn):
    from common import db

    return db.load_matrix(conn, FEATURE_SET)


def search(qvec, ids, M, k=5):
    raise NotImplementedError  # Làm ở T3.