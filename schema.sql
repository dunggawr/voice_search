-- schema.sql: hop dong CSDL chung cua nhom (SQLite). KHONG tu y doi ten bang/cot.
-- Moi tuyen chi GHI vao bang cua minh: A -> feat_embedding, B -> feat_prosodic, C -> feat_spectral.
PRAGMA foreign_keys = ON;

CREATE TABLE speakers (
    speaker_id    TEXT PRIMARY KEY,
    region_accent TEXT,            -- Nam / UK / US
    language      TEXT,            -- vi / en
    n_files       INTEGER
);

CREATE TABLE audio_files (
    file_id          TEXT PRIMARY KEY,   -- A_0001 ... A_1000
    speaker_id       TEXT NOT NULL REFERENCES speakers(speaker_id),
    path             TEXT,               -- duong dan toi file wav da chuan hoa (tuong doi so voi data/wav)
    duration_sec     REAL,
    sample_rate      INTEGER,
    age_group        TEXT,
    region_accent    TEXT,
    language         TEXT,
    recording_device TEXT,
    environment      TEXT,
    voice_note       TEXT,
    collector        TEXT
);
CREATE INDEX idx_audio_speaker ON audio_files(speaker_id);

CREATE TABLE feat_prosodic (
    file_id TEXT PRIMARY KEY REFERENCES audio_files(file_id),
    f0_mean REAL,
    f0_std REAL,
    f0_min REAL,
    f0_max REAL,
    energy_mean REAL,
    energy_std REAL,
    speaking_rate REAL,
    jitter_local REAL,
    shimmer_local REAL,
    hnr_mean REAL,
    zcr_mean REAL
);

CREATE TABLE feat_spectral (
    file_id TEXT PRIMARY KEY REFERENCES audio_files(file_id),
    mfcc_mean_1 REAL,
    mfcc_mean_2 REAL,
    mfcc_mean_3 REAL,
    mfcc_mean_4 REAL,
    mfcc_mean_5 REAL,
    mfcc_mean_6 REAL,
    mfcc_mean_7 REAL,
    mfcc_mean_8 REAL,
    mfcc_mean_9 REAL,
    mfcc_mean_10 REAL,
    mfcc_mean_11 REAL,
    mfcc_mean_12 REAL,
    mfcc_mean_13 REAL,
    mfcc_std_1 REAL,
    mfcc_std_2 REAL,
    mfcc_std_3 REAL,
    mfcc_std_4 REAL,
    mfcc_std_5 REAL,
    mfcc_std_6 REAL,
    mfcc_std_7 REAL,
    mfcc_std_8 REAL,
    mfcc_std_9 REAL,
    mfcc_std_10 REAL,
    mfcc_std_11 REAL,
    mfcc_std_12 REAL,
    mfcc_std_13 REAL,
    formant_f1 REAL,
    formant_f2 REAL,
    formant_f3 REAL,
    formant_f4 REAL,
    spectral_centroid REAL,
    spectral_bandwidth REAL,
    spectral_rolloff REAL,
    spectral_flux REAL,
    lpc_1 REAL,
    lpc_2 REAL,
    lpc_3 REAL,
    lpc_4 REAL,
    lpc_5 REAL,
    lpc_6 REAL,
    lpc_7 REAL,
    lpc_8 REAL,
    lpc_9 REAL,
    lpc_10 REAL,
    lpc_11 REAL,
    lpc_12 REAL
);

-- Embedding: vector float32 little-endian, dim = 192 (ECAPA-TDNN), luu dang BLOB (192*4 = 768 byte)
CREATE TABLE feat_embedding (
    file_id TEXT PRIMARY KEY REFERENCES audio_files(file_id),
    dim     INTEGER NOT NULL DEFAULT 192,
    vector  BLOB NOT NULL
);

-- Tham so z-score tinh tren toan bo 1000 file, dung chung cho truy van (feature_set: prosodic | spectral | embedding)
CREATE TABLE norm_params (
    feature_set  TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    mean         REAL NOT NULL,
    std          REAL NOT NULL,
    PRIMARY KEY (feature_set, feature_name)
);
