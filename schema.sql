-- schema.sql: hop dong CSDL chung cua nhom (PostgreSQL 14 tro len). KHONG tu y doi ten bang/cot.
-- Moi tuyen chi GHI vao bang cua minh: A -> feat_spectral, B -> feat_prosodic, C -> feat_embedding.
-- Toan bo bang nam trong schema "voice" (khong dung vao schema public). init_db.py chay file nay.
CREATE SCHEMA voice;
SET search_path TO voice;

CREATE TABLE speakers (
    speaker_id    TEXT PRIMARY KEY,
    region_accent TEXT,                 -- Nam / UK / US
    language      TEXT,                 -- vi / en
    n_files       INTEGER
);

CREATE TABLE audio_files (
    file_id          TEXT PRIMARY KEY,  -- A_0001 ... A_1000
    speaker_id       TEXT NOT NULL REFERENCES speakers(speaker_id),
    path             TEXT,              -- duong dan toi file wav da chuan hoa (data/wav/...)
    duration_sec     DOUBLE PRECISION,
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
    f0_mean DOUBLE PRECISION,
    f0_std DOUBLE PRECISION,
    f0_min DOUBLE PRECISION,
    f0_max DOUBLE PRECISION,
    energy_mean DOUBLE PRECISION,
    energy_std DOUBLE PRECISION,
    speaking_rate DOUBLE PRECISION,
    jitter_local DOUBLE PRECISION,
    shimmer_local DOUBLE PRECISION,
    hnr_mean DOUBLE PRECISION,
    zcr_mean DOUBLE PRECISION
);

CREATE TABLE feat_spectral (
    file_id TEXT PRIMARY KEY REFERENCES audio_files(file_id),
    mfcc_mean_1 DOUBLE PRECISION,
    mfcc_mean_2 DOUBLE PRECISION,
    mfcc_mean_3 DOUBLE PRECISION,
    mfcc_mean_4 DOUBLE PRECISION,
    mfcc_mean_5 DOUBLE PRECISION,
    mfcc_mean_6 DOUBLE PRECISION,
    mfcc_mean_7 DOUBLE PRECISION,
    mfcc_mean_8 DOUBLE PRECISION,
    mfcc_mean_9 DOUBLE PRECISION,
    mfcc_mean_10 DOUBLE PRECISION,
    mfcc_mean_11 DOUBLE PRECISION,
    mfcc_mean_12 DOUBLE PRECISION,
    mfcc_mean_13 DOUBLE PRECISION,
    mfcc_std_1 DOUBLE PRECISION,
    mfcc_std_2 DOUBLE PRECISION,
    mfcc_std_3 DOUBLE PRECISION,
    mfcc_std_4 DOUBLE PRECISION,
    mfcc_std_5 DOUBLE PRECISION,
    mfcc_std_6 DOUBLE PRECISION,
    mfcc_std_7 DOUBLE PRECISION,
    mfcc_std_8 DOUBLE PRECISION,
    mfcc_std_9 DOUBLE PRECISION,
    mfcc_std_10 DOUBLE PRECISION,
    mfcc_std_11 DOUBLE PRECISION,
    mfcc_std_12 DOUBLE PRECISION,
    mfcc_std_13 DOUBLE PRECISION,
    formant_f1 DOUBLE PRECISION,
    formant_f2 DOUBLE PRECISION,
    formant_f3 DOUBLE PRECISION,
    formant_f4 DOUBLE PRECISION,
    spectral_centroid DOUBLE PRECISION,
    spectral_bandwidth DOUBLE PRECISION,
    spectral_rolloff DOUBLE PRECISION,
    spectral_flux DOUBLE PRECISION,
    lpc_1 DOUBLE PRECISION,
    lpc_2 DOUBLE PRECISION,
    lpc_3 DOUBLE PRECISION,
    lpc_4 DOUBLE PRECISION,
    lpc_5 DOUBLE PRECISION,
    lpc_6 DOUBLE PRECISION,
    lpc_7 DOUBLE PRECISION,
    lpc_8 DOUBLE PRECISION,
    lpc_9 DOUBLE PRECISION,
    lpc_10 DOUBLE PRECISION,
    lpc_11 DOUBLE PRECISION,
    lpc_12 DOUBLE PRECISION
);

-- Embedding ECAPA-TDNN: mang REAL[] dung 192 phan tu (khong can cai them extension).
CREATE TABLE feat_embedding (
    file_id TEXT PRIMARY KEY REFERENCES audio_files(file_id),
    vector  REAL[] NOT NULL CHECK (cardinality(vector) = 192)
);

-- Tham so z-score tinh tren toan bo 1000 file, dung chung khi truy van (feature_set: prosodic | spectral | embedding)
CREATE TABLE norm_params (
    feature_set  TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    mean         DOUBLE PRECISION NOT NULL,
    std          DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (feature_set, feature_name)
);
