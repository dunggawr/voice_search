-- Cau SQL mau cho viec T2 (chay: psql "$DATABASE_URL" -f sql/examples.sql). Schema "voice".
SET search_path TO voice;

-- 1. So file moi speaker (5 speaker nhieu file nhat)
SELECT speaker_id, region_accent, COUNT(*) AS so_file
FROM audio_files GROUP BY speaker_id, region_accent ORDER BY so_file DESC LIMIT 5;

-- 2. Dac trung prosodic cua mot file, kem thong tin file
SELECT a.file_id, a.speaker_id, a.duration_sec, p.f0_mean, p.hnr_mean, p.jitter_local
FROM audio_files a JOIN feat_prosodic p USING (file_id) WHERE a.file_id = 'A_0001';

-- 3. F0 trung binh theo nhom vung/accent
SELECT a.region_accent, COUNT(*) AS so_file, ROUND(AVG(p.f0_mean)::numeric, 1) AS f0_tb
FROM audio_files a JOIN feat_prosodic p USING (file_id) GROUP BY a.region_accent ORDER BY f0_tb;

-- 4. 5 file co f0_mean gan nhat voi file A_0001 (tim kiem don gian ngay trong CSDL, chi minh hoa)
SELECT p.file_id, p.f0_mean, ABS(p.f0_mean - q.f0_mean) AS chenh_lech
FROM feat_prosodic p, (SELECT f0_mean FROM feat_prosodic WHERE file_id = 'A_0001') q
WHERE p.file_id <> 'A_0001' ORDER BY chenh_lech LIMIT 5;

-- 5. Embedding: 5 chieu dau cua mot file (mang REAL[] cua PostgreSQL)
SELECT file_id, vector[1:5] AS vec_5_dau, cardinality(vector) AS so_chieu FROM feat_embedding WHERE file_id = 'A_0001';

-- 6. Kiem tra toan ven: file nao thieu dac trung o bat ky bang nao (ket qua mong doi: 0 dong khi da nap du)
SELECT a.file_id FROM audio_files a
LEFT JOIN feat_prosodic p USING (file_id) LEFT JOIN feat_spectral s USING (file_id) LEFT JOIN feat_embedding e USING (file_id)
WHERE p.file_id IS NULL OR s.file_id IS NULL OR e.file_id IS NULL;
