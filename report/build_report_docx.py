"""Sinh file Word bao cao hoan chinh cho Yeu cau 3, 4, 5.

Chay:
    .venv/bin/python report/build_report_docx.py
Dau ra:
    report/Bao_cao_BTL_Nhom_1_Giai_doan_2.docx
"""
import json
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT_DOCX = ROOT / "report" / "Bao_cao_BTL_Nhom_1_Giai_doan_2.docx"

NAVY = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x55, 0x55, 0x55)


def set_cell_shading(cell, hex_color):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def set_table_borders(table, color="D0D7DE"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="1F4E79"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
        f"</w:tblBorders>"
    )
    tblPr.append(borders)


def configure_styles(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(13)
    normal.font.color.rgb = DARK
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(6)

    for level, size in [(1, 16), (2, 14), (3, 13)]:
        st = doc.styles[f"Heading {level}"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = NAVY
        st.paragraph_format.space_before = Pt(12 if level == 1 else 8)
        st.paragraph_format.space_after = Pt(4)
        st.paragraph_format.keep_with_next = True


def add_p(doc, text="", bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=13, color=DARK, space_after=6):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        r = p.add_run(text)
        r.bold = bold
        r.italic = italic
        r.font.name = "Times New Roman"
        r.font.size = Pt(size)
        r.font.color.rgb = color
    return p


def add_bullet(doc, lead, body):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    if lead:
        r1 = p.add_run(lead)
        r1.bold = True
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(13)
    r2 = p.add_run(body)
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(13)
    return p


def add_styled_table(doc, headers, rows, col_widths=None, right_align_cols=None):
    right_align_cols = set(right_align_cols or [])
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl)

    hdr_cells = tbl.rows[0].cells
    for j, h in enumerate(headers):
        cell = hdr_cells[j]
        set_cell_shading(cell, "1F4E79")
        set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j in right_align_cols else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(h))
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for i, row in enumerate(rows):
        cells = tbl.rows[i + 1].cells
        bg = "F2F6FA" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = cells[j]
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j in right_align_cols else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(str(val))
            r.font.name = "Times New Roman"
            r.font.size = Pt(11)
            r.font.color.rgb = DARK

    if col_widths:
        for row in tbl.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Cm(w)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)
    return tbl


def add_figure(doc, rel_path, caption, width_in=6.0):
    img_path = ROOT / rel_path
    if not img_path.exists():
        raise FileNotFoundError(f"Thieu hinh anh: {img_path}")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(img_path), width=Inches(width_in))
    add_p(doc, caption, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=11, color=MUTED, space_after=8)


def build():
    doc = Document()
    configure_styles(doc)

    # --- TRANG BIA ---
    add_p(doc, "HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, color=NAVY, space_after=2)
    add_p(doc, "KHOA CÔNG NGHỆ THÔNG TIN 1", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13, color=NAVY, space_after=24)
    add_p(doc, "BÁO CÁO BÀI TẬP LỚN", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=18, color=NAVY, space_after=6)
    add_p(doc, "HỌC PHẦN: HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN (INT1418)", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13, space_after=12)
    add_p(doc, "ĐỀ TÀI: XÂY DỰNG HỆ CƠ SỞ DỮ LIỆU LƯU TRỮ VÀ TÌM KIẾM GIỌNG NÓI ĐÀN ÔNG", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=14, color=NAVY, space_after=6)
    add_p(doc, "Yêu cầu 3, 4, 5: Trích rút đặc trưng, Hệ quản trị CSDL, Tìm kiếm Top-5, Đánh giá và Demo", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=12, color=MUTED, space_after=24)

    add_styled_table(
        doc,
        ["Mã sinh viên", "Họ và tên", "Nội dung chuyên môn phụ trách"],
        [
            ["B23DCCN138", "Nguyễn Thành Đạt", "Trích rút & tìm kiếm bộ đặc trưng Phổ (46 chiều); Đánh giá tổng hợp; Giao diện ứng dụng demo và kịch bản trình diễn"],
            ["B23DCCN208", "Phạm Việt Dũng", "Trích rút & tìm kiếm bộ đặc trưng Prosodic (11 chiều); Thuật toán hợp nhất điểm 3 bộ đặc trưng (search_fused) và tối ưu trọng số"],
            ["B23DCCN907", "Vũ Khánh Tùng", "Trích rút & tìm kiếm Speaker Embedding (192 chiều, ECAPA-TDNN); Thiết kế kiến trúc, sơ đồ khối và Cơ sở dữ liệu PostgreSQL"],
        ],
        col_widths=[2.8, 3.6, 10.1],
    )

    add_p(doc, "Nhóm: 01   |   Lớp: D23 lớp 4   |   Giảng viên hướng dẫn: TS. Nguyễn Đình Hóa", align=WD_ALIGN_PARAGRAPH.CENTER, size=12, space_after=4)
    add_p(doc, "HÀ NỘI 2026", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, size=13, color=NAVY, space_after=0)
    doc.add_page_break()

    # --- PHAN 3 ---
    doc.add_heading("PHẦN 3. TRÍCH RÚT ĐẶC TRƯNG VÀ XÂY DỰNG CƠ SỞ DỮ LIỆU (YÊU CẦU 3)", level=1)

    doc.add_heading("3.1 Chuẩn hoá đầu vào và giao diện trích rút đặc trưng thống nhất", level=2)
    add_p(
        doc,
        "Để đảm bảo file âm thanh mới (file kiểm thử hoặc bản thu thực tế) tương thích hoàn toàn với tập 1.000 file đã lưu trữ, "
        "toàn bộ hệ thống sử dụng chung quy trình tiền xử lý tại common/preprocess.py trước khi trích rút đặc trưng. "
        "Cả 3 bộ đặc trưng đều tuân thủ giao diện thống nhất gồm 3 hàm: extract(wav_path) trích vector từ file mới, "
        "load_matrix(conn) đọc ma trận đặc trưng từ CSDL, và search(qvec, ids, M, k=5) trả về Top-5 kết quả theo thứ tự giảm dần độ tương đồng trong [0, 1]."
    )
    add_bullet(doc, "Tiền xử lý âm thanh (common.preprocess.preprocess): ", "đọc bằng librosa.load(sr=16000, mono=True), cắt khoảng lặng hai đầu librosa.effects.trim(top_db=25), chuẩn hoá âm lượng -23 LUFS bằng pyloudnorm (file ngắn hơn 0,5 giây dùng mức RMS -22,97 dB tương đương) và giới hạn đỉnh PEAK_LIMIT = 0,99.")

    doc.add_heading("3.2 Thuật toán trích rút 3 bộ đặc trưng và kiểm chứng lặp lại", level=2)
    add_styled_table(
        doc,
        ["Bộ đặc trưng", "Module cài đặt", "Số chiều & Thuật toán", "Kết quả kiểm tra lặp lại trên 20 file gốc"],
        [
            ["Phổ (Spectral)", "tracks/track_c_spectral.py", "46 chiều: 26 MFCC (13 mean + 13 std), 4 Formant Burg (F1–F4, max 5000 Hz), centroid, bandwidth, rolloff, spectral_flux, 12 LPC", "Khớp công thức ban đầu, toàn bộ 46 chiều hữu hạn trên 1.000 file"],
            ["Prosodic & Chất giọng", "tracks/track_b_prosodic.py", "11 chiều: F0 (mean, std, min, max), Intensity (mean, std), speaking_rate, jitter_local, shimmer_local, hnr_mean, zcr_mean", "Đọc trực tiếp WAV chuẩn hoá: max_abs = 0,00e+00 (khớp 100%); chạy lại tiền xử lý: median_rel = 0,0000, max_rel = 0,1821"],
            ["Speaker Embedding", "tracks/track_a_embedding.py", "192 chiều: mạng nơ-ron ECAPA-TDNN pretrained (speechbrain/spkrec-ecapa-voxceleb), nạp mô hình một lần (lazy load)", "Cosine similarity trung bình 0,9996 (thấp nhất 0,9943), sai số tương đối trung vị 0,0004"],
        ],
        col_widths=[2.7, 3.8, 5.2, 4.3],
    )

    doc.add_heading("3.3 Thiết kế và nghiệm thu Cơ sở dữ liệu PostgreSQL", level=2)
    add_p(
        doc,
        "Hệ thống sử dụng hệ quản trị CSDL quan hệ PostgreSQL 16 (đóng gói trong container Docker voice_search_db, schema voice, định nghĩa tại schema.sql). "
        "Các tệp âm thanh WAV gốc (1.000 file, ~80 MB) được lưu trên đĩa tại data/wav/ và liên kết thông qua cột audio_files.path để phát lại trực tiếp trên ứng dụng. "
        "Dữ liệu được nạp theo lô qua các script tracks/load_t2_spectral.py, tracks/load_prosodic.py và tracks/load_t2_embedding.py."
    )
    add_styled_table(
        doc,
        ["Tên bảng", "Khóa & Kiểu lưu trữ", "Số dòng thực tế", "Vai trò"],
        [
            ["voice.speakers", "speaker_id (PK)", "84", "Thông tin người nói, vùng/accent (Nam/UK/US), ngôn ngữ, số file"],
            ["voice.audio_files", "file_id (PK), speaker_id (FK)", "1.000", "Siêu dữ liệu 1.000 file WAV 16 kHz và đường dẫn file"],
            ["voice.feat_spectral", "file_id (PK, FK), 46 cột DOUBLE PRECISION", "1.000", "46 đặc trưng phổ"],
            ["voice.feat_prosodic", "file_id (PK, FK), 11 cột DOUBLE PRECISION", "1.000", "11 đặc trưng prosodic & chất giọng"],
            ["voice.feat_embedding", "file_id (PK, FK), vector REAL[] CHECK(cardinality=192)", "1.000", "Vector embedding 192 chiều ECAPA-TDNN (kiểu mảng native của PostgreSQL)"],
            ["voice.norm_params", "(feature_set, feature_name) PK", "57", "Mean và std của 46 cột spectral + 11 cột prosodic phục vụ chuẩn hoá z-score"],
        ],
        col_widths=[3.6, 4.8, 2.3, 5.3],
        right_align_cols={2},
    )
    add_p(
        doc,
        "Kiểm tra toàn vẹn liên bảng (truy vấn LEFT JOIN trong sql/examples.sql): trả về 0 dòng thiếu đặc trưng trên cả 1.000 file. "
        "Thống kê F0 trung bình theo nhóm trực tiếp bằng SQL từ CSDL: UK (333 file) = 112,66 Hz; US (300 file) = 118,10 Hz; Nam tiếng Việt (367 file) = 144,31 Hz."
    )

    # --- PHAN 4 ---
    doc.add_heading("PHẦN 4. HỆ THỐNG TÌM KIẾM TƯƠNG ĐỒNG TOP-5 VÀ ĐÁNH GIÁ (YÊU CẦU 4)", level=1)

    doc.add_heading("4.1 Sơ đồ khối và quy trình truy vấn (Yêu cầu 4a)", level=2)
    add_p(
        doc,
        "Hệ thống tìm kiếm được tổ chức theo 5 phân tầng xử lý tuần tự: "
        "(1) Tiếp nhận file âm thanh truy vấn (file tải lên từ người dùng hoặc bản thu mới); "
        "(2) Tiền xử lý chuẩn hoá về 16 kHz mono, cắt lặng và cân bằng âm lượng -23 LUFS; "
        "(3) Trích rút song song 3 vector đặc trưng (Embedding 192-d, Spectral 46-d, Prosodic 11-d); "
        "(4) Đọc ma trận 1.000 file và tham số chuẩn hoá z-score từ PostgreSQL; "
        "(5) Tính điểm tương đồng từng bộ, hợp nhất có trọng số trong tracks/fusion.py và xuất Top-5 kết quả kèm trình phát âm thanh."
    )
    add_figure(doc, "report/block_diagram.png", "Hình 1. Sơ đồ khối tổng thể và luồng xử lý truy vấn của hệ thống tìm kiếm giọng nói.", width_in=6.2)

    doc.add_heading("4.2 Lựa chọn độ đo và hợp nhất điểm đa đặc trưng", level=2)
    add_p(
        doc,
        "Bộ truy vấn kiểm thử cố định (data/query_set.csv) gồm 50 truy vấn được chia thành 20 truy vấn phát triển (dev: 10 seen, 10 unseen) "
        "để lựa chọn độ đo và tối ưu trọng số, và 30 truy vấn đánh giá độc lập (test: 20 seen, 10 unseen). "
        "Với nhóm seen, hệ thống chỉ loại bỏ chính file truy vấn; với nhóm unseen, toàn bộ file của người nói truy vấn bị loại khỏi CSDL để mô phỏng người nói mới. "
        "Kết quả thử nghiệm các độ đo trên tập dev (sau chuẩn hoá z-score đối với Prosodic và Spectral; L2-normalize đối với Embedding):"
    )
    add_styled_table(
        doc,
        ["Bộ đặc trưng", "Độ đo thử nghiệm", "Seen Top-1 (dev)", "Seen P@5 (dev)", "Seen MAP@5 (dev)", "Unseen group@5 (dev)", "Lựa chọn"],
        [
            ["Prosodic", "Mahalanobis (reg 1e-3·I)", "0,100", "0,120", "0,0797", "0,880", "Chọn"],
            ["Prosodic", "Euclid", "0,100", "0,100", "0,0573", "0,880", "—"],
            ["Prosodic", "Cosine", "0,100", "0,080", "0,0700", "0,840", "—"],
            ["Spectral", "Cosine", "0,700", "0,460", "0,4170", "0,900", "Chọn"],
            ["Spectral", "Mahalanobis", "0,700", "0,460", "0,4112", "0,780", "—"],
            ["Spectral", "Euclid", "0,600", "0,420", "0,4000", "0,900", "—"],
            ["Embedding", "Cosine (L2-norm)", "1,000", "0,980", "0,9950", "0,800", "Chọn"],
        ],
        col_widths=[2.5, 3.4, 2.0, 1.9, 2.2, 2.2, 1.8],
        right_align_cols={2, 3, 4, 5},
    )
    add_p(
        doc,
        "Hợp nhất điểm 3 bộ đặc trưng (tracks/fusion.py): Chuẩn hoá điểm tương đồng của 3 bộ về cùng thang [0, 1] "
        "(Prosodic: Mahalanobis 1/(1+d); Spectral: z-score + Cosine 1/(1+d); Embedding: Cosine (cos+1)/2) "
        "và kết hợp tuyến tính sim_fused = w_pros·sim_pros + w_spec·sim_spec + w_emb·sim_emb. "
        "Tìm kiếm lưới (grid search) bước 0,05 trên tập dev (231 tổ hợp, hoàn toàn tách biệt khỏi tập test) cho bộ trọng số tối ưu: "
        "w_prosodic = 0,05, w_spectral = 0,15, w_embedding = 0,80 (đạt trên dev: Seen MAP@5 = 1,0000 và Unseen group@5 = 0,980)."
    )

    doc.add_heading("4.3 Minh hoạ kết quả trung gian khi truy vấn (Yêu cầu 4b)", level=2)
    with open(ROOT / "report" / "B" / "t4_prosodic_intermediate.json", encoding="utf8") as f:
        t4_b = json.load(f)
    q21 = t4_b["Q21"]
    q01 = t4_b["Q01"]
    pros_cols = list(q21["raw_vector"].keys())
    add_p(
        doc,
        "Bảng dưới trình bày đầy đủ 11 giá trị đặc trưng thô và sau chuẩn hoá z-score của bộ Prosodic cho 2 truy vấn minh hoạ: "
        "Q21 (nhóm seen: A_0744, SPK_VIVOSDEV03, vùng Nam) và Q01 (nhóm unseen: A_0652, SPK_VIVOSDEV01, vùng Nam):"
    )
    t4_rows = [
        [c, f"{q21['raw_vector'][c]:.3f}", f"{q21['zscore_vector'][c]:+.3f}", f"{q01['raw_vector'][c]:.3f}", f"{q01['zscore_vector'][c]:+.3f}"]
        for c in pros_cols
    ]
    add_styled_table(
        doc,
        ["Đặc trưng", "Q21 (seen) Gốc", "Q21 Z-score", "Q01 (unseen) Gốc", "Q01 Z-score"],
        t4_rows,
        col_widths=[3.6, 3.1, 3.1, 3.1, 3.1],
        right_align_cols={1, 2, 3, 4},
    )
    add_p(
        doc,
        "Kết quả xếp hạng Top-5 minh hoạ (bộ Prosodic): Với truy vấn Q21 (seen), cả 5/5 file trả về đều đúng người nói SPK_VIVOSDEV03 "
        "(#1 A_0722 sim=0,490, d=1,040 đến #5 A_0718 sim=0,408). Với truy vấn Q01 (unseen), toàn bộ file của SPK_VIVOSDEV01 bị loại, "
        "Top-5 trả về 5/5 giọng Nam tiếng Việt (#1 A_0797 SPK_VIVOSDEV05 sim=0,422, d=1,369)."
    )
    add_figure(doc, "report/B/fig_t4_prosodic_seen_compare.png", "Hình 2. Bộ Prosodic — So sánh vector z-score và Top-5 truy vấn seen Q21 (A_0744).", width_in=6.0)
    add_figure(doc, "report/B/fig_t4_prosodic_unseen_compare.png", "Hình 3. Bộ Prosodic — So sánh vector z-score và Top-5 truy vấn unseen Q01 (A_0652).", width_in=6.0)
    add_figure(doc, "report/A/t4_seen.png", "Hình 4. Bộ Phổ (Spectral) — Kết quả trung gian truy vấn seen Q41 (A_0997).", width_in=5.8)
    add_figure(doc, "report/A/t4_unseen.png", "Hình 5. Bộ Phổ (Spectral) — Kết quả trung gian truy vấn unseen Q11 (A_0851).", width_in=5.8)
    add_figure(doc, "report/C/t4_seen.png", "Hình 6. Bộ Embedding — Xếp hạng Top-5 truy vấn seen Q41 (5/5 đúng người nói).", width_in=5.6)
    add_figure(doc, "report/C/t4_unseen.png", "Hình 7. Bộ Embedding — Xếp hạng Top-5 truy vấn unseen Q11 (5/5 cùng vùng Nam).", width_in=5.6)

    doc.add_heading("4.4 Đánh giá kết quả và phân tích lỗi (Yêu cầu 4c)", level=2)
    comp = pd.read_csv(ROOT / "report" / "B" / "eval_fusion_comparison.csv")
    comp_rows = []
    name_map = {
        "prosodic": "Prosodic (11 chiều)",
        "spectral": "Spectral (46 chiều)",
        "embedding": "Embedding (192 chiều)",
        "fused": "Hợp nhất (0,05 / 0,15 / 0,80)",
    }
    for r in comp.itertuples():
        comp_rows.append([
            r.split,
            name_map.get(r.system, r.system),
            f"{r.seen_top1:.3f}",
            f"{r.seen_p5:.3f}",
            f"{r.seen_map5:.4f}",
            f"{r.unseen_group5:.3f}",
            f"{r.unseen_lang5:.3f}",
        ])
    add_styled_table(
        doc,
        ["Tập", "Hệ thống", "Seen Top-1", "Seen P@5", "Seen MAP@5", "Unseen group@5", "Unseen lang@5"],
        comp_rows,
        col_widths=[1.4, 4.4, 2.0, 1.9, 2.1, 2.1, 2.1],
        right_align_cols={2, 3, 4, 5, 6},
    )
    add_figure(doc, "report/B/fig_2b_fusion_comparison.png", "Hình 8. So sánh hiệu năng 3 bộ đặc trưng đơn lẻ và hệ thống hợp nhất trên tập test.", width_in=6.0)

    add_bullet(doc, "Nhận xét định lượng: ", "Trên tập test (20 truy vấn seen, 10 truy vấn unseen), hệ thống hợp nhất đạt tuyệt đối 1,000 ở cả Top-1, P@5 và MAP@5 cho nhóm seen (ngang bằng Embedding), đồng thời nâng tỷ lệ kết quả cùng vùng của nhóm unseen (group@5) từ 0,820 (Embedding đơn lẻ) lên 0,960 và cùng ngôn ngữ (lang@5) lên 1,000 nhờ bổ sung thông tin vùng/ngữ điệu từ Spectral (0,15) và Prosodic (0,05).")
    add_bullet(doc, "Phân tích lỗi ở từng bộ đơn lẻ: ", "Bộ Prosodic có 15/20 truy vấn seen sai ở hạng 1, tập trung vào các file ngắn 1,0–1,5 giây (jitter, shimmer, speaking_rate kém ổn định) và các giọng US/UK có dải F0 chồng lấn (113–118 Hz). Bộ Spectral đạt 12/20 seen Top-1 nhưng đôi khi nhầm giữa các người nói cùng vùng (Q25, Q37). Bộ Embedding đạt 20/20 seen Top-1 nhưng giảm khả năng giữ vùng địa lý ở Q05, Q06 (UK, group@5 = 0,2 dù lang@5 = 1,0).")
    add_bullet(doc, "Thử nghiệm file mới ngoài tập 1.000 file (kiểm tra lệch kênh thu): ", "Trên 2 bản thu iPhone tiếng Việt thực tế, bộ Spectral bị lệch sang giọng tiếng Anh do khác biệt thiết bị và phòng thu. Trên 2 file giọng nam tiếng Anh ngoài tập gốc (new_B_1.wav 3,62 giây và new_B_2.wav 4,18 giây; f0_mean = 106,4 Hz), cả Prosodic và Spectral đều trả về 5/5 giọng tiếng Anh, nhưng Prosodic phân tách new_B_1 (zcr=0,1875) vào UK và new_B_2 (zcr=0,2169) vào US — khẳng định zcr và hnr nhạy với kênh thu hơn F0.")

    add_styled_table(
        doc,
        ["Nội dung đối soát độc lập", "Phạm vi kiểm chứng", "Kết quả"],
        [
            ["Kiểm chứng hệ thống Prosodic", "Chạy lại từ đầu trên PostgreSQL với tập test (30 truy vấn)", "Trùng khớp 100% (30/30 dòng top-5 giống hệt)"],
            ["Kiểm chứng hệ thống Spectral", "Chạy lại toàn bộ pipeline trích rút, CSDL và đánh giá tập dev/test", "Trùng khớp 100% số liệu (seen 0,600/0,430/0,367; unseen 0,980/0,980)"],
            ["Kiểm chứng hệ thống Embedding", "Đối chiếu top-5 từ vector trên 50/50 truy vấn (dev + test)", "Khớp 50/50 truy vấn; trên máy Windows bị Windows Application Control chặn DLL PyTorch (WinError 4551) khi trích file mới"],
        ],
        col_widths=[4.2, 5.6, 6.2],
    )

    # --- PHAN 5 ---
    doc.add_heading("PHẦN 5. ỨNG DỤNG DEMO VÀ KỊCH BẢN TRÌNH DIỄN (YÊU CẦU 5)", level=1)
    add_p(
        doc,
        "Ứng dụng minh hoạ được xây dựng bằng Streamlit tại app/main.py, kết nối trực tiếp CSDL PostgreSQL (tự động chuyển sang đọc CSV khi chạy offline) gồm 4 tab chức năng:"
    )
    add_bullet(doc, "Tab 1 — Embedding (app/tab_embedding.py): ", "tải file mới hoặc chọn mẫu, trích vector 192 chiều bằng ECAPA-TDNN, trả Top-5 theo Cosine và phát âm thanh.")
    add_bullet(doc, "Tab 2 — Spectral (app/tab_spectral.py): ", "trích 46 đặc trưng phổ, chuẩn hoá z-score, hiển thị Top-5 kèm bảng điểm và trình phát âm thanh.")
    add_bullet(doc, "Tab 3 — Prosodic (app/tab_b_prosodic.py): ", "chọn độ đo (Mahalanobis/Euclid/Cosine), xem bảng 11 đặc trưng thô + z-score, Top-5 kèm audio và biểu đồ so sánh với kết quả hạng 1.")
    add_bullet(doc, "Tab 4 — Hợp nhất (Fused) (app/main.py): ", "hiển thị bảng đánh giá 4 hệ thống, biểu đồ so sánh và thử nghiệm tìm kiếm hợp nhất (0,05 / 0,15 / 0,80) kèm phát âm thanh truy vấn và Top-5.")
    add_p(doc, "Kịch bản trình diễn (5 bước, khởi chạy bằng lệnh: .venv/bin/python -m streamlit run app/main.py):", bold=True, space_after=3)
    add_bullet(doc, "Bước 1: ", "Khởi động ứng dụng, kiểm tra kết nối PostgreSQL schema voice.")
    add_bullet(doc, "Bước 2: ", "Ở Tab Spectral và Tab Prosodic, tải lên một file âm thanh mới ngoài tập 1.000 file (ví dụ bản thu điện thoại hoặc new_B_1.wav), mở bảng đặc trưng và nghe đối chiếu Top-5.")
    add_bullet(doc, "Bước 3: ", "Ở Tab Embedding, thực hiện truy vấn trên máy có hỗ trợ PyTorch để minh hoạ độ chính xác nhận diện người nói.")
    add_bullet(doc, "Bước 4: ", "Ở Tab Fused, chọn mẫu A_0001, quan sát kết quả hạng 1 (A_0005) và nghe lại âm thanh đối chiếu.")
    add_bullet(doc, "Bước 5: ", "Mở bảng tổng hợp trên tập test để minh hoạ phương pháp hợp nhất giữ 100% Seen Top-1 và nâng Unseen group@5 từ 82% lên 96%.")

    # --- PHAN 6 ---
    doc.add_heading("PHẦN 6. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", level=1)
    add_bullet(doc, "Kết quả đạt được: ", "Hoàn thiện trọn vẹn quy trình từ trích rút 3 bộ đặc trưng (46, 11, 192 chiều) cho file âm thanh mới, lưu trữ và kiểm tra toàn vẹn trên PostgreSQL 16 (84 người nói, 1.000 file, 57 tham số z-score, 0 dòng khuyết), tìm kiếm Top-5, hợp nhất đa đặc trưng tối ưu trên tập dev (0,05 / 0,15 / 0,80 đạt 100% Seen Top-1/P@5/MAP@5 và 96% Unseen group@5 trên test), cùng giao diện demo Streamlit 4 tab.")
    add_bullet(doc, "Hạn chế còn tồn tại: ", "Các đặc trưng thủ công nhạy với đoạn âm thanh ngắn (< 1,5 giây) và sự khác biệt kênh thu (micro điện thoại so với TIMIT/VCTK/VIVOS); bộ dữ liệu thu âm thực tế còn ít và cần mở rộng thêm đa dạng thiết bị, môi trường ghi âm.")
    add_bullet(doc, "Hướng phát triển: ", "Tích hợp extension pgvector (chỉ mục HNSW/IVFFlat) khi mở rộng quy mô lên hàng trăm nghìn file; áp dụng ngưỡng phân loại người nói mới (θ = 0,40) và hỗ trợ trích rút trực tiếp đồng thời cả 3 bộ đặc trưng cho file tải lên ngay trong Tab Hợp nhất.")

    doc.save(str(OUT_DOCX))
    print(f"Saved {OUT_DOCX}")


if __name__ == "__main__":
    build()
