# -*- coding: utf-8 -*-
"""Bản sao của LLM_N3 - Streamlit UI Production"""

import os
import re
import html
import unicodedata
import warnings
import tempfile
import json
import difflib
from dataclasses import dataclass
from datetime import datetime
import pytz

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import streamlit as st

warnings.filterwarnings("ignore")

# rapidfuzz giúp so khớp từ khóa "mờ" khi OCR đọc sai vài ký tự (không bắt buộc)
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Tự động dùng GPU nếu có
try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False

# --- Tên loại tài liệu ---
DT_CCCD = "Căn cước công dân"
DT_GXNDC = "Giấy xác nhận địa chỉ"
DT_KYC = "Form KYC khách hàng tổ chức"
DT_INVALID = "⚠️ KHÔNG HỢP LỆ"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DOCX_EXTS = {".docx"}

# --- Ngưỡng phân loại ---
CLS_MIN_SCORE = 0.30     # điểm tối thiểu để công nhận là 1 loại tài liệu
CLS_MIN_MARGIN = 0.10    # chênh lệch tối thiểu giữa loại cao nhất và loại thứ hai

COLUMNS = ["Tên file", "Loại tài liệu", "Tên trường", "Giá trị trích xuất"]

# ====================================================================
# CELL 3 | Tiện ích văn bản tiếng Việt
# ====================================================================
def strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt nhưng GIỮ NGUYÊN độ dài chuỗi (để map vị trí ký tự về văn bản gốc)."""
    out = []
    for ch in s:
        if ch == "đ":
            out.append("d")
        elif ch == "Đ":
            out.append("D")
        else:
            base = "".join(c for c in unicodedata.normalize("NFD", ch)
                           if unicodedata.category(c) != "Mn")
            out.append(base if len(base) == 1 else ch)
    return "".join(out)


def norm(s: str) -> str:
    """Chuẩn hóa để so khớp từ khóa: bỏ dấu + chữ thường."""
    return strip_accents(s).lower()


@dataclass
class Line:
    """Một dòng văn bản: nội dung + danh sách (start, end, conf) của từng khối OCR."""
    text: str
    words: list

    def conf_span(self, a: int, b: int) -> float:
        """Độ tin cậy OCR trung bình của các khối chữ nằm trong đoạn [a, b)."""
        cs = [c for (s, e, c) in self.words if s < b and e > a]
        if not cs:
            cs = [c for (_, _, c) in self.words] or [0.5]
        return float(np.mean(cs))


def lines_to_text(lines) -> str:
    return "\n".join(l.text for l in lines)

# ====================================================================
# CELL 4 | Đọc tài liệu: OCR (ảnh) & DOCX
# ====================================================================
_OCR_READER = None

def get_ocr_reader():
    """Khởi tạo EasyOCR một lần duy nhất (tiếng Việt + tiếng Anh)."""
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        _OCR_READER = easyocr.Reader(["vi", "en"], gpu=USE_GPU, verbose=False)
    return _OCR_READER


def preprocess_image(path: str) -> np.ndarray:
    """Xoay đúng chiều EXIF, chuẩn hóa kích thước ảnh để OCR ổn định."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest < 1400:
        scale = 1600 / longest          # ảnh nhỏ -> phóng to
    elif longest > 2400:
        scale = 2000 / longest          # ảnh quá lớn -> thu nhỏ
    else:
        scale = 1.0
    if scale != 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def ocr_image_to_lines(path: str):
    """OCR ảnh -> gom các khối chữ có cùng độ cao (trục Y) thành từng dòng, sắp xếp trái -> phải."""
    reader = get_ocr_reader()
    results = reader.readtext(preprocess_image(path), detail=1, paragraph=False,
                              width_ths=0.5, add_margin=0.1)
    items = []
    for bbox, text, conf in results:
        text = unicodedata.normalize("NFC", text).strip()
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append(dict(x0=min(xs), yc=(min(ys) + max(ys)) / 2,
                          h=max(ys) - min(ys), text=text, conf=float(conf)))
    items.sort(key=lambda i: i["yc"])

    # Gom dòng: khối nào lệch tâm Y nhỏ hơn 0.6 chiều cao chữ thì thuộc cùng dòng
    rows = []
    for it in items:
        if rows:
            row_yc = np.mean([x["yc"] for x in rows[-1]])
            row_h = np.median([x["h"] for x in rows[-1]])
            if abs(it["yc"] - row_yc) <= 0.6 * max(row_h, 1):
                rows[-1].append(it)
                continue
        rows.append([it])

    lines = []
    for row in rows:
        row.sort(key=lambda i: i["x0"])
        text, words = "", []
        for it in row:
            if text:
                text += " "
            words.append((len(text), len(text) + len(it["text"]), it["conf"]))
            text += it["text"]
        lines.append(Line(text, words))
    return lines


def read_docx_lines(path: str):
    """Đọc DOCX theo đúng thứ tự: đoạn văn -> 1 dòng, mỗi hàng của bảng -> 1 dòng (các ô nối bằng ' | ')."""
    doc = Document(path)
    lines = []

    def add(t: str):
        t = unicodedata.normalize("NFC", re.sub(r"\s+", " ", t)).strip()
        if t:
            lines.append(Line(t, [(0, len(t), 0.99)]))   # văn bản gốc -> độ tin cậy ~99%

    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            add(Paragraph(child, doc).text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells, seen = [], set()
                for c in row.cells:
                    if c._tc in seen:          # bỏ ô bị gộp (merged) lặp lại
                        continue
                    seen.add(c._tc)
                    ct = re.sub(r"\s+", " ", " ".join(p.text for p in c.paragraphs)).strip()
                    if ct:
                        cells.append(ct)
                add(" | ".join(cells))
    return lines


def read_document(path: str):
    """Trả về (danh sách dòng, is_image)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return ocr_image_to_lines(path), True
    if ext in DOCX_EXTS:
        return read_docx_lines(path), False
    raise ValueError(f"Định dạng '{ext}' chưa được hỗ trợ (chỉ nhận ảnh JPG/PNG và DOCX).")

# ====================================================================
# CELL 5 | Phân loại tài liệu
# ====================================================================
KEYWORDS = {
    DT_CCCD: [
        ("can cuoc cong dan", 3.0), ("cong hoa xa hoi chu nghia viet nam", 1.0), ("doc lap", 0.5),
        ("ho va ten", 1.5), (r"re:ngay\W{0,3}thang\W{0,3}nam\W{0,3}sinh", 2.0),
        ("gioi tinh", 1.5), ("quoc tich", 1.5), ("que quan", 1.5),
        ("noi thuong tru", 2.0), ("co gia tri den", 1.5), (r"re:(?<!\d)\d{12}(?!\d)", 1.0),
    ],
    DT_GXNDC: [
        ("giay xac nhan dia chi", 3.0), ("kinh gui", 1.0), ("ubnd", 1.0), ("toi ten la", 1.5),
        ("nguyen quan", 1.5), ("noi dang ky ho khau", 2.0), ("hien cu tru tai", 2.0),
        ("dia chi cu", 1.5), (r"re:cmnd|cccd", 1.0), ("cong hoa xa hoi chu nghia viet nam", 1.0),
        ("doc lap", 0.5), ("xac nhan", 1.0),
    ],
    DT_KYC: [
        ("thu thap thong tin kyc", 3.0), ("khach hang to chuc", 2.0), ("mau bieu 01b", 2.0),
        ("kyc", 1.5), ("techcombank", 1.0), ("msdn", 1.5), ("dia chi dat tru so", 2.0),
        ("dia chi hoat dong kinh doanh", 1.5), ("loai hinh doanh nghiep", 1.5),
        ("muc dich va ban chat", 2.0), ("san pham dich vu su dung", 1.5), ("thong tin chung", 0.5),
    ],
}


def _keyword_hit(text_n: str, kw: str) -> bool:
    """Kiểm tra từ khóa xuất hiện: regex / khớp chính xác / khớp mờ (chịu lỗi OCR)."""
    if kw.startswith("re:"):
        return re.search(kw[3:], text_n) is not None
    if kw in text_n:
        return True
    if len(kw) >= 10:                                    # chỉ so khớp mờ với cụm đủ dài
        if HAS_RAPIDFUZZ:
            return fuzz.partial_ratio(kw, text_n) >= 88
        return any(difflib.SequenceMatcher(None, kw, l).ratio() >= 0.88 for l in text_n.split("\n"))
    return False


def classify_document(lines):
    """Trả về (loại tài liệu hoặc None, điểm 0-1, dict điểm từng loại)."""
    text_n = norm(lines_to_text(lines))
    scores = {}
    for dt, kws in KEYWORDS.items():
        total = sum(w for _, w in kws)
        got = sum(w for kw, w in kws if _keyword_hit(text_n, kw))
        scores[dt] = got / total
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 >= CLS_MIN_SCORE and (s1 - s2) >= CLS_MIN_MARGIN:
        return best, s1, scores
    return None, s1, scores        # không đủ chắc chắn -> gắn cờ không hợp lệ

# ====================================================================
# CELL 6 | Đặc tả trường cần trích xuất
# ====================================================================
@dataclass
class FieldSpec:
    name: str                    # tên trường hiển thị
    labels: list                 # regex nhãn (đã bỏ dấu, chữ thường), ưu tiên từ trái sang phải
    kind: str                    # kiểu dữ liệu: dùng để chuẩn hóa + kiểm tra định dạng
    multiline: bool = False      # giá trị có thể xuống dòng (địa chỉ trên ảnh)
    fallback: str = ""           # chiến lược dự phòng khi không thấy nhãn
    phrase: str = ""             # cụm nhãn không dấu, dùng để khớp MỜ khi OCR đọc sai nhãn (chỉ áp dụng cho ảnh)

C = r"\s*[:;]?"                # dấu hai chấm tùy chọn sau nhãn
STAR = r"\s*(?:\(\s*\*?\s*\))?"  # dấu (*) tùy chọn

FIELD_SPECS = {
    DT_CCCD: [
        FieldSpec("Số CCCD", [r"(?<![a-z])so\s*[:;.]\s*(?=[\s\dolidsb|]{3})"], "id_cccd", fallback="cccd12"),
        FieldSpec("Họ và tên", [r"ho\s*va\s*ten" + C], "name", fallback="name_upper", phrase="ho va ten"),
        FieldSpec("Ngày sinh", [r"ngay\s*[,.]?\s*thang\s*[,.]?\s*nam\s*sinh" + C, r"ngay\s*sinh" + C], "date", fallback="first_date", phrase="ngay thang nam sinh"),
        FieldSpec("Giới tính", [r"gioi\s*tinh" + C], "gender", fallback="gender", phrase="gioi tinh"),
        FieldSpec("Quốc tịch", [r"quoc\s*tich" + C], "nationality", fallback="nationality", phrase="quoc tich"),
        FieldSpec("Quê quán", [r"que\s*quan" + C], "address", multiline=True, phrase="que quan"),
        FieldSpec("Nơi thường trú", [r"noi\s*thuong\s*tru" + C], "address", multiline=True, phrase="noi thuong tru"),
        FieldSpec("Ngày hết hạn", [r"co\s*gia\s*tri\s*den" + C, r"ngay\s*het\s*han" + C], "date", fallback="last_date", phrase="co gia tri den"),
    ],
    DT_GXNDC: [
        FieldSpec("Họ và tên", [r"toi\s*ten\s*la" + C, r"ho\s*va\s*ten" + C], "name"),
        FieldSpec("Năm sinh", [r"\bnam\s*sinh" + C, r"ngay\s*sinh" + C], "year_or_date"),
        FieldSpec("Giới tính", [r"gioi\s*tinh" + C], "gender"),
        FieldSpec("Nguyên quán", [r"nguyen\s*quan" + C, r"que\s*quan" + C], "address", multiline=True),
        FieldSpec("Số CMND/CCCD", [r"\b(?:so\s*)?(?:cmnd|cccd|cmt)(?:\s*/\s*(?:cmnd|cccd))?\s*(?:so\s*)?[:;]?"], "id_any"),
        FieldSpec("Nơi đăng ký hộ khẩu", [r"noi\s*dang\s*ky\s*ho\s*khau(?:\s*thuong\s*tru)?" + C], "address", multiline=True),
        FieldSpec("Hiện cư trú tại", [r"hien\s*cu\s*tru(?:\s*tai)?" + C, r"cho\s*o\s*hien\s*nay" + C], "address", multiline=True),
        FieldSpec("Địa chỉ cũ", [r"dia\s*chi\s*cu" + C], "address", multiline=True),
    ],
    DT_KYC: [
        FieldSpec("Tên khách hàng", [r"ten\s*khach\s*hang" + STAR + r"\s*:", r"ten\s*khach\s*hang\s*\(\s*bao\s*gom[^)]*\)"], "name_org"),
        FieldSpec("Số ID", [r"so\s*id" + STAR + C], "id_any"),
        FieldSpec("Đơn vị", [r"don\s*vi\s*:"], "org"),
        FieldSpec("Sản phẩm dịch vụ", [r"san\s*pham\s*dich\s*vu(?:\s*su\s*dung)?\s*:"], "text"),
        FieldSpec("MST/Mã giấy phép", [r"msdn\s*,?\s*mst(?:\s*,?\s*so\s*giay\s*phep(?:\s*thanh\s*lap)?)?" + C], "tax"),
        FieldSpec("Địa chỉ trụ sở", [r"dia\s*chi\s*(?:dat\s*)?tru\s*so(?:\s*chinh)?" + C], "address", multiline=True),
        FieldSpec("Địa chỉ kinh doanh", [r"dia\s*chi\s*hoat\s*dong\s*kinh\s*doanh(?:\s*/\s*dia\s*chi\s*gui\s*thu)?" + C], "address", multiline=True),
        FieldSpec("Thông tin liên hệ", [r"thong\s*tin\s*lien\s*he(?:\s*\([^)]*\))?" + C], "contact", multiline=True),
        FieldSpec("Loại hình doanh nghiệp", [r"loai\s*hinh\s*doanh\s*nghiep" + STAR + C], "business_type"),
        FieldSpec("Mục đích quan hệ", [r"muc\s*dich\s*va\s*ban\s*chat(?:\s*thiet\s*lap\s*quan\s*he)?" + STAR + C], "text", multiline=True),
    ],
}

# Các nhãn "chặn" (không trích xuất nhưng dùng để cắt giá trị đúng chỗ)
STOP_LABELS = {
    DT_CCCD: [],
    DT_GXNDC: [r"cap\s*ngay", r"noi\s*cap", r"kinh\s*gui", r"nay\s*toi\s*lam\s*don",
               r"xin\s*chan\s*thanh", r"xac\s*nhan\s*cua", r"nguoi\s*lam\s*don"],
    DT_KYC: [r"ngay\s*cap\s*nhat", r"website", r"linh\s*vuc\s*hoat\s*dong\s*kinh\s*doanh",
             r"ngay\s*thanh\s*lap\s*doanh\s*nghiep", r"dang\s*ky\s*lan\s*dau", r"nguoi\s*tai",
             r"ma\s*hieu", r"ngay\s*tai"],
}

# ====================================================================
# CELL 7 | Chuẩn hóa giá trị
# ====================================================================
DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[/\\.-]\s*(\d{1,2})\s*[/\\.-]\s*(\d{4})(?!\d)")
YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
                           "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2
