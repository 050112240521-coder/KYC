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
# Bảng sửa lỗi OCR hay nhầm giữa chữ và số (chỉ áp dụng cho trường số, ảnh chụp)
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1",
                           "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6"})
PROVINCES = ["An Giang", "Bà Rịa - Vũng Tàu", "Bắc Giang", "Bắc Kạn", "Bạc Liêu", "Bắc Ninh", "Bến Tre",
             "Bình Định", "Bình Dương", "Bình Phước", "Bình Thuận", "Cà Mau", "Cần Thơ", "Cao Bằng",
             "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên", "Đồng Nai", "Đồng Tháp", "Gia Lai", "Hà Giang",
             "Hà Nam", "Hà Nội", "Hà Tĩnh", "Hải Dương", "Hải Phòng", "Hậu Giang", "Hòa Bình", "Hưng Yên",
             "Khánh Hòa", "Kiên Giang", "Kon Tum", "Lai Châu", "Lâm Đồng", "Lạng Sơn", "Lào Cai", "Long An",
             "Nam Định", "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên", "Quảng Bình", "Quảng Nam",
             "Quảng Ngãi", "Quảng Ninh", "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình",
             "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Huế", "Tiền Giang", "Hồ Chí Minh", "Trà Vinh",
             "Tuyên Quang", "Vĩnh Long", "Vĩnh Phúc", "Yên Bái"]
_PROV_N = [(norm(p), p) for p in PROVINCES]

LABEL_WORDS = ("co", "gia", "tri", "den", "cogiatri", "giatriden", "hieu", "luc")


def strip_trailing_junk(t: str) -> str:
    """Loại phần 'rác' dính ở cuối địa chỉ khi OCR gom nhầm dòng 'Có giá trị đến: dd/mm/yyyy' (nhãn có thể bị
    đọc thành 'C6gtri €én'): cắt từ ngày trở đi, rồi bỏ các mảnh nhãn méo ngay trước ngày."""
    m = DATE_RE.search(t)
    had_date = m is not None
    if m:
        t = t[:m.start()]
    toks = t.split()
    while toks:
        tok = toks[-1]
        tn = norm(tok).strip(",.:;")
        symbol = re.search(r"[€$@#%&*+=<>{}\[\]\\|~^`]", tok) is not None
        mixed = re.search(r"\d", tok) and re.search(r"[^\W\d_]", tok)        # lẫn chữ và số: 'C6gtri'
        labelish = had_date and tn.isalpha() and len(tn) <= 9 and max(_fuzzy_ratio(tn, w) for w in LABEL_WORDS) >= 70
        if symbol or (had_date and (mixed or labelish)):
            toks.pop()
        else:
            break
    return " ".join(toks).strip(" ,:;")


def repair_address(t: str) -> str:
    """Sửa lỗi OCR phổ biến trong địa chỉ:
    - 'Thộn 4' -> 'Thôn 4', 'Xom' -> 'Xóm' (chỉ ở đầu mỗi đoạn, không động tới tên riêng)
    - đoạn CUỐI (thường là tỉnh/thành phố) được so khớp với danh sách 63 tỉnh để sửa dấu/chính tả."""
    segs = [x.strip() for x in t.split(",") if x.strip()]
    fixed = []
    for seg in segs:
        m = re.match(r"(\S+)(.*)", seg)
        if m and norm(m.group(1)) in ("thon", "xom") and m.group(1) not in ("Thôn", "Xóm"):
            seg = ("Thôn" if norm(m.group(1)) == "thon" else "Xóm") + m.group(2)
        fixed.append(seg)
    if fixed:
        last = fixed[-1]
        m = re.match(r"(?i)^((?:tinh|tỉnh|thanh pho|thành phố|tp\.?)\s+)?(.+)$", last)
        prefix, core = (m.group(1) or ""), m.group(2)
        best = max(_PROV_N, key=lambda pn: _fuzzy_ratio(norm(core), pn[0]))
        if _fuzzy_ratio(norm(core), best[0]) >= 88:
            fixed[-1] = prefix + best[1]
    return ", ".join(fixed)


ADMIN_KW = re.compile(r"\b(phuong|xa|thi tran|quan|huyen|thanh pho|tp|tinh|thon|xom|ap|khu|kcn|"
                     r"duong|pho|to dan pho|kdc|so \d+|lo [a-z0-9]+)\b")
BUSINESS_TYPES = ["doanh nghiep tu nhan", "cong ty tnhh", "cong ty co phan", "cong ty hop danh",
                 "doanh nghiep nha nuoc", "ho kinh doanh", "hop tac xa", "tnhh", "co phan",
                 "doanh nghiep fdi", "chi nhanh", "van phong dai dien"]


def gender_clean(n: str) -> str:
    """Loại các cụm chứa 'nam' KHÔNG phải giới tính: 'Việt Nam', 'năm sinh', 'ngày tháng năm sinh'."""
    n = re.sub(r"ngay\W{0,3}thang\W{0,3}n\w{1,2}m\W{0,3}sinh", " ", n)
    n = re.sub(r"\bn\w{1,2}m\s*sinh", " ", n)
    return re.sub(r"vi\w{0,2}t?\s*nam", " ", n)


def norm_date(t: str):
    """Tìm và chuẩn hóa ngày về dd/mm/yyyy; trả None nếu không hợp lệ."""
    m = DATE_RE.search(t)
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    try:
        datetime(y, mo, d)
    except ValueError:
        return None
    return f"{d:02d}/{mo:02d}/{y}"


def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^[\s:;|\-–.,]+", "", t)
    t = re.sub(r"[\s|;:\-–]+$", "", t)
    return t.strip(" ,")


def normalize_value(kind: str, raw: str, is_image: bool) -> str:
    """Chuẩn hóa giá trị thô theo kiểu dữ liệu. Trả về '' nếu không nhận diện được."""
    t = clean_text(raw)
    if not t:
        return ""
    if kind in ("id_cccd", "id_any"):
        s = re.sub(r"[\s.\-]", "", t)
        digits = re.sub(r"\D", "", s.translate(DIGIT_FIX) if is_image else s)
        if len(digits) >= 6:
            return digits
        return s if (kind == "id_any" and len(s) >= 4) else ""
    if kind == "date":
        d = norm_date(t)
        if d:
            return d
        return t if len(re.sub(r"\D", "", t)) >= 4 else ""
    if kind == "year_or_date":
        d = norm_date(t)
        if d:
            return d
        m = YEAR_RE.search(t)
        return m.group(1) if m else ""
    if kind == "gender":
        n = gender_clean(norm(t))
        if re.search(r"\bnam\b", n):
            return "Nam"
        if re.search(r"\bnu\b", n):
            return "Nữ"
        return ""
    if kind == "nationality":
        if "viet nam" in norm(t):
            return "Việt Nam"
        return t if re.fullmatch(r"[A-Za-zÀ-ỹ\s]{3,30}", t) else ""
    if kind == "name":
        t = re.sub(r"[^A-Za-zÀ-ỹ\s'.\-]", "", t)
        return re.sub(r"\s+", " ", t).strip(" .-'")
    if kind == "address":
        t = re.sub(r"\s*;\s*", ", ", t)                     # OCR hay đọc dấu phẩy thành ';'
        t = re.sub(r"\s+,", ",", t)
        t = re.sub(r",(\s*,)+", ",", t).strip(" ,")
        if is_image:
            t = strip_trailing_junk(t)
        return repair_address(t)
    if kind == "tax":
        s = re.sub(r"[^\d\-]", "", t.replace(" ", ""))
        return s if len(re.sub(r"\D", "", s)) >= 8 else t
    return t

# ====================================================================
# CELL 8 | Engine trích xuất
# ====================================================================
@dataclass
class Hit:
    field: str      # tên trường (None nếu là nhãn chặn)
    rank: int       # độ ưu tiên của mẫu nhãn
    line: int
    start: int
    end: int


def _fuzzy_ratio(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(a, b)
    return 100 * difflib.SequenceMatcher(None, a, b).ratio()


def fuzzy_label_hits(lines, specs, existing, thr=80):
    """Với trường chưa tìm thấy nhãn bằng regex (OCR đọc sai chữ, vd 'Nvi thường trú'),
    quét cửa sổ k từ liên tiếp và chọn cụm giống nhãn nhất (độ giống >= thr%)."""
    have = {h.field for h in existing if h.field}
    new = []
    for sp in specs:
        if sp.name in have or not sp.phrase:
            continue
        k, best = len(sp.phrase.split()), None
        for li, ln in enumerate(lines):
            n = norm(ln.text)
            toks = [(m.start(), m.end()) for m in re.finditer(r"[^\s:;,.|]+", n)]
            for i in range(0, len(toks) - k + 1):
                a, b = toks[i][0], toks[i + k - 1][1]
                win = re.sub(r"[\s:;,.|]+", " ", n[a:b]).strip()
                r = _fuzzy_ratio(sp.phrase, win)
                if r >= thr and (best is None or r > best[0]):
                    best = (r, li, a, b)
        if best:
            _, li, a, b = best
            b += re.match(r"\s*[:;]?", norm(lines[li].text)[b:]).end()
            new.append(Hit(sp.name, 0, li, a, b))
    return new


def fuzzy_label_hits_pass2(lines, specs, hits):
    """Bước 2 cho nhãn bị méo nặng (vd 'Nci thuang t' thay cho 'Nơi thường trú'):
    neo vào TỪ ĐẶC TRƯNG của nhãn (>=5 ký tự, vd 'thuong'), chỉ xét trong khoảng giữa hai trường
    đã biết vị trí ở hai bên (các trường trên thẻ luôn theo thứ tự cố định)."""
    known = list(hits)
    new = []
    for i, sp in enumerate(specs):
        if any(h.field == sp.name for h in known) or not sp.phrase:
            continue
        ptoks = sp.phrase.split()
        dts = [t for t in ptoks if len(t) >= 5]
        if not dts:
            continue
        prev_lines = [h.line for s2 in specs[:i] for h in known if h.field == s2.name]
        next_lines = [h.line for s2 in specs[i + 1:] for h in known if h.field == s2.name]
        lo = max(prev_lines) if prev_lines else 0
        hi = min(next_lines) if next_lines else len(lines) - 1
        best = None
        for li in range(lo, hi + 1):
            n = norm(lines[li].text)
            toks = [(m.start(), m.end()) for m in re.finditer(r"[^\s:;,.|]+", n)]
            for ti, (a, b) in enumerate(toks[:4]):            # nhãn nằm ở đầu dòng
                w = n[a:b]
                for dt in dts:
                    # độ dài từ phải gần bằng nhau để tránh nhầm 'thon'(Thôn) với 'thuong'
                    if abs(len(w) - len(dt)) <= 1 and _fuzzy_ratio(dt, w) >= 80:
                        if best is None or _fuzzy_ratio(dt, w) > best[0]:
                            best = (_fuzzy_ratio(dt, w), li, toks, ti, dt)
        if best:
            _, li, toks, ti, dt = best
            n = norm(lines[li].text)
            start, end, j = toks[0][0], toks[ti][1], ti + 1
            rest = ptoks[ptoks.index(dt) + 1:]
            for pt in rest:                                    # gộp mảnh nhãn còn sót ('t', 'tr', 'tru'...)
                if j < len(toks):
                    tk = n[toks[j][0]:toks[j][1]]
                    if tk.isalpha() and len(tk) <= len(pt) + 1 and (pt.startswith(tk) or _fuzzy_ratio(pt, tk) >= 70):
                        end = toks[j][1]
                        j += 1
            end += re.match(r"\s*[:;]?", n[end:]).end()
            h = Hit(sp.name, 0, li, start, end)
            new.append(h)
            known.append(h)
    return new


def find_label_hits(lines, specs, stops, is_image=False):
    """Quét toàn bộ văn bản, tìm vị trí mọi nhãn (trường + nhãn chặn), loại bỏ nhãn chồng lấn."""
    hits = []
    for li, ln in enumerate(lines):
        n = norm(ln.text)                               # cùng độ dài với ln.text nên vị trí khớp nhau
        for sp in specs:
            for rank, pat in enumerate(sp.labels):
                for m in re.finditer(pat, n):
                    hits.append(Hit(sp.name, rank, li, m.start(), m.end()))
        for pat in stops:
            for m in re.finditer(pat, n):
                hits.append(Hit(None, 0, li, m.start(), m.end()))
    if is_image:
        hits += fuzzy_label_hits(lines, specs, hits)       # cứu các nhãn bị OCR đọc sai
        hits += fuzzy_label_hits_pass2(lines, specs, hits)  # cứu nhãn bị méo nặng
    hits.sort(key=lambda h: (h.line, h.start, -(h.end - h.start)))
    kept = []
    for h in hits:
        if kept and kept[-1].line == h.line and h.start < kept[-1].end:
            continue
        kept.append(h)
    return kept


def extract_value(lines, kept, hit, spec, is_image):
    """Lấy phần chữ nằm sau nhãn đến nhãn kế tiếp; với ảnh có thể lấy thêm các dòng bên dưới."""
    ln = lines[hit.line]
    same = [h for h in kept if h.line == hit.line and h.start >= hit.end]
    end = same[0].start if same else len(ln.text)
    parts = [ln.text[hit.end:end]]

    if is_image and not same:
        max_extra = 2 if spec.multiline else 0
        if not clean_text(parts[0]):
            max_extra = max(max_extra, 1)        # nhãn đứng một mình -> giá trị ở dòng dưới
        j, extra = hit.line + 1, 0
        while extra < max_extra and j < len(lines):
            nxt = [h for h in kept if h.line == j]
            if nxt:                               # gặp nhãn khác -> lấy phần chữ TRƯỚC nhãn (nếu có) rồi dừng
                pre = lines[j].text[:min(h.start for h in nxt)]
                if clean_text(pre):
                    parts.append(pre)
                break
            parts.append(lines[j].text)
            j += 1
            extra += 1
    # Địa chỉ xuống dòng: nối bằng dấu phẩy nếu dòng trước chưa có; trường khác nối bằng khoảng trắng
    joined = parts[0].strip()
    for p in parts[1:]:
        p = p.strip()
        if not p:
            continue
        sep = " " if (spec.kind != "address" or joined.endswith(",") or not joined) else ", "
        joined += sep + p
    return joined


def locate_in_text(kind, text, is_image):
    """Tìm giá trị theo mẫu (ngày / giới tính / số định danh) trong một dòng bất kỳ."""
    if kind == "date":
        return norm_date(text)
    if kind == "year_or_date":
        return norm_date(text) or (YEAR_RE.search(text).group(1) if YEAR_RE.search(text) else None)
    if kind == "gender":
        m = re.search(r"\b(nam|nu)\b", gender_clean(norm(text)))
        return ("Nam" if m.group(1) == "nam" else "Nữ") if m else None
    if kind in ("id_cccd", "id_any"):
        t = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
        if is_image:
            t = re.sub(r"(?<=[OolI|])\s+(?=[OolI|])", "", t)
        m = re.search(r"(?<!\d)\d{9,12}(?!\d)", t)
        return m.group(0) if m else None
    return None


HEADER_WORDS = ("cong hoa", "can cuoc", "doc lap", "viet nam", "hanh phuc", "techcombank", "mau bieu", "giay xac nhan")

def global_fallback(strategy, lines, is_image):
    """Dự phòng khi không tìm thấy nhãn: dùng mẫu cấu trúc trên toàn văn bản (NER theo quy tắc)."""
    if not strategy:
        return None
    if strategy in ("first_date", "last_date"):
        found = [(i, norm_date(l.text)) for i, l in enumerate(lines) if norm_date(l.text)]
        if strategy == "first_date" and found:
            return found[0][1], lines[found[0][0]]
        if strategy == "last_date" and len(found) >= 2:
            return found[-1][1], lines[found[-1][0]]
        return None
    if strategy == "cccd12":
        for l in lines:
            v = locate_in_text("id_cccd", l.text, is_image)
            if v and len(v) == 12:
                return v, l
        return None
    if strategy == "gender":
        for l in lines:
            # xét từng cụm tách bởi nhãn: chỉ nhận khi phần còn lại đúng là 'Nam'/'Nữ' (vd 'Giới tính: Nữ')
            rest = re.sub(r"gioi\s*tinh\s*[:;]?", " ", gender_clean(norm(l.text)))
            rest = re.sub(r"quoc\s*tich\s*[:;]?", " ", rest).strip(" :;.")
            if rest in ("nam", "nu"):
                return ("Nam" if rest == "nam" else "Nữ"), l
        return None
    if strategy == "nationality":
        for l in lines:
            if "viet nam" in norm(l.text) and not l.text.isupper():
                return "Việt Nam", l
        return None
    if strategy == "name_upper":                 # dòng IN HOA 2-6 từ, không phải tiêu đề
        for l in lines:
            if any(w in norm(l.text) for w in HEADER_WORDS) or re.search(r"\d", l.text):
                continue
            run, best = [], []
            for tk in l.text.split() + [""]:
                if tk and tk.isalpha() and tk.isupper():
                    run.append(tk)
                else:
                    best = run if len(run) > len(best) else best
                    run = []
            if 2 <= len(best) <= 6:
                return " ".join(best), l
        return None
    return None


def extract_fields(lines, doc_type, is_image):
    """Trích xuất toàn bộ trường của một loại tài liệu -> danh sách (tên trường, giá trị)."""
    specs, stops = FIELD_SPECS[doc_type], STOP_LABELS[doc_type]
    kept = find_label_hits(lines, specs, stops, is_image)
    results = []
    for sp in specs:
        value, label_hit = "", None
        cands = sorted([h for h in kept if h.field == sp.name], key=lambda h: (h.rank, h.line, h.start))

        # (1) Tìm theo nhãn
        for h in cands:
            v = normalize_value(sp.kind, extract_value(lines, kept, h, sp, is_image), is_image)
            if v:
                value = v
                break
            label_hit = label_hit or h

        # (2) Tìm ở dòng lân cận nhãn (giá trị bị lệch dòng do OCR) - không áp dụng cho giới tính
        if not value and label_hit is not None and is_image and sp.kind != "gender":
            for j in (label_hit.line - 1, label_hit.line + 1):
                if 0 <= j < len(lines):
                    v = locate_in_text(sp.kind, lines[j].text, is_image)
                    if v:
                        value = v
                        break

        # (3) Dự phòng toàn văn bản
        if not value and sp.fallback:
            fb = global_fallback(sp.fallback, lines, is_image)
            if fb:
                value = normalize_value(sp.kind, fb[0], is_image)

        results.append((sp.name, value))
    return results

# ====================================================================
# CELL 9 | Pipeline -> DataFrame + HTML
# ====================================================================
DEBUG_OCR = {}     # lưu văn bản OCR thô của từng file để kiểm tra khi trích xuất sai


def process_file(path: str):
    """Xử lý 1 file: đọc -> phân loại -> trích xuất. Trả về (các dòng DataFrame, dòng tóm tắt)."""
    fname = os.path.basename(path)
    try:
        lines, is_image = read_document(path)
    except Exception as e:
        return [[fname, DT_INVALID, "-", f"Lỗi đọc file: {e}"]], [fname, DT_INVALID, "Lỗi đọc file"]

    DEBUG_OCR[fname] = "\n".join(f"{i:02d} | {l.text}" for i, l in enumerate(lines))
    doc_type, _, _ = classify_document(lines)
    if doc_type is None:
        return ([[fname, DT_INVALID, "-", "Không thuộc 3 loại tài liệu được hỗ trợ"]],
                [fname, DT_INVALID, "Đã gắn cờ không hợp lệ"])

    rows = [[fname, doc_type, name, val if val else "(không tìm thấy)"]
            for name, val in extract_fields(lines, doc_type, is_image)]
    n_ok = sum(1 for r in rows if r[3] != "(không tìm thấy)")
    return rows, [fname, doc_type, f"Trích xuất {n_ok}/{len(rows)} trường"]


def run_pipeline(paths, progress=None):
    """Chạy toàn bộ danh sách file -> (DataFrame chi tiết, DataFrame tóm tắt phân loại)."""
    all_rows, summary = [], []
    for i, p in enumerate(paths):
        if progress:
            progress((i, len(paths)), desc=f"Đang xử lý {os.path.basename(p)}")
        rows, sm = process_file(p)
        all_rows += rows
        summary.append(sm)
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    return df, pd.DataFrame(summary, columns=["Tên file", "Loại tài liệu", "Trạng thái"])


# ====================================================================
# GIAO DIỆN STREAMLIT CHUẨN MẪU GỐC (ĐỒ HỌA DARK THEME Y HỆT ẢNH)
# ====================================================================

st.set_page_config(
    page_title="Intelligent KYC — Customer Onboarding", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #0b1120 !important;
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] {
        background: #020617 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    .sidebar-brand {
        display: flex; align-items: center; gap: 12px; padding: 10px 4px 20px 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 15px;
    }
    .brand-icon {
        background: #ffffff; color: #020617; font-weight: 800; font-size: 15px; padding: 8px 12px; border-radius: 10px;
    }
    .brand-title { color: #ffffff; font-size: 16px; font-weight: 700; }
    .brand-subtitle { font-size: 11px; color: #94a3b8; font-weight: 500; }

    /* Custom Radio Navigation */
    .stRadio div[role="radiogroup"] {
        gap: 6px;
    }
    .stRadio div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }
    .stRadio div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }

    /* Main Content Card Container */
    .main-card {
        background: #1e293b; border-radius: 14px; padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        text-align: center; margin-top: 20px;
    }
    
    .empty-state-icon {
        font-size: 32px; background: rgba(255, 255, 255, 0.05);
        display: inline-flex; padding: 16px; border-radius: 14px; margin-bottom: 16px;
        color: #cbd5e1; border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .empty-title { font-size: 16px; font-weight: 700; color: #f8fafc; margin-bottom: 8px; }
    .empty-desc { font-size: 13px; color: #94a3b8; max-width: 500px; margin: 0 auto 24px auto; line-height: 1.5; }

    /* Buttons Styling */
    .stButton > button {
        background: #0f172a !important; color: #ffffff !important; font-weight: 600 !important;
        border-radius: 8px !important; border: 1px solid rgba(255, 255, 255, 0.15) !important;
        padding: 8px 16px !important; font-size: 13px !important;
    }
    .stButton > button:hover {
        background: #1e293b !important; border-color: rgba(255, 255, 255, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

menu_options = [
    "🎛️ Tổng quan Dashboard\n\nChỉ số & thống kê thực tế",
    "📄 Trích xuất tài liệu (OCR)\n\nCCCD, GXNĐC & Form KYC",
    "🔀 Đối chiếu chéo hồ sơ\n\nCross-check & kiểm tra số CCCD",
    "🛡️ Báo cáo KYC & Cờ rủi ro\n\nCompleteness check & Risk score",
    "👥 Hồ sơ khách hàng\n\n0 hồ sơ đã xử lý"
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = menu_options[0]

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">KYC</div>
        <div>
            <div class="brand-title">Intelligent KYC</div>
            <div class="brand-subtitle">Customer Onboarding</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 0.5px; margin-bottom: 8px;'>QUY TRÌNH THẨM ĐỊNH KYC</p>", unsafe_allow_html=True)
    
    current_index = menu_options.index(st.session_state["current_page"]) if st.session_state["current_page"] in menu_options else 0
    menu_choice = st.radio("QUY TRÌNH THẨM ĐỊNH KYC", options=menu_options, index=current_index, label_visibility="collapsed")
    st.session_state["current_page"] = menu_choice

    st.markdown("<div style='margin-top: 40px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    c_side1, c_side2 = st.columns(2)
    with c_side1:
        st.button("⚙️ Cấu hình", use_container_width=True)
    with c_side2:
        if st.button("🔄 Làm mới", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != "current_page":
                    del st.session_state[key]
            st.rerun()
    st.markdown("<p style='font-size: 11px; color: #64748b; margin-top: 12px; line-height: 1.4;'>Công cụ hỗ trợ quy trình thẩm định hồ sơ theo giải thuật IDP & Cross-check của dự án.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Lấy tên trang hiển thị
clean_choice = menu_choice.split("\n\n")[0].replace("🎛️ ", "").replace("📄 ", "").replace("🔀 ", "").replace("🛡️ ", "").replace("👥 ", "")

breadcrumb_map = {
    "Tổng quan Dashboard": "Quy trình 1 / Tổng quan hệ thống",
    "Trích xuất tài liệu (OCR)": "Quy trình 2 / Trích xuất dữ liệu định danh",
    "Đối chiếu chéo hồ sơ": "Quy trình 3 / Đối chiếu chéo dữ liệu",
    "Báo cáo KYC & Cờ rủi ro": "Quy trình 4 / Thẩm định & Đánh giá rủi ro",
    "Hồ sơ khách hàng": "Quản lý / Danh sách hồ sơ lưu trữ"
}

# Tiêu đề Header phía trên
c_top1, c_top2 = st.columns([3, 1])
with c_top1:
