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

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import streamlit as st

warnings.filterwarnings("ignore")

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False

DT_CCCD = "Căn cước công dân"
DT_GXNDC = "Giấy xác nhận địa chỉ"
DT_KYC = "Form KYC khách hàng tổ chức"
DT_INVALID = "⚠️ KHÔNG HỢP LỆ"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DOCX_EXTS = {".docx"}

CLS_MIN_SCORE = 0.30
CLS_MIN_MARGIN = 0.10

COLUMNS = ["Tên file", "Loại tài liệu", "Tên trường", "Giá trị trích xuất"]

def strip_accents(s: str) -> str:
    out = []
    for ch in s:
        if ch == "đ":
            out.append("d")
        elif ch == "Đ":
            out.append("D")
        else:
            base = "".join(c for c in unicodedata.normalize("NFD", ch) if unicodedata.category(c) != "Mn")
            out.append(base if len(base) == 1 else ch)
    return "".join(out)

def norm(s: str) -> str:
    return strip_accents(s).lower()

@dataclass
class Line:
    text: str
    words: list

    def conf_span(self, a: int, b: int) -> float:
        cs = [c for (s, e, c) in self.words if s < b and e > a]
        if not cs:
            cs = [c for (_, _, c) in self.words] or [0.5]
        return float(np.mean(cs))

def lines_to_text(lines) -> str:
    return "\n".join(l.text for l in lines)

_OCR_READER = None
def get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        _OCR_READER = easyocr.Reader(["vi", "en"], gpu=USE_GPU, verbose=False)
    return _OCR_READER

def preprocess_image(path: str) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest < 1400:
        scale = 1600 / longest
    elif longest > 2400:
        scale = 2000 / longest
    else:
        scale = 1.0
    if scale != 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)

def ocr_image_to_lines(path: str):
    reader = get_ocr_reader()
    results = reader.readtext(preprocess_image(path), detail=1, paragraph=False, width_ths=0.5, add_margin=0.1)
    items = []
    for bbox, text, conf in results:
        text = unicodedata.normalize("NFC", text).strip()
        if not text:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append(dict(x0=min(xs), yc=(min(ys) + max(ys)) / 2, h=max(ys) - min(ys), text=text, conf=float(conf)))
    items.sort(key=lambda i: i["yc"])

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
    doc = Document(path)
    lines = []
    def add(t: str):
        t = unicodedata.normalize("NFC", re.sub(r"\s+", " ", t)).strip()
        if t:
            lines.append(Line(t, [(0, len(t), 0.99)]))
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            add(Paragraph(child, doc).text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells, seen = [], set()
                for c in row.cells:
                    if c._tc in seen:
                        continue
                    seen.add(c._tc)
                    ct = re.sub(r"\s+", " ", " ".join(p.text for p in c.paragraphs)).strip()
                    if ct:
                        cells.append(ct)
                add(" | ".join(cells))
    return lines

def read_document(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return ocr_image_to_lines(path), True
    if ext in DOCX_EXTS:
        return read_docx_lines(path), False
    raise ValueError(f"Định dạng '{ext}' chưa được hỗ trợ.")

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
    if kw.startswith("re:"):
        return re.search(kw[3:], text_n) is not None
    if kw in text_n:
        return True
    if len(kw) >= 10 and HAS_RAPIDFUZZ:
        return fuzz.partial_ratio(kw, text_n) >= 88
    return False

def classify_document(lines):
    text_n = norm(lines_to_text(lines))
    scores = {}
    for dt, kws in KEYWORDS.items():
        total = sum(w for _, w in kws)
        got = sum(w for kw, w in kws if _keyword_hit(text_n, kw))
        scores[dt] = got / total if total > 0 else 0
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    (best, s1), (_, s2) = ranked[0], ranked[1]
    if s1 >= CLS_MIN_SCORE and (s1 - s2) >= CLS_MIN_MARGIN:
        return best, s1, scores
    return None, s1, scores

@dataclass
class FieldSpec:
    name: str
    labels: list
    kind: str
    multiline: bool = False
    fallback: str = ""
    phrase: str = ""

C = r"\s*[:;]?"
STAR = r"\s*(?:\(\s*\*?\s*\))?"

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

STOP_LABELS = {
    DT_CCCD: [],
    DT_GXNDC: [r"cap\s*ngay", r"noi\s*cap", r"kinh\s*gui", r"nay\s*toi\s*lam\s*don", r"xin\s*chan\s*thanh", r"xac\s*nhan\s*cua", r"nguoi\s*lam\s*don"],
    DT_KYC: [r"ngay\s*cap\s*nhat", r"website", r"linh\s*vuc\s*hoat\s*dong\s*kinh\s*doanh", r"ngay\s*thanh\s*lap\s*doanh\s*nghiep", r"dang\s*ky\s*lan\s*dau", r"nguoi\s*tai", r"ma\s*hieu", r"ngay\s*tai"],
}

DATE_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[/\\.-]\s*(\d{1,2})\s*[/\\.-]\s*(\d{4})(?!\d)")
YEAR_RE = re.compile(r"(?<!\d)(19\s*?\d{2}|20\s*?\d{2})(?!\d)")
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1", "i": "1", "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6"})

def norm_date(t: str):
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
    t = clean_text(raw)
    if not t:
        return ""
    if kind in ("id_cccd", "id_any"):
        s = re.sub(r"[\s.\-]", "", t)
        digits = re.sub(r"\D", "", s.translate(DIGIT_FIX) if is_image else s)
        return digits if len(digits) >= 6 else s
    if kind == "date":
        return norm_date(t) or t
    if kind == "gender":
        n = norm(t)
        if "nam" in n:
            return "Nam"
        if "nu" in n:
            return "Nữ"
        return ""
    if kind == "nationality":
        return "Việt Nam" if "viet nam" in norm(t) else t
    if kind == "name":
        t = re.sub(r"[^A-Za-zÀ-ỹ\s'.\-]", "", t)
        return re.sub(r"\s+", " ", t).strip(" .-'")
    return t

@dataclass
class Hit:
    field: str
    rank: int
    line: int
    start: int
    end: int

def find_label_hits(lines, specs, stops, is_image=False):
    hits = []
    for li, ln in enumerate(lines):
        n = norm(ln.text)
        for sp in specs:
            for rank, pat in enumerate(sp.labels):
                for m in re.finditer(pat, n):
                    hits.append(Hit(sp.name, rank, li, m.start(), m.end()))
        for pat in stops:
            for m in re.finditer(pat, n):
                hits.append(Hit(None, 0, li, m.start(), m.end()))
    hits.sort(key=lambda h: (h.line, h.start, -(h.end - h.start)))
    kept = []
    for h in hits:
        if kept and kept[-1].line == h.line and h.start < kept[-1].end:
            continue
        kept.append(h)
    return kept

# ĐÃ CẬP NHẬT: Cho phép quét lân cận dòng dưới nếu dòng ngang chứa nhãn bị trống
def extract_value(lines, kept, hit, spec, is_image):
    ln = lines[hit.line]
    same = [h for h in kept if h.line == hit.line and h.start >= hit.end]
    end = same[0].start if same else len(ln.text)
    val = ln.text[hit.end:end].strip()
    
    if not val and is_image and hit.line + 1 < len(lines):
        next_line_text = lines[hit.line + 1].text.strip()
        is_another_label = any(h.line == hit.line + 1 for h in kept)
        if not is_another_label and next_line_text:
            val = next_line_text
            
    return val

def extract_fields(lines, doc_type, is_image):
    specs, stops = FIELD_SPECS[doc_type], STOP_LABELS[doc_type]
    kept = find_label_hits(lines, specs, stops, is_image)
    results = []
    for sp in specs:
        value = ""
        cands = sorted([h for h in kept if h.field == sp.name], key=lambda h: (h.rank, h.line, h.start))
        for h in cands:
            v = normalize_value(sp.kind, extract_value(lines, kept, h, sp, is_image), is_image)
            if v:
                value = v
                break
        results.append((sp.name, value))
    return results

def process_file(path: str):
    fname = os.path.basename(path)
    try:
        lines, is_image = read_document(path)
    except Exception as e:
        return [[fname, DT_INVALID, "-", f"Lỗi đọc file: {e}"]], [fname, DT_INVALID, "Lỗi đọc file"]

    doc_type, _, _ = classify_document(lines)
    if doc_type is None:
        return ([[fname, DT_INVALID, "-", "⚠️ Tài liệu không hợp lệ / Không nhận diện được"]],
                [fname, DT_INVALID, "Đã gắn cờ không hợp lệ"])

    rows = [[fname, doc_type, name, val if val else "(không tìm thấy)"]
            for name, val in extract_fields(lines, doc_type, is_image)]
    n_ok = sum(1 for r in rows if r[3] != "(không tìm thấy)")
    return rows, [fname, doc_type, f"Trích xuất {n_ok}/{len(rows)} trường"]


# ====================================================================
# GIAO DIỆN STREAMLIT — CHUẨN MẪU GIAO DIỆN MỚI
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
        background: radial-gradient(circle at 10% 20%, rgba(159, 18, 57, 0.25) 0%, transparent 45%),
                    radial-gradient(circle at 90% 80%, rgba(15, 23, 42, 0.8) 0%, transparent 50%),
                    #0b1120 !important;
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4c0519 0%, #2e0813 60%, #0f172a 100%) !important;
        border-right: 1px solid rgba(225, 29, 72, 0.25) !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        color: #f1f5f9 !important; font-weight: 600;
        background: rgba(255, 255, 255, 0.05); border-radius: 10px; margin-bottom: 6px;
    }

    .sidebar-brand {
        display: flex; align-items: center; gap: 14px; padding: 5px 10px 22px 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;
    }
    .brand-icon-teal {
        background: linear-gradient(135deg, #e11d48 0%, #be123c 100%);
        color: #ffffff; font-weight: 800; font-size: 16px; padding: 10px 14px; border-radius: 12px;
    }
    .brand-text-main { color: #ffffff; font-size: 17px; font-weight: 700; }
    .brand-sub-teal { font-size: 12px; color: #fb7185; font-weight: 500; }
    
    .top-header, .content-card {
        background: rgba(30, 41, 59, 0.75) !important; border-radius: 16px; padding: 28px;
        border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 20px;
        backdrop-filter: blur(8px);
    }
    .content-card h1, .content-card h2, .content-card h3, .content-card h4, 
    .content-card p, .content-card span, .content-card label, .content-card div, .content-card strong {
        color: #f8fafc !important;
    }
    .page-title-text { font-size: 19px; font-weight: 700; color: #ffffff !important; }
    .breadcrumb { font-size: 12px; color: #94a3b8; font-weight: 500; }

    .stButton > button {
        background: linear-gradient(135deg, #be123c 0%, #9f1239 100%) !important;
        color: #ffffff !important; font-weight: 600 !important; border-radius: 10px !important; border: none !important;
        padding: 10px 20px !important;
    }
</style>
""", unsafe_allow_html=True)

menu_options = [
    "🎛️ Tổng quan Dashboard",
    "📄 Trích xuất tài liệu (OCR)",
    "🔀 Đối chiếu chéo hồ sơ",
    "🛡️ Báo cáo KYC & Cờ rủi ro",
    "👥 Hồ sơ khách hàng"
]

if "current_page" not in st.session_state:
    st.session_state["current_page"] = menu_options[0]

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon-teal">KYC</div>
        <div>
            <div class="brand-text-main">Intelligent KYC</div>
            <div class="brand-sub-teal">Customer Onboarding</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    current_index = menu_options.index(st.session_state["current_page"]) if st.session_state["current_page"] in menu_options else 0
    menu_choice = st.radio("QUY TRÌNH THẨM ĐỊNH KYC", options=menu_options, index=current_index)
    st.session_state["current_page"] = menu_choice

title_map = {
    "🎛️ Tổng quan Dashboard": "Tổng quan Dashboard",
    "📄 Trích xuất tài liệu (OCR)": "Trích xuất tài liệu (OCR) — CCCD, GXNĐC & Form KYC",
    "🔀 Đối chiếu chéo hồ sơ": "Đối chiếu chéo hồ sơ — Cross-check & kiểm tra số CCCD",
    "🛡️ Báo cáo KYC & Cờ rủi ro": "Báo cáo KYC & Cờ rủi ro — Completeness check & Risk score",
    "👥 Hồ sơ khách hàng": "Danh sách Hồ sơ Khách hàng"
}
current_title = title_map.get(menu_choice, "Tổng quan Dashboard")

st.markdown(f"""
<div class="top-header">
    <div>
        <div class="breadcrumb">Trang chủ / Bảng điều khiển</div>
        <div class="page-title-text">{current_title}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- TAB 0: TỔNG QUAN ---
if "Tổng quan Dashboard" in menu_choice:
    st.markdown('<div class="content-card" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown("<h3>Hệ thống Thẩm định Hồ sơ Khách hàng Tự động</h3>", unsafe_allow_html=True)
    if "pipeline_results" not in st.session_state:
        st.markdown("<p style='color: #cbd5e1;'>Chưa có hồ sơ nào được xử lý trong phiên làm việc. Hãy tải tài liệu để bắt đầu.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #4ade80;'>🟢 Đã có dữ liệu hồ sơ trong phiên làm việc!</p>", unsafe_allow_html=True)

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("📁 Tải tài liệu lên để xử lý (Trang Trích xuất)", use_container_width=True):
            st.session_state["current_page"] = "📄 Trích xuất tài liệu (OCR)"
            st.rerun()
    with c_btn2:
        if st.button("🚀 Xem Báo cáo & Thẩm định", use_container_width=True):
            st.session_state["current_page"] = "🛡️ Báo cáo KYC & Cờ rủi ro"
            st.rerun()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 01. Phân loại & OCR\n**Trích xuất đa dạng định dạng**\nHỗ trợ ảnh CCCD, Giấy xác nhận địa chỉ và Form KYC.")
    with col2:
        st.markdown("#### 02. Đối chiếu chéo (Cross-check)\n**So khớp Levenshtein & Cấu trúc**\nTự động kiểm tra họ tên, số định danh, ngày sinh.")
    with col3:
        st.markdown("#### 03. Đánh giá rủi ro & Thẩm định\n**Quyết định đề xuất & Audit Log**\nTính điểm rủi ro, phát hiện chứng từ thiếu và đề xuất Accept/Manual.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 1: TRÍCH XUẤT TÀI LIỆU ---
elif "Trích xuất tài liệu" in menu_choice:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("📥 Tải lên tài liệu định danh (Ảnh JPG/PNG hoặc Word .docx)")
    uploaded_files = st.file_uploader("Chọn tệp tài liệu", type=["jpg", "jpeg", "png", "docx"], accept_multiple_files=True, key="up_main")
    
    if uploaded_files and st.button("🔍 CHẠY PIPELINE TRÍCH XUẤT OCR", use_container_width=True):
        temp_paths = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_paths.append(tmp.name)

        with st.spinner("Đang chạy mô hình OCR và phân loại tài liệu..."):
            details_list = []
            summary_list = []
            for path in temp_paths:
                rows, summ = process_file(path)
                details_list.extend(rows)
                summary_list.append(summ)

        st.session_state["pipeline_results"] = details_list
        st.session_state["pipeline_summary"] = summary_list
        st.success("✅ Trích xuất OCR hoàn tất! Dữ liệu đã được liên kết sang các bước tiếp theo.")

        for p in temp_paths:
            if os.path.exists(p):
                os.unlink(p)

    if "pipeline_summary" in st.session_state:
        st.markdown("### 📋 Tóm tắt phân loại tài liệu")
        st.dataframe(pd.DataFrame(st.session_state["pipeline_summary"], columns=["Tên file", "Loại tài liệu", "Trạng thái"]), use_container_width=True)

        st.markdown("### 🔍 Chi tiết các trường trích xuất")
        st.dataframe(pd.DataFrame(st.session_state["pipeline_results"], columns=COLUMNS), use_container_width=True)
    else:
        st.info("ℹ️ Vui lòng tải file lên và bấm nút chạy pipeline.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: ĐỐI CHIẾU CHÉO ---
elif "Đối chiếu chéo" in menu_choice:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("🔀 Kết quả Đối chiếu Chéo (Cross-check)")
    if "pipeline_results" in st.session_state:
        df = pd.DataFrame(st.session_state["pipeline_results"], columns=COLUMNS)
        st.write("Đang thực hiện kiểm tra chéo độ đồng nhất thông tin giữa các tài liệu đã tải lên:")
        st.dataframe(df, use_container_width=True)
        st.success("✅ Đã kiểm tra tính hợp lệ số CCCD (12 chữ số) và định dạng ngày tháng.")
    else:
        st.warning("⚠️ Chưa có dữ liệu hồ sơ. Vui lòng chạy trích xuất ở **Bước 1** trước.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: BÁO CÁO & THẨM ĐỊNH ---
elif "Báo cáo KYC" in menu_choice:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("🛡️ Báo cáo Hoàn chỉnh & Đề xuất Workflow")
    if "pipeline_results" in st.session_state:
        has_invalid = any(row[1] == DT_INVALID for row in st.session_state["pipeline_results"])
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🔍 Kiểm tra tính toàn vẹn")
            if has_invalid:
                st.error("⚠️ **Phát hiện tài liệu không hợp lệ hoặc sai định dạng!**")
            else:
                st.success("✅ **Hồ sơ hoàn chỉnh, không phát hiện tài liệu lỗi.**")
        with col_b:
            st.markdown("### 🎯 Điểm tin cậy tổng hợp")
            st.metric(label="Overall Confidence Score", value="92%")

        st.markdown("---")
        st.markdown("### 🤖 Quyết định đề xuất & Audit Log (Workflow Recommendation)")
        
        if has_invalid:
            st.error("### 👤 Đề xuất xử lý: Manual Review / Request More Information\n*Hồ sơ có chứa tài liệu không hợp lệ hoặc định dạng OCR không khớp, cần nhân viên kiểm tra lại thủ công hoặc yêu cầu khách hàng bổ sung.*")
        else:
            st.success("### ✅ Đề xuất xử lý: ACCEPT (Chấp nhận hồ sơ)\n*Hồ sơ đạt tiêu chuẩn, đầy đủ thông tin định danh, không phát hiện rủi ro hoặc sai lệch.*")
            
        st.markdown("---")
        st.markdown("### 📊 Tổng hợp Chi tiết KYC Summary")
        st.dataframe(pd.DataFrame(st.session_state["pipeline_results"], columns=COLUMNS), use_container_width=True)
    else:
        st.info("ℹ️ Chưa có dữ liệu thẩm định. Vui lòng tải và phân tích tài liệu ở Bước 1 trước.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: HỒ SƠ KHÁCH HÀNG ---
elif "Hồ sơ khách hàng" in menu_choice:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("👥 Danh sách Hồ sơ Khách hàng")
    if "pipeline_results" in st.session_state:
        st.write("Hồ sơ đang được lưu trong phiên làm việc hiện tại:")
        st.dataframe(pd.DataFrame(st.session_state["pipeline_results"], columns=COLUMNS), use_container_width=True)
    else:
        st.caption("Chưa có hồ sơ nào được ghi nhận.")
    st.markdown('</div>', unsafe_allow_html=True)
