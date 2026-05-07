"""Extraction texte PDF avec auto-détection pdfplumber vs PyMuPDF par dossier."""
import base64
import re
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path

# Files we know how to process (PDFs + raster images).
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}
SUPPORTED_EXTENSIONS = {".pdf"} | IMAGE_EXTENSIONS


def is_image(path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_supported(path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def list_supported_files(root: str | Path) -> list[Path]:
    """Return all PDF + image files under root (sorted, case-insensitive on suffix)."""
    return sorted(
        p for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _garbage_score(text: str) -> float:
    """Score de pollution : 0 = propre, > 0.05 = très sale."""
    if not text:
        return 1.0
    total = len(text)
    if total == 0:
        return 1.0

    # Motifs cid:NNN
    cid = len(re.findall(r"cid:\d+", text, re.I))
    # Caractères de contrôle (sauf \n \r \t)
    control = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    # Mojibakes fréquents sur les PDF mal encodés
    mojibake = sum(1 for c in text if c in "ØŁ")

    # Pondération : cid très grave, control grave, mojibake modéré
    score = (cid * 10 + control * 2 + mojibake * 2) / total
    return score


def extract_text_pdfplumber(pdf_path: str) -> str:
    parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    parts.append(txt)
    except Exception as exc:
        print(f"  [pdfplumber err] {exc}")
    return "\n".join(parts)


def extract_text_pymupdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    try:
        parts = []
        for page in doc:
            parts.append(page.get_text())
        return "\n".join(parts)
    except Exception as exc:
        print(f"  [pymupdf err] {exc}")
        return ""
    finally:
        doc.close()


def extract_text_from_pdf(pdf_path: str, preferred: str = "pdfplumber") -> str:
    """Extrait le texte en essayant l'extracteur préféré, fallback sur l'autre si trop sale."""
    extractors = {
        "pdfplumber": extract_text_pdfplumber,
        "pymupdf": extract_text_pymupdf,
    }
    fallback = "pymupdf" if preferred == "pdfplumber" else "pdfplumber"

    text = extractors[preferred](pdf_path)
    score = _garbage_score(text)
    if score < 0.02:
        return text

    text_fb = extractors[fallback](pdf_path)
    score_fb = _garbage_score(text_fb)
    if score_fb < score:
        return text_fb
    return text


def detect_best_extractor_for_folder(folder_path: str) -> str:
    """Teste pdfplumber vs PyMuPDF sur le 1er PDF du dossier et retourne le meilleur."""
    folder = Path(folder_path)
    pdfs = sorted(folder.rglob("*.pdf"))
    if not pdfs:
        return "pdfplumber"

    sample = pdfs[0]
    t1 = extract_text_pdfplumber(str(sample))
    s1 = _garbage_score(t1)

    t2 = extract_text_pymupdf(str(sample))
    s2 = _garbage_score(t2)

    winner = "pdfplumber" if s1 <= s2 else "pymupdf"
    print(f"  [extractor] {folder.name}: pdfplumber={s1:.3f} pymupdf={s2:.3f} -> {winner}")
    return winner


# ── Vision helpers (inchangés) ─────────────────────────────────────

def pdf_page_to_base64(pdf_path: str, page_num: int = 0, dpi: int = 200) -> str:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")
    finally:
        doc.close()


def pdf_pages_to_base64_list(pdf_path: str, dpi: int = 200, max_pages: int = 5) -> list[str]:
    """Render PDF pages OR a single raster image to base64 PNG strings.

    Accepts both PDFs and image files (JPG/PNG/WebP/...). Images are treated
    as 1-page documents and re-encoded as PNG for consistent downstream handling.
    Kept under its original name for backward compatibility.
    """
    p = Path(pdf_path)
    if p.suffix.lower() in IMAGE_EXTENSIONS:
        # PyMuPDF opens raster images as 1-page docs natively
        doc = fitz.open(str(p))
        try:
            page = doc.load_page(0)
            pix = page.get_pixmap()  # native resolution, no DPI rescaling needed
            return [base64.b64encode(pix.tobytes("png")).decode("utf-8")]
        finally:
            doc.close()

    doc = fitz.open(pdf_path)
    try:
        images = []
        for i in range(min(len(doc), max_pages)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi)
            b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            images.append(b64)
        return images
    finally:
        doc.close()
