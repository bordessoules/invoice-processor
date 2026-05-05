"""Extrait et stocke les textes PDF (déterministes, 1 seule fois)."""
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from pathlib import Path
from db import init_db, insert_pdf_text
from utils.pdf_utils import extract_text_pdfplumber, extract_text_pymupdf

def main():
    init_db()
    factures = list(Path("factures").rglob("*.pdf"))
    print(f"[SEED] {len(factures)} PDFs trouves")

    for pdf in factures:
        p = str(pdf)
        # pdfplumber
        t1 = extract_text_pdfplumber(p)
        if t1:
            insert_pdf_text(p, "pdfplumber", t1)
        # pymupdf
        t2 = extract_text_pymupdf(p)
        if t2:
            insert_pdf_text(p, "pymupdf", t2)
        # mineru md (deja genere)
        md = pdf.with_suffix(".md")
        if md.exists():
            insert_pdf_text(p, "mineru", md.read_text(encoding="utf-8"))
        print(f"  [OK] {pdf.name}")

    print("[SEED] Termine")

if __name__ == "__main__":
    main()
