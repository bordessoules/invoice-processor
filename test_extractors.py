"""Compare pdfplumber, pdfminer.six et PyMuPDF sur un échantillon de PDF."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

# --- Extraction pdfplumber (actuel) ---
import pdfplumber
def extract_pdfplumber(pdf_path: str) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                parts.append(txt)
    return "\n".join(parts)

# --- Extraction pdfminer.six ---
from pdfminer.high_level import extract_text
def extract_pdfminer(pdf_path: str) -> str:
    try:
        return extract_text(pdf_path)
    except Exception as exc:
        return f"[ERR pdfminer] {exc}"

# --- Extraction PyMuPDF ---
import fitz
def extract_pymupdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    try:
        parts = []
        for page in doc:
            parts.append(page.get_text())
        return "\n".join(parts)
    finally:
        doc.close()

# --- Nettoyage d'encodage avec ftfy si dispo ---
try:
    import ftfy
    HAS_FTFY = True
except Exception:
    HAS_FTFY = False

def clean_text(text: str) -> str:
    if HAS_FTFY:
        return ftfy.fix_text(text)
    return text

SAMPLES = [
    r"factures\bouygues-telecom.fr\Bouyguestelecom_Facture_20250102.pdf",
    r"factures\aliexpress.com\3061532787153743-remboursement.pdf",
    r"factures\accessoiresasus.com\Facture-FC0082368.pdf",
    r"factures\communication-navigo.fr\facture_10442139.pdf",
]

out_path = Path("output/compare_extractors.txt")
out_path.parent.mkdir(exist_ok=True)

with open(out_path, "w", encoding="utf-8") as fh:
    for p in SAMPLES:
        fh.write(f"{'='*80}\nFICHIER: {p}\n{'='*80}\n\n")
        if not os.path.exists(p):
            fh.write("[FICHIER INTROUVABLE]\n\n")
            continue

        # pdfplumber
        t1 = extract_pdfplumber(p)
        fh.write(f"--- pdfplumber ({len(t1)} chars) ---\n")
        fh.write(t1[:1200])
        fh.write("\n...\n")
        fh.write(t1[-600:])
        fh.write("\n\n")

        # pdfminer
        t2 = extract_pdfminer(p)
        fh.write(f"--- pdfminer.six ({len(t2)} chars) ---\n")
        fh.write(t2[:1200])
        fh.write("\n...\n")
        fh.write(t2[-600:])
        fh.write("\n\n")

        # pymupdf
        t3 = extract_pymupdf(p)
        fh.write(f"--- PyMuPDF ({len(t3)} chars) ---\n")
        fh.write(t3[:1200])
        fh.write("\n...\n")
        fh.write(t3[-600:])
        fh.write("\n\n")

        # ftfy sur pdfplumber
        if HAS_FTFY:
            t1c = clean_text(t1)
            fh.write(f"--- pdfplumber + ftfy ({len(t1c)} chars) ---\n")
            fh.write(t1c[:1200])
            fh.write("\n...\n")
            fh.write(t1c[-600:])
            fh.write("\n\n")

        fh.write("\n\n")

print(f"[OK] Résultats dans {out_path}")
