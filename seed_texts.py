"""Extrait et stocke les textes PDF + détecte images (déterministes, 1 seule fois).

Les images (JPG/PNG/...) ne contiennent pas de texte extractible : elles sont juste
loggées pour traçabilité, et seront traitées en mode vision plus tard.
"""
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from pathlib import Path
from db import init_db, insert_pdf_text
from utils.pdf_utils import (
    extract_text_pdfplumber,
    extract_text_pymupdf,
    is_image,
    list_supported_files,
)


def main():
    init_db()
    files = list_supported_files("factures")
    n_pdf = sum(1 for f in files if not is_image(f))
    n_img = sum(1 for f in files if is_image(f))
    print(f"[SEED] {n_pdf} PDFs + {n_img} images trouvés")

    for f in files:
        p = str(f)
        if is_image(f):
            # Pas de texte extractible — sera traité en mode vision
            print(f"  [IMG] {f.name}  (vision-only)")
            continue

        # pdfplumber
        t1 = extract_text_pdfplumber(p)
        if t1:
            insert_pdf_text(p, "pdfplumber", t1)
        # pymupdf
        t2 = extract_text_pymupdf(p)
        if t2:
            insert_pdf_text(p, "pymupdf", t2)
        # mineru md (déjà généré)
        md = f.with_suffix(".md")
        if md.exists():
            insert_pdf_text(p, "mineru", md.read_text(encoding="utf-8"))
        print(f"  [OK]  {f.name}")

    print("[SEED] Termine")


if __name__ == "__main__":
    main()
