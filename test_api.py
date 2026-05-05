import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import time
import json
from pathlib import Path
from utils.pdf_utils import extract_text_from_pdf
from extractors.native import extract_from_text

pdf = Path("factures/accessoiresasus.com/Facture-FC0082368.pdf")
text = extract_text_from_pdf(str(pdf), preferred="pdfplumber")

print(f"[TEST] PDF: {pdf.name}")
print(f"[TEST] Texte extrait: {len(text)} chars")

t0 = time.time()
inv = extract_from_text(text, pdf.name)
elapsed = time.time() - t0

result = {
    "elapsed_sec": round(elapsed, 1),
    "numero_facture": inv.numero_facture,
    "date_facture": inv.date_facture,
    "nom_fournisseur": inv.nom_fournisseur,
    "montant_ht": str(inv.montant_ht) if inv.montant_ht else None,
    "montant_ttc": str(inv.montant_ttc) if inv.montant_ttc else None,
    "type_document": inv.type_document,
    "categorie": inv.categorie,
    "moyen_paiement": inv.moyen_paiement,
    "lignes_count": len(inv.lignes) if inv.lignes else 0,
    "lignes": [li.model_dump() for li in inv.lignes] if inv.lignes else [],
}

with open("output/test_api_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=str)

print(f"[OK] Qwen texte en {elapsed:.1f}s")
print(f"     N°={inv.numero_facture}  Fournisseur={inv.nom_fournisseur}  HT={inv.montant_ht}  TTC={inv.montant_ttc}")
print(f"     Type={inv.type_document}  Cat={inv.categorie}  Lignes={len(inv.lignes) if inv.lignes else 0}")
