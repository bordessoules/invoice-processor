import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from pathlib import Path
from utils.pdf_utils import detect_best_extractor_for_folder
from pipeline import process_one, save_jsonl, load_done_set, finalize_output

SAMPLES = [
    Path("factures/accessoiresasus.com/Facture-FC0082368.pdf"),
    Path("factures/bouygues-telecom.fr/Bouyguestelecom_Facture_20250102.pdf"),
    Path("factures/communication-navigo.fr/facture_10442139.pdf"),
    Path("factures/aliexpress.com/3061532787153743-remboursement.pdf"),
    Path("factures/2checkout.com/invoice_BV97034136.pdf"),
]

output_dir = Path("output")
jsonl_path = output_dir / "test_invoices.jsonl"

# cleanup precedent test
if jsonl_path.exists():
    jsonl_path.unlink()

folder_extractors = {}
for p in SAMPLES:
    folder = p.parent.name
    if folder not in folder_extractors:
        folder_extractors[folder] = detect_best_extractor_for_folder(str(p.parent))

for p in SAMPLES:
    extractor = folder_extractors[p.parent.name]
    try:
        result = process_one(p, extractor)
        save_jsonl(jsonl_path, result)
    except Exception as exc:
        print(f"[ERR] {p.name}: {exc}")

finalize_output(jsonl_path, output_dir)
