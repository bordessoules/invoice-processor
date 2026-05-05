#!/usr/bin/env python3
import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

"""
Invoice Processor -- Pipeline hybride 4 routes avec consensus.
Sauvegarde incrementale en JSON Lines.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from utils.pdf_utils import detect_best_extractor_for_folder
from pipeline import process_one, save_jsonl, load_done_set, finalize_output


def main() -> None:
    input_dir = Path(config.INPUT_DIR)
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "invoices.jsonl"

    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"[!] Aucun PDF trouve dans {input_dir.absolute()}")
        return

    # --- Phase 1 : detecter le meilleur extracteur par dossier ------
    print("=" * 70)
    print("[PHASE 1] Detection extracteur par dossier")
    print("=" * 70)
    folders = sorted({p.parent for p in pdfs})
    folder_extractors = {}
    for folder in folders:
        folder_extractors[folder.name] = detect_best_extractor_for_folder(str(folder))

    # --- Phase 2 : reprendre si deja commence -----------------------
    done = load_done_set(jsonl_path)
    if done:
        print(f"\n[INFO] {len(done)} facture(s) deja traitee(s), reprise.")
    todo = [p for p in pdfs if p.name not in done]

    print("\n" + "=" * 70)
    print(f"[PHASE 2] Traitement : {len(todo)}/{len(pdfs)} factures")
    print(f"[CONC]    Concurrence PDFs : {config.MAX_CONCURRENT}")
    print(f"[API]     Endpoint         : {config.OPENAI_BASE_URL}")
    print(f"[QWEN]    {config.MODEL_QWEN}")
    print(f"[GEMMA]   {config.MODEL_GEMMA}")
    print("=" * 70)

    # --- Phase 3 : traitement avec sauvegarde incrementale ----------
    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT) as executor:
        futures = {}
        for pdf in todo:
            extractor = folder_extractors.get(pdf.parent.name, "pdfplumber")
            fut = executor.submit(process_one, pdf, extractor)
            futures[fut] = pdf

        for future in as_completed(futures):
            pdf = futures[future]
            try:
                invoice_dict = future.result()
                save_jsonl(jsonl_path, invoice_dict)
            except Exception as exc:
                print(f"[ERR] {pdf.name} -> {exc}")

    # --- Phase 4 : finalisation -------------------------------------
    print("\n" + "=" * 70)
    print("[PHASE 4] Finalisation")
    print("=" * 70)
    finalize_output(jsonl_path, output_dir)


if __name__ == "__main__":
    main()
