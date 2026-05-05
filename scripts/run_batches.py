"""Traite les factures par lots pour eviter le timeout shell (300s max)."""
import os, sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models.invoice import Invoice
from extractors.native import extract_from_text
from extractors.vision import extract_from_vision
from utils.pdf_utils import extract_text_from_pdf
from utils.consensus import compute_invoice_consensus

BATCH_SIZE = 20
OUTPUT_DIR = Path(config.OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = OUTPUT_DIR / "batch_state.json"


def _extract_route(route, pdf_path, raw_text):
    filename = pdf_path.name
    try:
        if route == "qwen_text":
            if not raw_text or len(raw_text) < 50:
                return route, None
            return route, extract_from_text(raw_text, filename, model=config.MODEL_QWEN, sampling=config.SAMPLING_QWEN)
        elif route == "qwen_vision":
            return route, extract_from_vision(str(pdf_path), filename, model=config.MODEL_QWEN, sampling=config.SAMPLING_QWEN)
        elif route == "gemma_text":
            if not raw_text or len(raw_text) < 50:
                return route, None
            return route, extract_from_text(raw_text, filename, model=config.MODEL_GEMMA, sampling=config.SAMPLING_GEMMA)
        elif route == "gemma_vision":
            return route, extract_from_vision(str(pdf_path), filename, model=config.MODEL_GEMMA, sampling=config.SAMPLING_GEMMA)
    except Exception as exc:
        print(f"   [{route}] ERR: {exc}")
        return route, None


def process_one(pdf_path):
    filename = pdf_path.name
    raw_text = extract_text_from_pdf(str(pdf_path))
    text_len = len(raw_text.strip()) if raw_text else 0
    print(f"[PDF] {filename} ({text_len} chars)")

    routes = ["qwen_text", "qwen_vision", "gemma_text", "gemma_vision"]
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_extract_route, r, pdf_path, raw_text): r for r in routes}
        for future in as_completed(futures):
            route, inv = future.result()
            results[route] = inv
            print(f"   [{route:15}] {'OK' if inv else 'FAIL'}")

    consensus = compute_invoice_consensus(results)
    cfields = consensus["fields"]
    winner_route = consensus.get("winner_route")
    wr_obj = results.get(winner_route)
    lignes = wr_obj.lignes if wr_obj and hasattr(wr_obj, "lignes") else None

    inv = Invoice(
        numero_facture=cfields.get("numero_facture"),
        date_facture=cfields.get("date_facture"),
        date_echeance=cfields.get("date_echeance"),
        nom_fournisseur=cfields.get("nom_fournisseur"),
        siret_fournisseur=cfields.get("siret_fournisseur"),
        adresse_fournisseur=getattr(wr_obj, "adresse_fournisseur", None) if wr_obj else None,
        nom_client=getattr(wr_obj, "nom_client", None) if wr_obj else None,
        montant_ht=cfields.get("montant_ht"),
        montant_tva=cfields.get("montant_tva"),
        montant_ttc=cfields.get("montant_ttc"),
        devise=cfields.get("devise"),
        taux_tva=cfields.get("taux_tva"),
        lignes=lignes,
        type_document=cfields.get("type_document"),
        categorie=cfields.get("categorie"),
        moyen_paiement=cfields.get("moyen_paiement"),
        iban_fournisseur=cfields.get("iban_fournisseur"),
        numero_tva=cfields.get("numero_tva"),
        numero_commande_client=cfields.get("numero_commande_client"),
        url_fichier=str(pdf_path.resolve()),
        raw_text=raw_text[:2000] if raw_text else None,
        extraction_method="consensus",
        source_file=filename,
        confidence=consensus["confidence"],
        consensus_score=consensus["global_score"],
        consensus_confidence=consensus["confidence"],
        consensus_votes={k: v for k, v in consensus["field_scores"].items()},
    )
    inv._raw_results = {
        r: (i.model_dump() if i else None)
        for r, i in results.items()
    }
    print(f"   [CONSENSUS] score={consensus['global_score']:.0f}%  conf={consensus['confidence']}")
    return inv


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done": [], "results": []}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


def main():
    pdfs = sorted(Path(config.INPUT_DIR).rglob("*.pdf"))
    if not pdfs:
        print("[!] Aucun PDF trouve")
        return

    state = load_state()
    done_files = set(state["done"])
    remaining = [p for p in pdfs if p.name not in done_files]

    print(f"[BATCH] Total: {len(pdfs)} | Deja faits: {len(done_files)} | Restants: {len(remaining)}")

    if not remaining:
        print("[BATCH] Tout est deja traite !")
        return

    batch = remaining[:BATCH_SIZE]
    print(f"[BATCH] Traitement du lot: {len(batch)} factures\n")

    for pdf in batch:
        try:
            inv = process_one(pdf)
            state["done"].append(pdf.name)
            d = inv.model_dump()
            if hasattr(inv, "_raw_results"):
                d["_raw_results"] = inv._raw_results
            state["results"].append(d)
            save_state(state)
        except Exception as exc:
            print(f"[ERR] Echec sur {pdf.name}: {exc}")

    print(f"\n[BATCH] Lot termine. {len(state['done'])}/{len(pdfs)} factures traitees.")

    # Export final si tout est fait
    if len(state["done"]) >= len(pdfs):
        print("[BATCH] Export final...")
        export(state["results"])


def export(results):
    # JSON
    json_path = OUTPUT_DIR / "invoices.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"[SAVE] JSON : {json_path}")

    # CSV factures
    flat = []
    for d in results:
        d2 = dict(d)
        d2.pop("lignes", None)
        d2.pop("raw_text", None)
        d2.pop("_raw_results", None)
        flat.append(d2)
    import pandas as pd
    df = pd.DataFrame(flat)
    csv_path = OUTPUT_DIR / "invoices.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] CSV  : {csv_path}")

    # CSV lignes
    lines = []
    for d in results:
        if d.get("lignes"):
            for li in d["lignes"]:
                ld = dict(li)
                ld["source_file"] = d.get("source_file")
                ld["numero_facture"] = d.get("numero_facture")
                ld["consensus_score"] = d.get("consensus_score")
                lines.append(ld)
    if lines:
        df_lines = pd.DataFrame(lines)
        lines_path = OUTPUT_DIR / "invoices_lines.csv"
        df_lines.to_csv(lines_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] Lignes: {lines_path}")

    # Stats
    high = sum(1 for r in results if r.get("consensus_confidence") == "high")
    med = sum(1 for r in results if r.get("consensus_confidence") == "medium")
    low = sum(1 for r in results if r.get("consensus_confidence") == "low")
    avg = sum(r.get("consensus_score", 0) for r in results) / len(results)
    print(f"\n[STATS] High:{high}  Medium:{med}  Low:{low}  Score moy:{avg:.1f}%")


if __name__ == "__main__":
    main()
