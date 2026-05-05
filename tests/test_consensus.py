"""Test rapide du pipeline a 4 routes + consensus sur 3 factures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config
from models.invoice import Invoice
from extractors.native import extract_from_text
from extractors.vision import extract_from_vision
from utils.pdf_utils import extract_text_from_pdf
from utils.consensus import compute_invoice_consensus
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        extraction_method="consensus",
        source_file=filename,
        confidence=consensus["confidence"],
        consensus_score=consensus["global_score"],
        consensus_confidence=consensus["confidence"],
        consensus_votes={k: v for k, v in consensus["field_scores"].items()},
    )
    return inv, results, consensus


pdfs = sorted(Path(r".\factures").rglob("*.pdf"))[:3]
print(f"Test sur {len(pdfs)} factures\n")
for pdf in pdfs:
    inv, results, consensus = process_one(pdf)
    print(f"\n  === RESULTAT: {inv.source_file} ===")
    print(f"  Score global    : {inv.consensus_score}%")
    print(f"  Confiance       : {inv.consensus_confidence}")
    print(f"  Route gagnante  : {consensus.get('winner_route', 'N/A')}")
    print(f"  Fournisseur     : {inv.nom_fournisseur}")
    print(f"  HT              : {inv.montant_ht}")
    print(f"  TTC             : {inv.montant_ttc}")
    print(f"  N° facture      : {inv.numero_facture}")
    print(f"  Date            : {inv.date_facture}")
    if inv.consensus_votes:
        low_fields = [k for k, v in inv.consensus_votes.items() if v < 75]
        if low_fields:
            print(f"  Champs faibles  : {low_fields}")
    print()
