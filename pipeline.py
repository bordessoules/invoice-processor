"""Pipeline modulaire : extraction 4 routes + consensus + sauvegarde incrémentale."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from models.invoice import Invoice
from extractors.native import extract_from_text
from extractors.vision import extract_from_vision
from utils.pdf_utils import extract_text_from_pdf
from utils.consensus import compute_invoice_consensus


def _extract_route(route: str, pdf_path: Path, raw_text: str):
    """Extrait une facture avec une route donnee. Retourne (route, invoice_or_none)."""
    filename = pdf_path.name
    try:
        if route == "qwen_text":
            if not raw_text or len(raw_text) < 50:
                return route, None
            inv = extract_from_text(raw_text, filename, model=config.MODEL_QWEN, sampling=config.SAMPLING_QWEN)
            return route, inv
        elif route == "qwen_vision":
            inv = extract_from_vision(str(pdf_path), filename, model=config.MODEL_QWEN, sampling=config.SAMPLING_QWEN)
            return route, inv
        elif route == "gemma_text":
            if not raw_text or len(raw_text) < 50:
                return route, None
            inv = extract_from_text(raw_text, filename, model=config.MODEL_GEMMA, sampling=config.SAMPLING_GEMMA)
            return route, inv
        elif route == "gemma_vision":
            inv = extract_from_vision(str(pdf_path), filename, model=config.MODEL_GEMMA, sampling=config.SAMPLING_GEMMA)
            return route, inv
    except Exception as exc:
        print(f"   [{route}] ERR: {exc}")
        return route, None


def process_one(pdf_path: Path, extractor: str) -> dict:
    """Traite un PDF : extraction texte, 4 routes LLM, consensus -> dict."""
    filename = pdf_path.name
    full_path = str(pdf_path.resolve())
    print(f"[PDF] {filename}")

    raw_text = extract_text_from_pdf(str(pdf_path), preferred=extractor)
    text_len = len(raw_text.strip()) if raw_text else 0
    print(f"   -> texte brut ({extractor}): {text_len} chars")

    routes = ["qwen_text", "qwen_vision", "gemma_text", "gemma_vision"]
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_extract_route, r, pdf_path, raw_text): r for r in routes}
        for future in as_completed(futures):
            route, inv = future.result()
            results[route] = inv
            status = "OK" if inv else "FAIL"
            print(f"   [{route:15}] {status}")

    # Consensus
    consensus = compute_invoice_consensus(results)
    cfields = consensus["fields"]

    # On prend les lignes de la route gagnante
    winner_route = consensus.get("raw_results", {}).get(consensus.get("winner_route"))
    lignes = winner_route.lignes if winner_route and hasattr(winner_route, "lignes") else None

    invoice = Invoice(
        numero_facture=cfields.get("numero_facture"),
        date_facture=cfields.get("date_facture"),
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
        numero_tva=cfields.get("numero_tva"),
        numero_commande_client=cfields.get("numero_commande_client"),
        url_fichier=full_path,
        raw_text=raw_text[:2000] if raw_text else None,
        extraction_method="consensus",
        source_file=filename,
        confidence=consensus["confidence"],
        consensus_score=consensus["global_score"],
        consensus_confidence=consensus["confidence"],
        consensus_votes={k: v for k, v in consensus["field_scores"].items()},
        consensus_winner_route=consensus["winner_route"],
    )

    # Stocke les 4 resultats bruts pour debug
    d = invoice.model_dump()
    d["_raw_results"] = {
        r: (inv.model_dump() if inv else None)
        for r, inv in results.items()
    }

    print(f"   [CONSENSUS] score={consensus['global_score']:.0f}%  conf={consensus['confidence']}")
    return d


# ── Sauvegarde incrémentale ─────────────────────────────────────────

def save_jsonl(path: Path, invoice_dict: dict) -> None:
    """Append une ligne JSON au fichier .jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(invoice_dict, ensure_ascii=False, default=str) + "\n")


def load_done_set(jsonl_path: Path) -> set[str]:
    """Retourne l'ensemble des source_file deja presents dans le jsonl."""
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sf = obj.get("source_file")
                if sf:
                    done.add(sf)
            except json.JSONDecodeError:
                continue
    return done


# ── Finalisation ────────────────────────────────────────────────────

def finalize_output(jsonl_path: Path, output_dir: Path) -> None:
    """Lit le .jsonl et produit invoices.json + invoices.csv + invoices_lines.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    invoices = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            invoices.append(json.loads(line))

    # JSON array
    json_path = output_dir / "invoices.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(invoices, fh, ensure_ascii=False, indent=2, default=str)
    print(f"[SAVE] JSON  : {json_path}")

    # CSV factures (sans lignes ni raw_text ni _raw_results)
    import pandas as pd
    flat = []
    for inv in invoices:
        d = {k: v for k, v in inv.items() if k not in ("lignes", "raw_text", "_raw_results")}
        flat.append(d)
    df = pd.DataFrame(flat)
    csv_path = output_dir / "invoices.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] CSV   : {csv_path}")

    # CSV lignes
    lines = []
    for inv in invoices:
        lignes = inv.get("lignes") or []
        for li in lignes:
            ld = dict(li) if isinstance(li, dict) else {}
            ld["source_file"] = inv.get("source_file")
            ld["numero_facture"] = inv.get("numero_facture")
            ld["consensus_score"] = inv.get("consensus_score")
            lines.append(ld)
    if lines:
        df_lines = pd.DataFrame(lines)
        lines_path = output_dir / "invoices_lines.csv"
        df_lines.to_csv(lines_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] Lignes: {lines_path}")

    # Stats
    high = sum(1 for r in invoices if r.get("consensus_confidence") == "high")
    med = sum(1 for r in invoices if r.get("consensus_confidence") == "medium")
    low = sum(1 for r in invoices if r.get("consensus_confidence") == "low")
    scores = [r.get("consensus_score", 0) or 0 for r in invoices]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"\n[OK]  Traitees : {len(invoices)}")
    print(f"      High     : {high}")
    print(f"      Medium   : {med}")
    print(f"      Low      : {low}")
    print(f"      Score moy: {avg_score:.1f}%")
