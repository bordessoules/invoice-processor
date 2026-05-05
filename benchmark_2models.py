"""Benchmark : qwen3.6-35b-a3b vs gemma-4-26b-a4b
Compare champ par champ sur un echantillon de factures.
"""
import os, sys, json, time, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.pdf_utils import extract_text_from_pdf
from extractors.native import extract_from_text

# ── Config ─────────────────────────────────────────────────────────
SAMPLE_SIZE = 20
MODEL_QWEN = "qwen/qwen3.6-35b-a3b"
MODEL_GEMMA = "google/gemma-4-26b-a4b"
INPUT_DIR = r".\factures"

# ── Liste des champs à comparer ────────────────────────────────────
FIELDS = [
    "numero_facture",
    "date_facture",
    "date_echeance",
    "nom_fournisseur",
    "siret_fournisseur",
    "montant_ht",
    "montant_tva",
    "montant_ttc",
    "devise",
    "taux_tva",
    "nb_lignes",
    "type_document",
    "categorie",
    "moyen_paiement",
    "iban_fournisseur",
    "numero_tva",
    "numero_commande_client",
]


def normalize(val):
    """Normalise une valeur pour comparaison."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        # arrondi a 2 decimales pour les montants
        return round(float(val), 2)
    s = str(val).strip()
    if s.lower() in ("", "null", "none", "n/a", "nan"):
        return None
    # Normalisation texte : minuscules, sans accents basiques
    return s.lower()


def safe_get(invoice, field):
    if invoice is None:
        return None
    if field == "nb_lignes":
        lignes = getattr(invoice, "lignes", None)
        return len(lignes) if lignes else 0
    return getattr(invoice, field, None)


def field_match(v1, v2, field):
    """Compare deux valeurs pour un champ donne."""
    n1, n2 = normalize(v1), normalize(v2)
    if n1 is None and n2 is None:
        return True
    if n1 is None or n2 is None:
        return False
    # Comparaison numerique pour les montants
    if field in ("montant_ht", "montant_tva", "montant_ttc"):
        try:
            return abs(float(n1) - float(n2)) < 0.05
        except:
            return n1 == n2
    # Comparaison numerique pour taux_tva
    if field == "taux_tva":
        try:
            return abs(float(n1) - float(n2)) < 0.5
        except:
            return n1 == n2
    # Texte : egalite simple apres normalisation
    return n1 == n2


def extract_one(pdf_path, model_name):
    """Extrait une facture avec un modele donne."""
    text = extract_text_from_pdf(str(pdf_path))
    if not text or len(text) < 50:
        return None, "texte_vide"
    try:
        inv = extract_from_text(text, pdf_path.name, model=model_name)
        return inv, "ok"
    except Exception as exc:
        return None, str(exc)


def run():
    # --- Liste des PDFs ---
    pdfs = list(Path(INPUT_DIR).rglob("*.pdf"))
    if len(pdfs) < SAMPLE_SIZE:
        print(f"[WARN] Seulement {len(pdfs)} PDFs trouves, benchmark sur tout.")
        sample = pdfs
    else:
        random.seed(42)
        sample = random.sample(pdfs, SAMPLE_SIZE)

    print(f"Benchmark : {len(sample)} factures")
    print(f"  Model A : {MODEL_QWEN}")
    print(f"  Model B : {MODEL_GEMMA}")
    print("-" * 80)

    results = []
    for pdf_path in sample:
        filename = pdf_path.name
        print(f"\n{filename}")

        # Extraction sequentielle (evite de saturer LM Studio)
        t0 = time.time()
        inv_q, status_q = extract_one(pdf_path, MODEL_QWEN)
        t_q = time.time() - t0

        t0 = time.time()
        inv_g, status_g = extract_one(pdf_path, MODEL_GEMMA)
        t_g = time.time() - t0

        if inv_q is None:
            print(f"  [QWEN]  ERREUR: {status_q}")
        else:
            print(f"  [QWEN]  OK  ({t_q:.1f}s)  N={inv_q.numero_facture or '---'}  F={inv_q.nom_fournisseur or '---'}  HT={inv_q.montant_ht or '---'}")
        if inv_g is None:
            print(f"  [GEMMA] ERREUR: {status_g}")
        else:
            print(f"  [GEMMA] OK  ({t_g:.1f}s)  N={inv_g.numero_facture or '---'}  F={inv_g.nom_fournisseur or '---'}  HT={inv_g.montant_ht or '---'}")

        results.append({
            "file": filename,
            "qwen": inv_q,
            "gemma": inv_g,
            "time_qwen": t_q,
            "time_gemma": t_g,
        })

    # --- Tableau comparatif ---
    print("\n" + "=" * 100)
    print("COMPARAISON CHAMP PAR CHAMP")
    print("=" * 100)

    # Header
    header = f"{'Fichier':<45} | {'Champ':<22} | {'Qwen':<20} | {'Gemma':<20} | {'Match'}"
    print(header)
    print("-" * len(header))

    field_stats = defaultdict(lambda: {"match": 0, "total": 0})

    for r in results:
        file = r["file"]
        first_row = True
        diffs_for_file = []
        for field in FIELDS:
            vq = safe_get(r["qwen"], field)
            vg = safe_get(r["gemma"], field)
            match = field_match(vq, vg, field)
            field_stats[field]["total"] += 1
            if match:
                field_stats[field]["match"] += 1
            else:
                diffs_for_file.append((field, vq, vg))

        # Affiche uniquement les diffs (sinon ca fait trop de lignes)
        if diffs_for_file:
            for field, vq, vg in diffs_for_file:
                vq_s = str(vq)[:18] if vq is not None else "-"
                vg_s = str(vg)[:18] if vg is not None else "-"
                print(f"{file[:45]:<45} | {field:<22} | {vq_s:<20} | {vg_s:<20} | DIFF")
        else:
            print(f"{file[:45]:<45} | {'(tous champs OK)':<22} | {'':<20} | {'':<20} | OK")

    # --- Stats globales ---
    print("\n" + "=" * 100)
    print("SCORES PAR CHAMP (pourcentage d'accord)")
    print("=" * 100)
    print(f"{'Champ':<25} | {'Match':>6} | {'Total':>6} | {'%':>6}")
    print("-" * 50)
    total_match = 0
    total_fields = 0
    for field in FIELDS:
        st = field_stats[field]
        pct = (st["match"] / st["total"] * 100) if st["total"] else 0
        total_match += st["match"]
        total_fields += st["total"]
        marker = "  ***" if pct < 70 else ""
        print(f"{field:<25} | {st['match']:>6} | {st['total']:>6} | {pct:>5.1f}%{marker}")

    overall = (total_match / total_fields * 100) if total_fields else 0
    print("-" * 50)
    print(f"{'GLOBAL':<25} | {total_match:>6} | {total_fields:>6} | {overall:>5.1f}%")

    # --- Temps ---
    avg_q = sum(r["time_qwen"] for r in results) / len(results)
    avg_g = sum(r["time_gemma"] for r in results) / len(results)
    print(f"\nTemps moyen : Qwen={avg_q:.1f}s  Gemma={avg_g:.1f}s  (ratio {avg_g/avg_q:.1f}x)")

    # --- Sauvegarde JSON ---
    out = {
        "models": {"qwen": MODEL_QWEN, "gemma": MODEL_GEMMA},
        "sample_size": len(sample),
        "field_stats": dict(field_stats),
        "overall_pct": overall,
        "avg_time_qwen": avg_q,
        "avg_time_gemma": avg_g,
    }
    with open("benchmark_2models_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print("\n[OK] Resultats sauvegardes dans benchmark_2models_results.json")


if __name__ == "__main__":
    run()
