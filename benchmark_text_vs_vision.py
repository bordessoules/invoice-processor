"""Benchmark : Texte vs Vision pour Qwen et Gemma-26b
Compare les 4 combinaisons sur le meme echantillon.
Format de sortie optimise pour comparer facture par facture.
"""
import os, sys, json, time, random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from utils.pdf_utils import extract_text_from_pdf, pdf_pages_to_base64_list
from utils.llm_utils import extract_json
from models.invoice import Invoice
import config

# ── Config ─────────────────────────────────────────────────────────
SAMPLE_SIZE = 10
MODEL_QWEN = "qwen/qwen3.6-35b-a3b"
MODEL_GEMMA = "google/gemma-4-26b-a4b"
INPUT_DIR = r".\factures"

# Parametres de sampling par modele (fournis par l'utilisateur)
SAMPLING = {
    MODEL_QWEN: {
        "temperature": 0.7,
        "top_p": 0.80,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    MODEL_GEMMA: {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    },
}

client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY, timeout=120.0)

_SYSTEM_TEXT = (
    "Tu extrais des factures. Reponds UNIQUEMENT en JSON.\n"
    "Champs obligatoires (noms EXACTS) :\n"
    "- numero_facture, date_facture (YYYY-MM-DD), nom_fournisseur, siret_fournisseur,\n"
    "  montant_ht, montant_tva, montant_ttc, devise, taux_tva,\n"
    "  lignes [{description, quantite, prix_unitaire_ht, montant_ht, taux_tva}],\n"
    "- type_document (facture / avoir / remboursement / ticket),\n"
    "- categorie (informatique, telecom, transport, assurance, alimentation, energie, autres),\n"
    "- moyen_paiement (CB, virement, prelevement, especes),\n"
    "- iban_fournisseur (string ou null),\n"
    "- numero_tva (numero TVA intracommunautaire ou null),\n"
    "- numero_commande_client (string ou null).\n"
    "Regles : champs absents -> null. Pas de markdown, juste le JSON."
)

_SYSTEM_VISION = (
    "Tu vois une ou plusieurs images de pages d'une facture. Reponds UNIQUEMENT en JSON.\n"
    "Champs obligatoires (noms EXACTS) :\n"
    "- numero_facture, date_facture (YYYY-MM-DD), nom_fournisseur, siret_fournisseur,\n"
    "  montant_ht, montant_tva, montant_ttc, devise, taux_tva,\n"
    "  lignes [{description, quantite, prix_unitaire_ht, montant_ht, taux_tva}],\n"
    "- type_document (facture / avoir / remboursement / ticket),\n"
    "- categorie (informatique, telecom, transport, assurance, alimentation, energie, autres),\n"
    "- moyen_paiement (CB, virement, prelevement, especes),\n"
    "- iban_fournisseur (string ou null),\n"
    "- numero_tva (numero TVA intracommunautaire ou null),\n"
    "- numero_commande_client (string ou null).\n"
    "Regles : champs absents -> null. Pas de markdown, juste le JSON."
)

# ── Champs affiches dans le tableau par facture ───────────────────
DISPLAY_FIELDS = [
    ("numero_facture", "N°"),
    ("date_facture", "Date"),
    ("nom_fournisseur", "Fournisseur"),
    ("montant_ht", "HT"),
    ("montant_tva", "TVA"),
    ("montant_ttc", "TTC"),
    ("taux_tva", "%TVA"),
    ("nb_lignes", "Lignes"),
    ("type_document", "Type"),
    ("categorie", "Cat"),
]

FIELDS = [f for f, _ in DISPLAY_FIELDS]


def normalize(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    s = str(val).strip()
    if s.lower() in ("", "null", "none", "n/a", "nan"):
        return None
    return s.lower()


def safe_get(invoice, field):
    if invoice is None:
        return None
    if field == "nb_lignes":
        lignes = getattr(invoice, "lignes", None)
        return len(lignes) if lignes else 0
    return getattr(invoice, field, None)


def field_match(v1, v2, field):
    n1, n2 = normalize(v1), normalize(v2)
    if n1 is None and n2 is None:
        return True
    if n1 is None or n2 is None:
        return False
    if field in ("montant_ht", "montant_tva", "montant_ttc"):
        try:
            return abs(float(n1) - float(n2)) < 0.05
        except:
            return n1 == n2
    if field == "taux_tva":
        try:
            return abs(float(n1) - float(n2)) < 0.5
        except:
            return n1 == n2
    return n1 == n2


def fmt(val, width=16):
    if val is None:
        return "-".rjust(width)
    s = str(val)[:width]
    return s.rjust(width)


def build_extra(cfg):
    extra = {}
    for k in ("top_k", "min_p", "repetition_penalty"):
        if k in cfg:
            extra[k] = cfg[k]
    return extra if extra else None


def extract_text(pdf_path, model):
    text = extract_text_from_pdf(str(pdf_path))
    if not text or len(text) < 50:
        return None, "texte_vide"
    cfg = SAMPLING.get(model, {"temperature": 0.7})
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_TEXT},
                {"role": "user", "content": text[:6000]},
            ],
            temperature=cfg["temperature"],
            top_p=cfg.get("top_p", 1.0),
            presence_penalty=cfg.get("presence_penalty", 0.0),
            frequency_penalty=cfg.get("frequency_penalty", 0.0),
            extra_body=build_extra(cfg),
            max_tokens=2048,
        )
        data = extract_json(resp.choices[0].message.content)
        inv = Invoice.model_validate(data)
        inv.extraction_method = "native_text"
        inv.source_file = pdf_path.name
        return inv, "ok"
    except Exception as exc:
        return None, str(exc)[:80]


def extract_vision(pdf_path, model):
    b64_images = pdf_pages_to_base64_list(str(pdf_path), dpi=config.PDF_DPI, max_pages=5)
    if not b64_images:
        return None, "pas_d_images"
    content = [{"type": "text", "text": f"Extrais cette facture ({len(b64_images)} page(s)):"}]
    for b64 in b64_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    cfg = SAMPLING.get(model, {"temperature": 0.7})
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_VISION},
                {"role": "user", "content": content},
            ],
            temperature=cfg["temperature"],
            top_p=cfg.get("top_p", 1.0),
            presence_penalty=cfg.get("presence_penalty", 0.0),
            frequency_penalty=cfg.get("frequency_penalty", 0.0),
            extra_body=build_extra(cfg),
            max_tokens=2048,
        )
        data = extract_json(resp.choices[0].message.content)
        inv = Invoice.model_validate(data)
        inv.extraction_method = "vision"
        inv.source_file = pdf_path.name
        return inv, "ok"
    except Exception as exc:
        return None, str(exc)[:80]


def run():
    pdfs = list(Path(INPUT_DIR).rglob("*.pdf"))
    if len(pdfs) < SAMPLE_SIZE:
        sample = pdfs
    else:
        random.seed(42)
        sample = random.sample(pdfs, SAMPLE_SIZE)

    print(f"Benchmark TEXT vs VISION : {len(sample)} factures")
    print(f"  Qwen  : {MODEL_QWEN}   (sampling: {SAMPLING[MODEL_QWEN]})")
    print(f"  Gemma : {MODEL_GEMMA}  (sampling: {SAMPLING[MODEL_GEMMA]})")
    print("-" * 100)

    results = []
    for pdf_path in sample:
        filename = pdf_path.name
        print(f"\n{filename}")
        r = {"file": filename}

        for key, model, fn in [
            ("qw_t", MODEL_QWEN, extract_text),
            ("qw_v", MODEL_QWEN, extract_vision),
            ("gm_t", MODEL_GEMMA, extract_text),
            ("gm_v", MODEL_GEMMA, extract_vision),
        ]:
            t0 = time.time()
            r[key], status = fn(pdf_path, model)
            r[f"t_{key}"] = time.time() - t0
            label = {"qw_t": "Q-TXT", "qw_v": "Q-VIS", "gm_t": "G-TXT", "gm_v": "G-VIS"}[key]
            ok = "OK" if r[key] else f"ERR:{status[:40]}"
            print(f"  [{label}] {r[f't_{key}']:.1f}s {ok}")

        results.append(r)

    # ── Tableau comparatif par facture ────────────────────────────
    print("\n" + "=" * 130)
    print("COMPARAISON FACTURE PAR FACTURE")
    print("=" * 130)

    for r in results:
        print(f"\n>>> {r['file']}")
        header = f"{'Champ':<16} | {'Qwen-TXT':>16} | {'Qwen-VIS':>16} | {'Gemma-TXT':>16} | {'Gemma-VIS':>16} | {'Accord'}"
        print(header)
        print("-" * len(header))

        for field, label in DISPLAY_FIELDS:
            vals = {
                "qw_t": safe_get(r["qw_t"], field),
                "qw_v": safe_get(r["qw_v"], field),
                "gm_t": safe_get(r["gm_t"], field),
                "gm_v": safe_get(r["gm_v"], field),
            }
            # Determine si les 4 sont d'accord
            all_match = True
            ref = None
            for k, v in vals.items():
                if ref is None:
                    ref = v
                elif not field_match(ref, v, field):
                    all_match = False

            markers = "OK" if all_match else "  "
            print(f"{label:<16} | {fmt(vals['qw_t'])} | {fmt(vals['qw_v'])} | {fmt(vals['gm_t'])} | {fmt(vals['gm_v'])} | [{markers}]")

    # ── Stats globales ────────────────────────────────────────────
    print("\n" + "=" * 130)
    print("SCORES PAR CHAMP (accord entre les 4 routes)")
    print("=" * 130)
    print(f"{'Champ':<16} | {'Match':>6} | {'Total':>6} | {'%':>6}")
    print("-" * 45)

    stats = defaultdict(lambda: {"match": 0, "total": 0})
    for r in results:
        for field, _ in DISPLAY_FIELDS:
            vals = [safe_get(r[k], field) for k in ("qw_t", "qw_v", "gm_t", "gm_v")]
            # Compte comme "match" si au moins 3 des 4 sont d'accord (majorite)
            # ou si tous les non-null sont d'accord
            non_null = [(i, v) for i, v in enumerate(vals) if v is not None]
            if len(non_null) <= 1:
                match = True  # pas assez de donnees pour juger
            else:
                # Regarde si au moins 3 sont identiques
                match = False
                for i, vi in enumerate(vals):
                    if vi is None:
                        continue
                    cnt = sum(1 for vj in vals if vj is not None and field_match(vi, vj, field))
                    if cnt >= 3:
                        match = True
                        break
            stats[field]["total"] += 1
            if match:
                stats[field]["match"] += 1

    total_m = 0
    total_t = 0
    for field, label in DISPLAY_FIELDS:
        st = stats[field]
        pct = (st["match"] / st["total"] * 100) if st["total"] else 0
        total_m += st["match"]
        total_t += st["total"]
        marker = "  ***" if pct < 70 else ""
        print(f"{label:<16} | {st['match']:>6} | {st['total']:>6} | {pct:>5.1f}%{marker}")

    print("-" * 45)
    overall = (total_m / total_t * 100) if total_t else 0
    print(f"{'GLOBAL':<16} | {total_m:>6} | {total_t:>6} | {overall:>5.1f}%")

    # ── Temps ─────────────────────────────────────────────────────
    print("\n" + "=" * 130)
    print("TEMPS MOYENS")
    print("=" * 130)
    for key, label in [("t_qw_t", "Qwen-Text"), ("t_qw_v", "Qwen-Vis"), ("t_gm_t", "Gemma-Text"), ("t_gm_v", "Gemma-Vis")]:
        vals = [r[key] for r in results]
        ok = sum(1 for r in results if r[key.replace("t_", "")] is not None)
        print(f"  {label:<15} : {sum(vals)/len(vals):.1f}s  (min={min(vals):.1f} max={max(vals):.1f})  [{ok}/{len(vals)} OK]")

    # Sauvegarde
    out = {
        "models": {"qwen": MODEL_QWEN, "gemma": MODEL_GEMMA},
        "sampling": SAMPLING,
        "sample_size": len(sample),
        "field_stats": dict(stats),
        "overall_pct": overall,
    }
    with open("benchmark_text_vs_vision_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print("\n[OK] Resultats sauvegardes dans benchmark_text_vs_vision_results.json")


if __name__ == "__main__":
    run()
