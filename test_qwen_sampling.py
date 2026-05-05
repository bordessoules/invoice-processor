"""Test rapide des parametres de sampling Qwen sur 2 factures."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from utils.pdf_utils import extract_text_from_pdf
from utils.llm_utils import extract_json
from pathlib import Path
import config

client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY, timeout=60.0)
MODEL = "qwen/qwen3.6-35b-a3b"

_SYSTEM = (
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

CONFIGS = {
    "general": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "reasoning": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
}

_factures_dir = Path(r".\factures")
PDFS = [
    next(_factures_dir.rglob("Bouyguestelecom_Facture_20250516.pdf")),
    next(_factures_dir.rglob("ES25FC04264606.pdf")),
]


def test_pdf(pdf_path, cfg_name, cfg):
    text = extract_text_from_pdf(str(pdf_path))
    if not text:
        print(f"  [{cfg_name:10}] {pdf_path.name:<50} -> ERR | texte vide (fichier non trouve?)")
        return False
    extra = {k: v for k, v in cfg.items() if k in ("top_k", "min_p", "repetition_penalty")}
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": text[:6000]},
            ],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            presence_penalty=cfg.get("presence_penalty", 0.0),
            frequency_penalty=0.0,
            extra_body=extra if extra else None,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content
        data = extract_json(raw)
        ok = True
        msg = f"OK  | N={data.get('numero_facture','---')}  F={data.get('nom_fournisseur','---')}  HT={data.get('montant_ht','---')}"
    except Exception as exc:
        ok = False
        msg = f"ERR | {str(exc)[:60]}"
    print(f"  [{cfg_name:10}] {pdf_path.name:<50} -> {msg}")
    return ok


print(f"Test rapide Qwen ({MODEL}) sur 2 factures x 2 configs = 4 appels\n")
results = []
for pdf in PDFS:
    print(f"{pdf.name}")
    for name, cfg in CONFIGS.items():
        results.append((pdf.name, name, test_pdf(pdf, name, cfg)))

print("\n--- Recap ---")
ok_general = sum(1 for _, n, o in results if n == "general" and o)
ok_reasoning = sum(1 for _, n, o in results if n == "reasoning" and o)
print(f"General  : {ok_general}/2 JSON valides")
print(f"Reasoning: {ok_reasoning}/2 JSON valides")
