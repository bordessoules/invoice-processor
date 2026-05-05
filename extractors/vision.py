"""Skill d'extraction via vision (fallback pour scans ou PDF image)."""
from openai import OpenAI
from models.invoice import Invoice
from utils.llm_utils import extract_json
import config
from utils.pdf_utils import pdf_pages_to_base64_list

_client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY, timeout=60.0)

_SYSTEM = (
    "Tu vois une ou plusieurs images de pages d'un document commercial. Reponds UNIQUEMENT en JSON.\n"
    "Champs obligatoires (noms EXACTS, null si absent) :\n"
    "- numero_facture : string ou null\n"
    "- date_facture : YYYY-MM-DD ou null\n"
    "- nom_fournisseur : string ou null\n"
    "- siret_fournisseur : string ou null\n"
    "- montant_ht, montant_tva, montant_ttc : nombre ou null\n"
    "- devise : string (default EUR)\n"
    "- taux_tva : nombre ou null (ex: 20.0)\n"
    "- lignes : [{description, quantite, prix_unitaire_ht, montant_ht, taux_tva}] ou []\n"
    "- type_document : facture / avoir / remboursement / ticket / devis / null\n"
    "- categorie : informatique / telecom / transport / assurance / alimentation / energie / meubles / autres / null\n"
    "- moyen_paiement : CB / virement / prelevement / especes / paypal / autre / null\n"
    "- numero_tva : string ou null\n"
    "- numero_commande_client : string ou null\n"
    "\n"
    "REGLES STRICTES :\n"
    "1. Si un champ n'est pas visible dans l'image, mets EXACTEMENT null. Ne l'invente JAMAIS.\n"
    "2. prix_unitaire_ht doit etre le prix HORS TAXE, pas TTC.\n"
    "3. Liste chaque produit/service distinct dans 'lignes'. NE PAS agreger en categories generiques.\n"
    "4. numero_commande_client est le numero de commande du CLIENT, pas le numero de facture fournisseur.\n"
    "5. Pas de markdown, juste le JSON brut."
)


def _build_extra(sampling: dict) -> dict:
    extra = {}
    for k in ("top_k", "min_p", "repetition_penalty"):
        if k in sampling:
            extra[k] = sampling[k]
    return extra if extra else None


def extract_from_vision(pdf_path: str, filename: str, model: str = None, sampling: dict = None) -> Invoice:
    model = model or config.MODEL_VISION
    sampling = sampling or {"temperature": 0.7, "top_p": 0.8}
    b64_images = pdf_pages_to_base64_list(pdf_path, dpi=config.PDF_DPI, max_pages=3)

    content = [
        {"type": "text", "text": f"Extrais ce document ({len(b64_images)} page(s)) :"},
    ]
    for b64 in b64_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": content},
        ],
        temperature=sampling.get("temperature", 0.7),
        top_p=sampling.get("top_p", 0.8),
        presence_penalty=sampling.get("presence_penalty", 0.0),
        frequency_penalty=sampling.get("frequency_penalty", 0.0),
        extra_body=_build_extra(sampling),
        max_tokens=2048,
    )
    data = extract_json(resp.choices[0].message.content)
    invoice = Invoice.model_validate(data)
    invoice.extraction_method = "vision"
    invoice.source_file = filename
    return invoice
