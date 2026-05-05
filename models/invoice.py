from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from decimal import Decimal


class LigneFacture(BaseModel):
    description: Optional[str] = Field(None, description="Description / libelle de la ligne")
    quantite: Optional[float] = Field(None, description="Quantite")
    prix_unitaire_ht: Optional[Decimal] = Field(None, description="Prix unitaire HT")
    montant_ht: Optional[Decimal] = Field(None, description="Montant HT de la ligne")
    taux_tva: Optional[float] = Field(None, description="Taux de TVA en %, ex: 20.0")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        if not isinstance(data, dict):
            return data
        mapping = {
            "designation": "description",
            "libelle": "description",
            "total_ligne_ht": "montant_ht",
            "montant_tva_ligne": "montant_tva",
            "prix_unitaire": "prix_unitaire_ht",
            "pu_ht": "prix_unitaire_ht",
        }
        for old, new in mapping.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        return data


class Invoice(BaseModel):
    numero_facture: Optional[str] = Field(None, description="Numero de facture")
    date_facture: Optional[str] = Field(None, description="Date de facture (ISO YYYY-MM-DD)")
    nom_fournisseur: Optional[str] = Field(None, description="Nom / Raison sociale du fournisseur")
    siret_fournisseur: Optional[str] = Field(None, description="SIRET du fournisseur")
    montant_ht: Optional[Decimal] = Field(None, description="Montant total HT")
    montant_tva: Optional[Decimal] = Field(None, description="Montant total TVA")
    montant_ttc: Optional[Decimal] = Field(None, description="Montant total TTC")
    devise: Optional[str] = Field("EUR", description="Devise")
    taux_tva: Optional[float] = Field(None, description="Taux de TVA principal en %")
    lignes: Optional[List[LigneFacture]] = Field(None, description="Lignes de detail")
    type_document: Optional[str] = Field(None, description="facture / avoir / remboursement / ticket / devis")
    categorie: Optional[str] = Field(None, description="Categorie: informatique, telecom, transport, assurance, alimentation, energie, meubles, autres")
    moyen_paiement: Optional[str] = Field(None, description="CB, virement, prelevement, especes, paypal, autre")
    numero_tva: Optional[str] = Field(None, description="Numero de TVA intracommunautaire si visible")
    numero_commande_client: Optional[str] = Field(None, description="Numero de commande client si present")
    url_fichier: Optional[str] = Field(None, description="Chemin complet du fichier PDF")
    raw_text: Optional[str] = Field(None, description="Texte brut extrait")
    extraction_method: Optional[str] = Field(None, description="native_text | vision | consensus | failed")
    source_file: Optional[str] = Field(None, description="Nom du fichier source")
    confidence: Optional[str] = Field("medium", description="high | medium | low")

    # --- Champs de consensus ---
    consensus_score: Optional[float] = Field(None, description="Score global de consensus 0-100")
    consensus_confidence: Optional[str] = Field(None, description="high | medium | low")
    consensus_votes: Optional[dict] = Field(None, description="Nombre de votes par champ")
    consensus_winner_route: Optional[str] = Field(None, description="Route gagnante globale")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        if not isinstance(data, dict):
            return data
        mapping = {
            "numero": "numero_facture",
            "n_facture": "numero_facture",
            "date": "date_facture",
            "fournisseur": "nom_fournisseur",
            "total_ht": "montant_ht",
            "total_tva": "montant_tva",
            "total_ttc": "montant_ttc",
        }
        for old, new in mapping.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        # Nettoie les taux TVA mal formates (ex: "20%" -> 20.0)
        for tva_key in ("taux_tva",):
            if tva_key in data and isinstance(data[tva_key], str):
                s = data[tva_key].replace("%", "").replace(",", ".").strip()
                try:
                    data[tva_key] = float(s)
                except ValueError:
                    pass
        return data
