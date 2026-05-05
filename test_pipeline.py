from pathlib import Path
from pipeline import process_one

pdf = Path("factures/accessoiresasus.com/Facture-FC0082368.pdf")
result = process_one(pdf, "pdfplumber")

print("\n--- RESULTAT ---")
print(f"numero_facture: {result['numero_facture']}")
print(f"date_facture: {result['date_facture']}")
print(f"nom_fournisseur: {result['nom_fournisseur']}")
print(f"montant_ht: {result['montant_ht']}")
print(f"montant_ttc: {result['montant_ttc']}")
print(f"consensus_score: {result['consensus_score']}")
print(f"consensus_confidence: {result['consensus_confidence']}")
print(f"lignes: {len(result.get('lignes', []))} ligne(s)")
print(f"type_document: {result['type_document']}")
print(f"categorie: {result['categorie']}")
