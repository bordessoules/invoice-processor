import pandas as pd

df = pd.read_csv("output/invoices.csv")
print("=== APERCU FACTURES ===")
cols = ["source_file", "numero_facture", "date_facture", "nom_fournisseur",
        "montant_ttc", "type_document", "categorie", "consensus_score", "consensus_confidence"]
print(df[cols].to_string(index=False))

print("\n=== STATS ===")
print(df["consensus_confidence"].value_counts())
print(f"\nScore moyen: {df['consensus_score'].mean():.1f}%")
print(f"Min: {df['consensus_score'].min():.1f}%  Max: {df['consensus_score'].max():.1f}%")

print("\n=== TOP ERREURS (score < 70) ===")
low = df[df["consensus_score"] < 70]
if len(low):
    print(low[["source_file", "consensus_score"]].to_string(index=False))
else:
    print("Aucune")
