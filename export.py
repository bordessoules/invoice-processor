"""Export les reviews manuelles en CSV pour la déclaration TVA.

Génère 2 fichiers dans output/ :
  - export_synthese.csv : 1 ligne par facture (pour déclaration et totaux par poste)
  - export_lignes.csv   : 1 ligne par item de ligne (détail produits/services)
"""
import csv
import json
import sqlite3
from pathlib import Path

DB_PATH = "output/invoices.db"
OUT_SYNTH = Path("output/export_synthese.csv")
OUT_LIGNES = Path("output/export_lignes.csv")


def main():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT pdf_path, status, reviewed_at, final_data_json FROM manual_review ORDER BY pdf_path"
    ).fetchall()
    conn.close()

    if not rows:
        print("[ERR] Aucune review en base.")
        return

    # ── Synthese (1 ligne par facture) ────────────────────────────
    synth_fields = [
        "fichier", "dossier", "status",
        "date_facture", "categorie", "nom_fournisseur",
        "numero_facture", "type_document", "moyen_paiement",
        "montant_ht", "montant_tva", "montant_ttc", "taux_tva",
        "siret_fournisseur", "numero_tva", "numero_commande_client",
        "nb_lignes", "lignes_source",
        "reviewed_at", "pdf_path",
    ]

    OUT_SYNTH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SYNTH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=synth_fields)
        w.writeheader()
        for path, status, reviewed_at, raw in rows:
            d = json.loads(raw) if raw else {}
            p = Path(path)
            w.writerow({
                "fichier": p.name,
                "dossier": p.parent.name,
                "status": status,
                "date_facture": d.get("date_facture"),
                "categorie": d.get("categorie"),
                "nom_fournisseur": d.get("nom_fournisseur"),
                "numero_facture": d.get("numero_facture"),
                "type_document": d.get("type_document"),
                "moyen_paiement": d.get("moyen_paiement"),
                "montant_ht": d.get("montant_ht"),
                "montant_tva": d.get("montant_tva"),
                "montant_ttc": d.get("montant_ttc"),
                "taux_tva": d.get("taux_tva"),
                "siret_fournisseur": d.get("siret_fournisseur"),
                "numero_tva": d.get("numero_tva"),
                "numero_commande_client": d.get("numero_commande_client"),
                "nb_lignes": len(d.get("lignes") or []),
                "lignes_source": d.get("_lignes_source"),
                "reviewed_at": reviewed_at,
                "pdf_path": path,
            })

    # ── Lignes (1 ligne par item, avec rappel facture) ───────────
    lignes_fields = [
        "fichier", "date_facture", "categorie", "nom_fournisseur", "status",
        "description", "quantite", "prix_unitaire_ht", "montant_ht", "taux_tva",
        "pdf_path",
    ]

    n_lignes = 0
    with OUT_LIGNES.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lignes_fields)
        w.writeheader()
        for path, status, _, raw in rows:
            d = json.loads(raw) if raw else {}
            for l in d.get("lignes") or []:
                if not isinstance(l, dict):
                    continue
                w.writerow({
                    "fichier": Path(path).name,
                    "date_facture": d.get("date_facture"),
                    "categorie": d.get("categorie"),
                    "nom_fournisseur": d.get("nom_fournisseur"),
                    "status": status,
                    "description": l.get("description"),
                    "quantite": l.get("quantite"),
                    "prix_unitaire_ht": l.get("prix_unitaire_ht"),
                    "montant_ht": l.get("montant_ht"),
                    "taux_tva": l.get("taux_tva"),
                    "pdf_path": path,
                })
                n_lignes += 1

    # ── Stats ────────────────────────────────────────────────────
    print(f"[OK] {OUT_SYNTH}  ({len(rows)} factures)")
    print(f"[OK] {OUT_LIGNES}  ({n_lignes} lignes)")

    # Aggreges par poste pour aperçu console
    from collections import defaultdict
    totaux = defaultdict(lambda: {"ht": 0.0, "tva": 0.0, "ttc": 0.0, "n": 0})
    for path, status, _, raw in rows:
        if status != "validated":
            continue
        d = json.loads(raw) if raw else {}
        cat = d.get("categorie") or "(non renseigne)"
        try:
            totaux[cat]["ht"] += float(d.get("montant_ht") or 0)
            totaux[cat]["tva"] += float(d.get("montant_tva") or 0)
            totaux[cat]["ttc"] += float(d.get("montant_ttc") or 0)
            totaux[cat]["n"] += 1
        except (ValueError, TypeError):
            pass

    print()
    print("=== Totaux par poste comptable (status=validated) ===")
    print(f"{'poste':<22} {'#':>4} {'HT':>10} {'TVA':>10} {'TTC':>10}")
    print("-" * 60)
    for cat in sorted(totaux):
        t = totaux[cat]
        print(f"{cat:<22} {t['n']:>4} {t['ht']:>10.2f} {t['tva']:>10.2f} {t['ttc']:>10.2f}")
    print("-" * 60)
    grand = {"n":0, "ht":0.0, "tva":0.0, "ttc":0.0}
    for t in totaux.values():
        for k in ("n","ht","tva","ttc"):
            grand[k] += t[k]
    print(f"{'TOTAL':<22} {grand['n']:>4} {grand['ht']:>10.2f} {grand['tva']:>10.2f} {grand['ttc']:>10.2f}")


if __name__ == "__main__":
    main()
