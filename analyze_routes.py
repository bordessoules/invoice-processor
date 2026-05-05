"""Analyse les resultats par route dans la base SQLite."""
import sys, io, json
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from collections import defaultdict
from db import _get_conn
from utils.consensus import _match


def main():
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM llm_extractions").fetchall()
    conn.close()

    # Stats par route
    routes = defaultdict(lambda: {"ok": 0, "err": 0, "total": 0})
    for r in rows:
        k = f"{r['llm_name']}_{r['mode']}"
        routes[k]["total"] += 1
        routes[k]["ok" if r["status"] == "ok" else "err"] += 1

    print("=== TAUX DE SUCCES PAR ROUTE ===")
    for k, v in sorted(routes.items()):
        pct = v["ok"] / v["total"] * 100
        print(f"  {k:<20}: {v['ok']:>3}/{v['total']} OK ({pct:>5.1f}%)")

    # Champs les plus discordants
    by_pdf = defaultdict(list)
    for r in rows:
        if r["status"] == "ok" and r["result_json"]:
            by_pdf[r["pdf_path"]].append(json.loads(r["result_json"]))

    field_agree = defaultdict(lambda: {"match": 0, "total": 0})
    for pdf_path, results in by_pdf.items():
        if len(results) < 2:
            continue
        for field in ["numero_facture", "date_facture", "nom_fournisseur", "montant_ht", "montant_ttc", "montant_tva", "taux_tva", "type_document", "categorie", "moyen_paiement"]:
            vals = [r.get(field) for r in results]
            non_null = [v for v in vals if v is not None]
            if len(non_null) <= 1:
                continue
            # Au moins 2 routes d'accord ?
            agree = False
            for i, vi in enumerate(vals):
                if vi is None:
                    continue
                cnt = sum(1 for vj in vals if vj is not None and _match(vi, vj, field))
                if cnt >= 2:
                    agree = True
                    break
            field_agree[field]["total"] += 1
            if agree:
                field_agree[field]["match"] += 1

    print("\n=== ACCORD ENTRE ROUTES (%) ===")
    for field, st in sorted(field_agree.items(), key=lambda x: x[1]["match"] / x[1]["total"]):
        pct = st["match"] / st["total"] * 100 if st["total"] else 0
        print(f"  {field:<20}: {st['match']:>3}/{st['total']} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
