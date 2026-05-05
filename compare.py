"""Consensus a la volee depuis la base SQLite."""
import sys, io, json
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import pandas as pd
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
from db import _get_conn
from utils.consensus import _normalize, _match, FIELD_WEIGHTS


def get_all_ok_extractions():
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM llm_extractions WHERE status='ok' ORDER BY pdf_path, id").fetchall()
    conn.close()
    return rows


def compute_consensus(results: list[dict], field: str):
    routes = {f"r{i}": r.get(field) for i, r in enumerate(results)}
    groups = []
    assigned = set()
    for k in routes:
        if k in assigned:
            continue
        val = routes[k]
        group = {k}
        for k2 in routes:
            if k2 == k or k2 in assigned:
                continue
            if _match(val, routes[k2], field):
                group.add(k2)
        repr_val = next((routes[k3] for k3 in routes if k3 in group and routes[k3] is not None), None)
        groups.append((repr_val, group))
        assigned.update(group)
    groups.sort(key=lambda g: len(g[1]), reverse=True)
    winner_val, winner_routes = groups[0]
    votes = len(winner_routes)
    score = (votes / len(results)) * 100 if results else 0
    return {"value": winner_val, "votes": votes, "score": round(score, 1)}


def main():
    rows = get_all_ok_extractions()
    print(f"[COMPARE] {len(rows)} extractions OK chargees")

    # Groupe par PDF
    by_pdf = defaultdict(list)
    for r in rows:
        d = json.loads(r["result_json"]) if r["result_json"] else {}
        by_pdf[r["pdf_path"]].append(d)

    results = []
    for pdf_path, extractions in by_pdf.items():
        consensus_fields = {}
        field_scores = {}
        total_weight = 0
        weighted_sum = 0.0

        for field in FIELD_WEIGHTS:
            c = compute_consensus(extractions, field)
            consensus_fields[field] = c["value"]
            field_scores[field] = c["score"]
            weight = FIELD_WEIGHTS.get(field, 1)
            total_weight += weight
            weighted_sum += c["score"] * weight

        global_score = round(weighted_sum / total_weight, 1) if total_weight else 0.0
        confidence = "high" if global_score >= 90 else ("medium" if global_score >= 60 else "low")

        results.append({
            "pdf_path": pdf_path,
            "numero_facture": consensus_fields.get("numero_facture"),
            "date_facture": consensus_fields.get("date_facture"),
            "nom_fournisseur": consensus_fields.get("nom_fournisseur"),
            "montant_ht": consensus_fields.get("montant_ht"),
            "montant_ttc": consensus_fields.get("montant_ttc"),
            "montant_tva": consensus_fields.get("montant_tva"),
            "devise": consensus_fields.get("devise"),
            "taux_tva": consensus_fields.get("taux_tva"),
            "type_document": consensus_fields.get("type_document"),
            "categorie": consensus_fields.get("categorie"),
            "consensus_score": global_score,
            "consensus_confidence": confidence,
            "nb_routes": len(extractions),
        })

    df = pd.DataFrame(results)
    out = Path("output/compare.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[COMPARE] {out}  ({len(df)} PDFs)")
    print(f"  Score moyen: {df['consensus_score'].mean():.1f}%")
    print(f"  High: {len(df[df['consensus_confidence']=='high'])}  Medium: {len(df[df['consensus_confidence']=='medium'])}  Low: {len(df[df['consensus_confidence']=='low'])}")


if __name__ == "__main__":
    main()
