"""Compare pdfplumber+Qwen vs MinerU+Qwen et calcule le consensus ameliore."""
import sys, io, json
from pathlib import Path
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import pandas as pd
from db import _get_conn
from utils.consensus import _match, FIELD_WEIGHTS


def compute_consensus(results: list[dict], field: str):
    vals = [r.get(field) for r in results]
    groups = []
    assigned = [False] * len(vals)
    for i, vi in enumerate(vals):
        if assigned[i]:
            continue
        group = {i}
        for j, vj in enumerate(vals):
            if j == i or assigned[j]:
                continue
            if _match(vi, vj, field):
                group.add(j)
        repr_val = next((vals[k] for k in group if vals[k] is not None), None)
        groups.append((repr_val, group))
        for k in group:
            assigned[k] = True
    groups.sort(key=lambda g: len(g[1]), reverse=True)
    winner_val, winner_routes = groups[0]
    votes = len(winner_routes)
    score = (votes / len(results)) * 100 if results else 0
    return {"value": winner_val, "votes": votes, "score": round(score, 1)}


def main():
    conn = _get_conn()
    rows = conn.execute("""
        SELECT pdf_path, pdf_extractor, result_json, status
        FROM llm_extractions
        WHERE llm_name='qwen3.6-35b' AND mode='text' AND status='ok'
        AND (pdf_extractor='mineru' OR pdf_extractor='pdfplumber')
    """).fetchall()
    conn.close()

    by_pdf = {}
    for r in rows:
        by_pdf.setdefault(r["pdf_path"], {})[r["pdf_extractor"]] = json.loads(r["result_json"]) if r["result_json"] else {}

    fields = list(FIELD_WEIGHTS.keys())
    results = []
    diffs = []

    for pdf_path, extr in by_pdf.items():
        if "mineru" not in extr or "pdfplumber" not in extr:
            continue
        m = extr["mineru"]
        p = extr["pdfplumber"]

        # Diffs mineru vs pdfplumber
        for f in fields:
            if not _match(m.get(f), p.get(f), f):
                diffs.append({"pdf": pdf_path.split("\\")[-1], "field": f, "mineru": m.get(f), "pdfplumber": p.get(f)})

        # Consensus avec les 2 extracteurs
        consensus_fields = {}
        field_scores = {}
        total_weight = 0
        weighted_sum = 0.0
        for f in fields:
            c = compute_consensus([m, p], f)
            consensus_fields[f] = c["value"]
            field_scores[f] = c["score"]
            weight = FIELD_WEIGHTS.get(f, 1)
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
            "type_document": consensus_fields.get("type_document"),
            "categorie": consensus_fields.get("categorie"),
            "consensus_score": global_score,
            "consensus_confidence": confidence,
        })

    df = pd.DataFrame(results)
    out = Path("output/compare_mineru_vs_pdfplumber.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[COMPARE] {out}  ({len(df)} PDFs)")
    print(f"  Score moyen: {df['consensus_score'].mean():.1f}%")
    print(f"  High: {len(df[df['consensus_confidence']=='high'])}  Medium: {len(df[df['consensus_confidence']=='medium'])}  Low: {len(df[df['consensus_confidence']=='low'])}")

    if diffs:
        df_diffs = pd.DataFrame(diffs)
        out_diff = Path("output/diffs_mineru_pdfplumber.csv")
        df_diffs.to_csv(out_diff, index=False, encoding="utf-8-sig")
        print(f"[DIFFS] {out_diff}  ({len(df_diffs)} diffs)")
        print("\nTop 10 diffs:")
        print(df_diffs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
