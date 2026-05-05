"""Compare les resultats MinerU+Qwen vs pdfplumber+Qwen sur les PDF deja traites."""
import sys, io, json
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from db import _get_conn
from utils.consensus import _match


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

    fields = ["numero_facture", "date_facture", "nom_fournisseur", "montant_ht", "montant_ttc", "montant_tva", "taux_tva", "type_document", "categorie"]
    agree = {f: {"match": 0, "total": 0} for f in fields}
    diffs = []

    for pdf, extr in by_pdf.items():
        if "mineru" not in extr or "pdfplumber" not in extr:
            continue
        m = extr["mineru"]
        p = extr["pdfplumber"]
        for f in fields:
            agree[f]["total"] += 1
            if _match(m.get(f), p.get(f), f):
                agree[f]["match"] += 1
            else:
                diffs.append((pdf.split("\\")[-1], f, m.get(f), p.get(f)))

    print("=== ACCORD MinerU vs pdfplumber (Qwen texte) ===")
    for f, st in sorted(agree.items(), key=lambda x: x[1]["match"] / x[1]["total"] if x[1]["total"] else 0):
        pct = st["match"] / st["total"] * 100 if st["total"] else 0
        print(f"  {f:<20}: {st['match']:>2}/{st['total']} ({pct:>5.1f}%)")

    if diffs:
        print("\n=== DIFFS ===")
        for pdf, f, mv, pv in diffs[:20]:
            print(f"  {pdf:<50} {f:<20} MinerU={mv}  pdfplumber={pv}")


if __name__ == "__main__":
    main()
