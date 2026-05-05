"""Analyse comparative des extracteurs texte stockes dans la base."""
import sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import re
from db import _get_conn


def score_text(text: str) -> dict:
    if not text:
        return {"len": 0, "printable": 0, "cid": 0, "ctrl": 0, "keywords": 0}
    keywords = len(re.findall(r"\b(montant|total|facture|tva|ht|ttc|date|numero|commande)\b", text, re.I))
    cid = len(re.findall(r"cid:\d+", text, re.I))
    ctrl = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    return {
        "len": len(text),
        "printable": round(printable / len(text) * 100, 1),
        "cid": cid,
        "ctrl": ctrl,
        "keywords": keywords,
    }


def main():
    conn = _get_conn()
    rows = conn.execute("SELECT pdf_path, extractor, content FROM pdf_texts").fetchall()
    conn.close()

    by_pdf = {}
    for r in rows:
        by_pdf.setdefault(r["pdf_path"], {})[r["extractor"]] = r["content"]

    print("=== COMPARATIF EXTRACTEURS (echantillon) ===\n")
    print(f"{'PDF':<50} {'Extracteur':<12} {'Len':>6} {'%Print':>7} {'CID':>4} {'Ctrl':>4} {'Keywords':>8}")
    print("-" * 100)

    count = 0
    for pdf, extractors in by_pdf.items():
        if count >= 20:
            break
        for ext, text in extractors.items():
            s = score_text(text)
            fname = pdf.split("\\")[-1][:48]
            print(f"{fname:<50} {ext:<12} {s['len']:>6} {s['printable']:>7} {s['cid']:>4} {s['ctrl']:>4} {s['keywords']:>8}")
        count += 1

    # Stats globales
    print("\n=== STATS GLOBALES ===")
    totals = {}
    for pdf, extractors in by_pdf.items():
        for ext, text in extractors.items():
            totals.setdefault(ext, {"count": 0, "len": 0, "cid": 0, "ctrl": 0, "keywords": 0})
            s = score_text(text)
            totals[ext]["count"] += 1
            totals[ext]["len"] += s["len"]
            totals[ext]["cid"] += s["cid"]
            totals[ext]["ctrl"] += s["ctrl"]
            totals[ext]["keywords"] += s["keywords"]

    for ext, st in totals.items():
        n = st["count"]
        print(f"{ext:<12}: {n} PDFs  len_moy={st['len']//n}  cid_total={st['cid']}  ctrl_total={st['ctrl']}  keywords_moy={st['keywords']//n}")


if __name__ == "__main__":
    main()
