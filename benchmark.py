"""Benchmark : compare plusieurs configs LLM sur un sample de factures.

Vérité terrain = les `manual_review` validated par toi dans le viewer.

Usage :
  # Tout par défaut : auto-sélection des 4 PDFs les plus tordus, tous les configs openrouter-*.json
  python benchmark.py

  # Custom :
  python benchmark.py \
    --pdfs factures/2025/bouygues-telecom.fr/Bouyguestelecom_Facture_20250716.pdf \
           factures/2025/paybyphone.com/PayByPhoneParkingReceipt-13.pdf \
    --configs configs/openrouter-claude-haiku-4.5-think-low.json \
              configs/openrouter-gemini-3.1-pro-think-low.json \
    --extractor mineru \
    --out output/benchmark.csv
"""
import sys, io, os, json, time, argparse, sqlite3, csv
from pathlib import Path
from collections import defaultdict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config as cfg
from db import get_pdf_text
from run_llm import _make_client, _sampling_kwargs
from utils.consensus import _match, FIELD_WEIGHTS
from utils.llm_utils import extract_json
from utils.pdf_utils import is_image, pdf_pages_to_base64_list

DB_PATH = "output/invoices.db"

# Champs qu'on score (ceux de FIELD_WEIGHTS, hors nb_lignes qui est dérivé).
SCORED_FIELDS = [f for f in FIELD_WEIGHTS if f != "nb_lignes"]


def load_truth(pdf_abs_path: str) -> dict | None:
    """Récupère la valeur validée dans manual_review."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT final_data_json FROM manual_review WHERE pdf_path=? AND status='validated'",
        (pdf_abs_path,),
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def auto_pick_pdfs(n: int = 4) -> list[str]:
    """Auto-pick les N PDFs validés les plus 'tordus' (max désaccord historique vs vérité)."""
    conn = sqlite3.connect(DB_PATH)
    # PDFs validés qu'on a en vérité terrain
    rows = conn.execute("""
        SELECT pdf_path, final_data_json FROM manual_review WHERE status='validated'
    """).fetchall()
    candidates = []
    for path, raw in rows:
        truth = json.loads(raw) if raw else {}
        # Compte le nombre d'extractions LLM passées qui divergent de la vérité
        exs = conn.execute("""
            SELECT result_json FROM llm_extractions
            WHERE pdf_path=? AND status='ok'
        """, (path,)).fetchall()
        if not exs:
            continue
        n_disagree = 0
        n_total = 0
        for (rj,) in exs:
            try:
                d = json.loads(rj) if rj else {}
                if isinstance(d, list):
                    d = d[0] if d and isinstance(d[0], dict) else {}
            except Exception:
                continue
            for f in ("numero_facture", "montant_ht", "montant_ttc", "nom_fournisseur", "date_facture"):
                if f not in truth:
                    continue
                n_total += 1
                if not _match(d.get(f), truth.get(f), f):
                    n_disagree += 1
        if n_total == 0:
            continue
        score = n_disagree / n_total
        candidates.append((score, path))
    conn.close()
    # Sort by disagreement rate desc + diversify by vendor folder
    candidates.sort(reverse=True)
    picked: list[str] = []
    seen_vendors: set[str] = set()
    for score, p in candidates:
        vendor = Path(p).parent.name
        if vendor in seen_vendors:
            continue
        picked.append(p)
        seen_vendors.add(vendor)
        if len(picked) >= n:
            break
    return picked


def fetch_openrouter_metadata(generation_id: str, api_key: str) -> dict:
    """Fetch detailed generation metadata from OpenRouter (provider, applied params, etc.).

    Returns {} on failure. The endpoint may return 404 briefly after generation,
    so we do 2 quick retries with 0.5s backoff.
    """
    if not generation_id:
        return {}
    import urllib.request, urllib.error
    import time as _time
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt < 2:
                _time.sleep(0.5 * (attempt + 1))
            continue
    return {}


def call_llm(model_cfg: dict, pdf_path: str, extractor: str) -> dict:
    """Run a single LLM call. Returns dict with data, status, timings, tokens, provider."""
    client = _make_client(model_cfg)

    if extractor == "vision":
        b64_images = pdf_pages_to_base64_list(pdf_path, dpi=cfg.PDF_DPI)
        content = [{"type": "text", "text": f"Extrais ce document ({len(b64_images)} page(s)) :"}]
        for b64 in b64_images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        messages = [
            {"role": "system", "content": model_cfg["prompt_vision"]},
            {"role": "user", "content": content},
        ]
    else:
        # text mode
        rel = str(Path(pdf_path).relative_to(Path.cwd()) if Path(pdf_path).is_absolute() else pdf_path)
        text = get_pdf_text(rel, extractor)
        if not text:
            # try absolute lookup
            text = get_pdf_text(pdf_path, extractor)
        if not text:
            return {"data": None, "status": "no_text_in_db", "time_s": 0.0,
                    "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        truncated = text[:model_cfg.get("text_max_chars", 6000)]
        messages = [
            {"role": "system", "content": model_cfg["prompt_text"]},
            {"role": "user", "content": truncated},
        ]

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=messages,
            max_tokens=model_cfg.get("max_tokens", 8192),
            **_sampling_kwargs(model_cfg),
        )
        dt = time.time() - t0
        msg = resp.choices[0].message
        content_str = (msg.content or "").strip()
        # Token usage
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        completion_tokens = resp.usage.completion_tokens if resp.usage else 0
        reasoning_tokens = (
            getattr(resp.usage.completion_tokens_details, "reasoning_tokens", 0)
            if resp.usage and resp.usage.completion_tokens_details else 0
        )
        # OpenRouter generation id + provider lookup
        gen_id = getattr(resp, "id", None) or ""
        provider = ""
        applied_params = {}
        if model_cfg.get("base_url", "").startswith("https://openrouter.ai"):
            api_key = os.getenv(model_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
            if api_key and gen_id:
                meta = fetch_openrouter_metadata(gen_id, api_key)
                inner = meta.get("data", meta) if isinstance(meta, dict) else {}
                provider = inner.get("provider_name", "") or inner.get("provider", "")
                # OpenRouter returns the applied parameters under various keys
                for k in ("params", "applied_params", "settings"):
                    if isinstance(inner.get(k), dict):
                        applied_params = inner[k]
                        break

        result_base = {
            "time_s": dt,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "provider": provider,
            "generation_id": gen_id,
            "applied_params": json.dumps(applied_params, ensure_ascii=False) if applied_params else "",
        }

        if not content_str:
            return {"data": None, "status": "empty_content", **result_base}
        try:
            data = extract_json(content_str)
            if isinstance(data, list):
                data = data[0] if data and isinstance(data[0], dict) else {}
        except Exception as e:
            return {"data": None, "status": f"parse_error: {str(e)[:80]}", **result_base}
        return {"data": data, "status": "ok", **result_base}
    except Exception as exc:
        return {"data": None, "status": f"api_error: {str(exc)[:80]}", "time_s": time.time() - t0,
                "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                "provider": "", "generation_id": "", "applied_params": ""}


def score_against_truth(pred: dict, truth: dict) -> dict:
    """Returns per-field correctness + global weighted score."""
    if not pred:
        return {"per_field": {}, "global": 0.0, "n_correct": 0, "n_total": 0}
    per_field = {}
    weighted = 0.0
    total_w = 0
    n_correct = 0
    n_total = 0
    for f in SCORED_FIELDS:
        if f not in truth:
            continue
        truth_v = truth.get(f)
        pred_v = pred.get(f)
        ok = _match(pred_v, truth_v, f)
        per_field[f] = ok
        w = FIELD_WEIGHTS.get(f, 1)
        weighted += (1 if ok else 0) * w
        total_w += w
        n_total += 1
        if ok:
            n_correct += 1
    glob = (weighted / total_w * 100) if total_w else 0.0
    return {"per_field": per_field, "global": round(glob, 1),
            "n_correct": n_correct, "n_total": n_total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdfs", nargs="*", default=None,
                   help="Paths PDF/image (relatif à invoice-processor/). Si absent: auto-pick 4 plus tordus.")
    p.add_argument("--configs", nargs="*", default=None,
                   help="Paths configs JSON. Si absent: tous configs/openrouter-*.json.")
    p.add_argument("--extractor", default="mineru",
                   choices=["mineru", "pdfplumber", "pymupdf", "vision"])
    p.add_argument("--n", type=int, default=4, help="Si auto-pick: nombre de PDFs.")
    p.add_argument("--out", default="output/benchmark.csv")
    args = p.parse_args()

    # Resolve PDFs
    if args.pdfs:
        pdfs = [str(Path(x).resolve()) for x in args.pdfs]
    else:
        print(f"[BENCH] Auto-pick: {args.n} PDFs les plus tordus (max désaccord historique vs vérité)")
        pdfs = auto_pick_pdfs(args.n)
    if not pdfs:
        print("[ERR] Aucun PDF avec validated truth.")
        return

    # Resolve configs
    if args.configs:
        cfg_paths = args.configs
    else:
        cfg_paths = sorted(str(p) for p in Path("configs").glob("openrouter-*.json"))
    configs = [(p, json.loads(Path(p).read_text(encoding="utf-8"))) for p in cfg_paths]

    print(f"[BENCH] {len(configs)} configs × {len(pdfs)} PDFs × extractor={args.extractor} = {len(configs)*len(pdfs)} calls")
    for pdf in pdfs:
        print(f"  PDF: {Path(pdf).name}")

    # Run
    results = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for cfg_path, cfg_dict in configs:
        cfg_label = cfg_dict.get("name") or Path(cfg_path).stem
        for pdf in pdfs:
            truth = load_truth(pdf)
            if not truth:
                print(f"  [SKIP] {Path(pdf).name} : pas de vérité validée")
                continue
            t0 = time.time()
            print(f"  [{cfg_label}] {Path(pdf).name} ...", end=" ", flush=True)
            r = call_llm(cfg_dict, pdf, args.extractor)
            scored = score_against_truth(r["data"], truth)
            elapsed = time.time() - t0
            row = {
                "config": cfg_label,
                "model_id": cfg_dict["model_id"],
                "pdf": Path(pdf).name,
                "vendor": Path(pdf).parent.name,
                "extractor": args.extractor,
                "status": r["status"],
                "time_s": round(r["time_s"], 1),
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "reasoning_tokens": r["reasoning_tokens"],
                "provider": r.get("provider", ""),
                "generation_id": r.get("generation_id", ""),
                "applied_params": r.get("applied_params", ""),
                "score_pct": scored["global"],
                "n_correct": scored["n_correct"],
                "n_total": scored["n_total"],
            }
            for f in SCORED_FIELDS:
                row[f"f_{f}"] = ("OK" if scored["per_field"].get(f) is True
                                else "KO" if scored["per_field"].get(f) is False
                                else "")
            results.append(row)
            tag = "OK" if r["status"] == "ok" else r["status"][:30]
            rt = f"reason={r['reasoning_tokens']}t " if r["reasoning_tokens"] else ""
            prov = f" via {r['provider']}" if r.get("provider") else ""
            print(f"{tag}  {scored['global']:5.1f}%  {rt}({r['time_s']:.1f}s){prov}")

    # Per-config summary
    print("\n=== Résumé par config ===")
    by_cfg = defaultdict(list)
    for r in results:
        if r["status"] == "ok":
            by_cfg[r["config"]].append(r)
    print(f"{'config':<32} {'avg score':>10} {'avg time':>10} {'avg reason':>12} {'n_ok':>5}")
    print("-" * 78)
    summary = []
    for cfg_label in sorted(by_cfg, key=lambda c: -sum(r["score_pct"] for r in by_cfg[c]) / max(len(by_cfg[c]), 1)):
        rows = by_cfg[cfg_label]
        avg_score = sum(r["score_pct"] for r in rows) / len(rows)
        avg_time = sum(r["time_s"] for r in rows) / len(rows)
        avg_reason = sum(r["reasoning_tokens"] for r in rows) / len(rows)
        print(f"{cfg_label:<32} {avg_score:>9.1f}% {avg_time:>8.1f}s {avg_reason:>11.0f}t {len(rows):>5}")
        summary.append({"config": cfg_label, "avg_score": round(avg_score, 1),
                        "avg_time_s": round(avg_time, 1), "avg_reason_tokens": int(avg_reason),
                        "n_ok": len(rows)})

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"\n[OK] Détails écrits dans {args.out}")

    # Write summary CSV
    if summary:
        sum_path = Path(args.out).with_name(Path(args.out).stem + "_summary.csv")
        with open(sum_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            for s in summary:
                w.writerow(s)
        print(f"[OK] Résumé écrit dans {sum_path}")


if __name__ == "__main__":
    main()
