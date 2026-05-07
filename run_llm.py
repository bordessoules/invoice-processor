"""Lance un run LLM : 1 config + 1 extractor."""
import sys, io, os, json, argparse, time
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from pathlib import Path
from openai import OpenAI

import config as cfg
from db import init_db, get_pdf_text, insert_extraction, insert_prompt
from utils.llm_utils import extract_json
from utils.pdf_utils import pdf_pages_to_base64_list, is_image, list_supported_files


def _build_extra(sampling: dict) -> dict:
    extra = {}
    for k in ("top_k", "min_p", "repetition_penalty"):
        if k in sampling:
            extra[k] = sampling[k]
    return extra if extra else None


def _sampling_kwargs(model_cfg: dict) -> dict:
    """Build OpenAI-SDK kwargs from sampling block — only pass keys explicitly set.

    If `sampling` is absent or empty, return {} so the provider applies its own defaults
    (useful for OpenRouter where each backend has its own preferred temperature etc.).
    """
    s = model_cfg.get("sampling") or {}
    out = {}
    for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        if k in s:
            out[k] = s[k]
    extra = _build_extra(s) or {}

    # OpenRouter unified `reasoning` parameter — passes through to provider.
    # Schema: {"enabled": bool, "effort": "low|medium|high", "max_tokens": int, "exclude": bool}
    # See: https://openrouter.ai/docs/api-reference/parameters
    if model_cfg.get("reasoning"):
        extra["reasoning"] = model_cfg["reasoning"]

    # Qwen open-source: chat_template_kwargs forwarded to vLLM-style backends
    if model_cfg.get("chat_template_kwargs"):
        extra["chat_template_kwargs"] = model_cfg["chat_template_kwargs"]

    if extra:
        out["extra_body"] = extra
    return out


def _make_client(model_cfg: dict) -> OpenAI:
    """Pick base_url + api_key from config, fall back to global env vars.

    api_key_env : name of env var holding the real key (e.g. OPENROUTER_API_KEY).
                  If unset/missing, fall back to cfg.OPENAI_API_KEY ('not-needed' for LM Studio).
    """
    base_url = model_cfg.get("base_url", cfg.OPENAI_BASE_URL)
    key_env = model_cfg.get("api_key_env")
    api_key = os.getenv(key_env) if key_env else None
    if not api_key:
        api_key = cfg.OPENAI_API_KEY
    timeout = float(model_cfg.get("timeout", 120.0))
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def _parse_completion(resp, finish_reason_hint: str = "") -> tuple[dict | None, str]:
    """Pull JSON out of a chat completion. Reports clean status for empty/truncated."""
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    finish = getattr(resp.choices[0], "finish_reason", finish_reason_hint) or ""
    if not content:
        # Thinking model burnt the whole budget on reasoning_content with nothing left
        if finish == "length":
            return None, "empty_content (max_tokens reached during reasoning)"
        return None, "empty_content"
    try:
        return extract_json(content), "ok"
    except Exception as exc:
        return None, str(exc)[:200]


def run_text(pdf_path: str, text: str, model_cfg: dict) -> tuple[dict | None, str]:
    client = _make_client(model_cfg)
    try:
        resp = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=[
                {"role": "system", "content": model_cfg["prompt_text"]},
                {"role": "user", "content": text[:model_cfg.get("text_max_chars", 6000)]},
            ],
            max_tokens=model_cfg.get("max_tokens", 2048),
            **_sampling_kwargs(model_cfg),
        )
        return _parse_completion(resp)
    except Exception as exc:
        return None, str(exc)[:200]


def run_vision(pdf_path: str, model_cfg: dict) -> tuple[dict | None, str]:
    client = _make_client(model_cfg)
    b64_images = pdf_pages_to_base64_list(pdf_path, dpi=cfg.PDF_DPI)
    content = [{"type": "text", "text": f"Extrais ce document ({len(b64_images)} page(s)) :"}]
    for b64 in b64_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    try:
        resp = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=[
                {"role": "system", "content": model_cfg["prompt_vision"]},
                {"role": "user", "content": content},
            ],
            max_tokens=model_cfg.get("max_tokens", 2048),
            **_sampling_kwargs(model_cfg),
        )
        return _parse_completion(resp)
    except Exception as exc:
        return None, str(exc)[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Chemin vers configs/<nom>.json")
    parser.add_argument("--extractor", required=True, choices=["pdfplumber", "pymupdf", "mineru", "vision"])
    parser.add_argument("--limit", type=int, default=None, help="Ne traiter que les N premiers PDFs (test)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip PDFs qui ont déjà une extraction OK (model+extractor) en DB")
    parser.add_argument("--image-only", action="store_true",
                        help="Ne traiter que les fichiers image (jpg/png/...). Pour mode vision uniquement.")
    parser.add_argument("--missing-only", action="store_true",
                        help="Ne traiter que les fichiers SANS AUCUNE extraction en DB (orphelins).")
    args = parser.parse_args()

    model_cfg = json.loads(Path(args.model).read_text(encoding="utf-8"))
    mode = "vision" if args.extractor == "vision" else "text"
    prompt = model_cfg["prompt_vision"] if mode == "vision" else model_cfg["prompt_text"]
    prompt_hash = insert_prompt(prompt)

    init_db()
    files = list_supported_files("factures")
    # Text-mode extractors (pdfplumber/pymupdf/mineru) skip images — they'll only
    # surface in the vision route below.
    if mode == "text":
        files = [f for f in files if not is_image(f)]
    if args.image_only:
        if mode != "vision":
            print("[ERR] --image-only n'a de sens qu'avec --extractor vision")
            return
        files = [f for f in files if is_image(f)]
    if args.missing_only:
        import sqlite3
        conn = sqlite3.connect("output/invoices.db")
        seen = {r[0] for r in conn.execute("SELECT DISTINCT pdf_path FROM llm_extractions").fetchall()}
        conn.close()
        before = len(files)
        files = [f for f in files if str(f.resolve()) not in seen]
        print(f"[RUN] Missing-only: {before - len(files)} avec extraction skipped, {len(files)} orphelins à traiter")
    if args.limit:
        files = files[:args.limit]

    if args.skip_existing:
        import sqlite3
        conn = sqlite3.connect("output/invoices.db")
        already = {r[0] for r in conn.execute(
            "SELECT pdf_path FROM llm_extractions WHERE llm_model=? AND pdf_extractor=? AND status='ok'",
            (model_cfg["model_id"], args.extractor),
        ).fetchall()}
        conn.close()
        before = len(files)
        files = [p for p in files if str(p.resolve()) not in already]
        print(f"[RUN] Skip-existing: {before - len(files)} déjà OK skipped, {len(files)} à traiter")

    print(f"[RUN] Model={model_cfg['name']}  Extractor={args.extractor}  Mode={mode}  Files={len(files)}")
    pdfs = files  # keep variable name below for minimal downstream changes

    for pdf in pdfs:
        p_abs = str(pdf.resolve())
        p_rel = str(pdf)
        if mode == "text":
            text = get_pdf_text(p_rel, args.extractor)
            if not text:
                print(f"  [SKIP] {pdf.name} : texte non trouve pour {args.extractor}")
                continue
            raw = text[:2000]
            data, status = run_text(p_abs, text, model_cfg)
        else:
            raw = f"vision:{len(pdf_pages_to_base64_list(p_abs, dpi=cfg.PDF_DPI))}pages"
            data, status = run_vision(p_abs, model_cfg)

        row = {
            "pdf_path": p_abs,
            "mode": mode,
            "pdf_extractor": args.extractor,
            "llm_name": model_cfg["name"],
            "llm_model": model_cfg["model_id"],
            "sampling_json": json.dumps(model_cfg.get("sampling", {})),
            "prompt_hash": prompt_hash,
            "raw_input": raw,
            "result_json": json.dumps(data, ensure_ascii=False, default=str) if data else None,
            "status": status,
        }
        insert_extraction(row)
        marker = "OK" if status == "ok" else f"ERR:{status[:40]}"
        print(f"  [{marker}] {pdf.name}")

    print("[RUN] Termine")


if __name__ == "__main__":
    main()
