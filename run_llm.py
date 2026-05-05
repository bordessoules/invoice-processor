"""Lance un run LLM : 1 config + 1 extractor."""
import sys, io, json, argparse, time
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from pathlib import Path
from openai import OpenAI

import config as cfg
from db import init_db, get_pdf_text, insert_extraction, insert_prompt
from utils.llm_utils import extract_json
from utils.pdf_utils import pdf_pages_to_base64_list


def _build_extra(sampling: dict) -> dict:
    extra = {}
    for k in ("top_k", "min_p", "repetition_penalty"):
        if k in sampling:
            extra[k] = sampling[k]
    return extra if extra else None


def run_text(pdf_path: str, text: str, model_cfg: dict) -> tuple[dict | None, str]:
    client = OpenAI(base_url=cfg.OPENAI_BASE_URL, api_key=cfg.OPENAI_API_KEY, timeout=120.0)
    s = model_cfg["sampling"]
    try:
        resp = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=[
                {"role": "system", "content": model_cfg["prompt_text"]},
                {"role": "user", "content": text[:6000]},
            ],
            temperature=s.get("temperature", 0.7),
            top_p=s.get("top_p", 0.8),
            presence_penalty=s.get("presence_penalty", 0.0),
            frequency_penalty=s.get("frequency_penalty", 0.0),
            extra_body=_build_extra(s),
            max_tokens=2048,
        )
        data = extract_json(resp.choices[0].message.content)
        return data, "ok"
    except Exception as exc:
        return None, str(exc)[:200]


def run_vision(pdf_path: str, model_cfg: dict) -> tuple[dict | None, str]:
    client = OpenAI(base_url=cfg.OPENAI_BASE_URL, api_key=cfg.OPENAI_API_KEY, timeout=120.0)
    s = model_cfg["sampling"]
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
            temperature=s.get("temperature", 0.7),
            top_p=s.get("top_p", 0.8),
            presence_penalty=s.get("presence_penalty", 0.0),
            frequency_penalty=s.get("frequency_penalty", 0.0),
            extra_body=_build_extra(s),
            max_tokens=2048,
        )
        data = extract_json(resp.choices[0].message.content)
        return data, "ok"
    except Exception as exc:
        return None, str(exc)[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Chemin vers configs/<nom>.json")
    parser.add_argument("--extractor", required=True, choices=["pdfplumber", "pymupdf", "mineru", "vision"])
    args = parser.parse_args()

    model_cfg = json.loads(Path(args.model).read_text(encoding="utf-8"))
    mode = "vision" if args.extractor == "vision" else "text"
    prompt = model_cfg["prompt_vision"] if mode == "vision" else model_cfg["prompt_text"]
    prompt_hash = insert_prompt(prompt)

    init_db()
    pdfs = sorted(Path("factures").rglob("*.pdf"))
    print(f"[RUN] Model={model_cfg['name']}  Extractor={args.extractor}  Mode={mode}  PDFs={len(pdfs)}")

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
            "sampling_json": json.dumps(model_cfg["sampling"]),
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
