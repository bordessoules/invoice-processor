"""Importe les resultats existants (invoices.jsonl) dans la base SQLite."""
import sys, io, json
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from pathlib import Path
from db import init_db, insert_extraction, insert_prompt


def main():
    init_db()
    jsonl = Path("output/invoices.jsonl")
    if not jsonl.exists():
        print("[ERR] invoices.jsonl introuvable")
        return

    # Prompts communs (on importe les hashs rapidement)
    qwen_cfg = json.loads(Path("configs/qwen.json").read_text(encoding="utf-8"))
    gemma_cfg = json.loads(Path("configs/gemma.json").read_text(encoding="utf-8"))
    pt = insert_prompt(qwen_cfg["prompt_text"])
    pv = insert_prompt(qwen_cfg["prompt_vision"])
    gt = insert_prompt(gemma_cfg["prompt_text"])
    gv = insert_prompt(gemma_cfg["prompt_vision"])

    prompt_map = {
        "qwen_text": pt, "qwen_vision": pv,
        "gemma_text": gt, "gemma_vision": gv,
    }
    model_map = {
        "qwen_text": ("qwen3.6-35b", "qwen/qwen3.6-35b-a3b"),
        "qwen_vision": ("qwen3.6-35b", "qwen/qwen3.6-35b-a3b"),
        "gemma_text": ("gemma4-e4b", "google/gemma-4-e4b"),
        "gemma_vision": ("gemma4-e4b", "google/gemma-4-e4b"),
    }
    extractor_map = {
        "qwen_text": "pdfplumber", "qwen_vision": "vision",
        "gemma_text": "pdfplumber", "gemma_vision": "vision",
    }
    sampling_map = {
        "qwen_text": json.dumps(qwen_cfg["sampling"]),
        "qwen_vision": json.dumps(qwen_cfg["sampling"]),
        "gemma_text": json.dumps(gemma_cfg["sampling"]),
        "gemma_vision": json.dumps(gemma_cfg["sampling"]),
    }

    count = 0
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            inv = json.loads(line)
            raw = inv.get("_raw_results", {})
            for route, result in raw.items():
                status = "ok" if result is not None else "err"
                name, model_id = model_map[route]
                row = {
                    "pdf_path": inv["url_fichier"],
                    "mode": "text" if "text" in route else "vision",
                    "pdf_extractor": extractor_map[route],
                    "llm_name": name,
                    "llm_model": model_id,
                    "sampling_json": sampling_map[route],
                    "prompt_hash": prompt_map[route],
                    "raw_input": inv.get("raw_text", "")[:2000] if "text" in route else None,
                    "result_json": json.dumps(result, ensure_ascii=False, default=str) if result else None,
                    "status": status,
                }
                insert_extraction(row)
                count += 1

    print(f"[IMPORT] {count} extractions importees")


if __name__ == "__main__":
    main()
