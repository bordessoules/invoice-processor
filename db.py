"""Base SQLite : pdf_texts + llm_extractions + prompts."""
import sqlite3
import hashlib
import json
from pathlib import Path

DB_PATH = Path("output/invoices.db")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS pdf_texts (
        pdf_path  TEXT,
        extractor TEXT,
        content   TEXT,
        pdf_hash  TEXT,
        created_at TEXT,
        PRIMARY KEY (pdf_path, extractor)
    );

    CREATE TABLE IF NOT EXISTS llm_extractions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pdf_path      TEXT,
        mode          TEXT,
        pdf_extractor TEXT,
        llm_name      TEXT,
        llm_model     TEXT,
        sampling_json TEXT,
        prompt_hash   TEXT,
        raw_input     TEXT,
        result_json   TEXT,
        status        TEXT,
        created_at    TEXT,
        UNIQUE (pdf_path, mode, pdf_extractor, llm_model)
    );

    CREATE TABLE IF NOT EXISTS prompts (
        hash TEXT PRIMARY KEY,
        content TEXT
    );

    CREATE TABLE IF NOT EXISTS manual_review (
        pdf_path        TEXT PRIMARY KEY,
        final_data_json TEXT,
        status          TEXT,
        reviewed_at     TEXT
    );
    """)
    conn.commit()
    conn.close()


def pdf_hash(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def insert_pdf_text(pdf_path: str, extractor: str, content: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO pdf_texts (pdf_path, extractor, content, pdf_hash, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
        (pdf_path, extractor, content, pdf_hash(pdf_path)),
    )
    conn.commit()
    conn.close()


def get_pdf_text(pdf_path: str, extractor: str) -> str | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT content FROM pdf_texts WHERE pdf_path=? AND extractor=?",
        (pdf_path, extractor),
    ).fetchone()
    conn.close()
    return row["content"] if row else None


def insert_extraction(row: dict) -> None:
    conn = _get_conn()
    conn.execute("""
    INSERT OR REPLACE INTO llm_extractions
    (pdf_path, mode, pdf_extractor, llm_name, llm_model, sampling_json, prompt_hash, raw_input, result_json, status, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        row["pdf_path"], row["mode"], row["pdf_extractor"], row["llm_name"],
        row["llm_model"], row["sampling_json"], row["prompt_hash"],
        row.get("raw_input"), row.get("result_json"), row["status"],
    ))
    conn.commit()
    conn.close()


def get_extractions_for_pdf(pdf_path: str) -> list[sqlite3.Row]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM llm_extractions WHERE pdf_path=? AND status='ok' ORDER BY id",
        (pdf_path,),
    ).fetchall()
    conn.close()
    return rows


def get_all_pdf_paths() -> list[str]:
    conn = _get_conn()
    rows = conn.execute("SELECT DISTINCT pdf_path FROM pdf_texts").fetchall()
    conn.close()
    return [r["pdf_path"] for r in rows]


def prompt_exists(p_hash: str) -> bool:
    conn = _get_conn()
    row = conn.execute("SELECT 1 FROM prompts WHERE hash=?", (p_hash,)).fetchone()
    conn.close()
    return row is not None


def insert_prompt(content: str) -> str:
    p_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    conn = _get_conn()
    conn.execute("INSERT OR IGNORE INTO prompts (hash, content) VALUES (?, ?)", (p_hash, content))
    conn.commit()
    conn.close()
    return p_hash
