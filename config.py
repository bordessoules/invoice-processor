import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── LM Studio Link (ou LM Studio local) ──────────────────────────────
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")

# ── Modèles chargés sur ton infra ───────────────────────────────────
MODEL_QWEN  = os.getenv("MODEL_QWEN", "qwen/qwen3.6-35b-a3b")
MODEL_GEMMA = os.getenv("MODEL_GEMMA", "google/gemma-4-e4b")

# Parametres de sampling recommandes par l'utilisateur
SAMPLING_QWEN = {
    "temperature": 0.7,
    "top_p": 0.80,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
SAMPLING_GEMMA = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}

# Retro-compatibilite
MODEL_TEXT = os.getenv("MODEL_TEXT", MODEL_QWEN)
MODEL_VISION = os.getenv("MODEL_VISION", MODEL_QWEN)

# ── Paramètres de traitement ────────────────────────────────────────
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))   # PDFs traites en parallele (x4 routes = 8 requetes max)
PDF_DPI = int(os.getenv("PDF_DPI", "150"))                # résolution rendu image
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "150")) # seuil pour passer en vision

# ── Dossiers ────────────────────────────────────────────────────────
INPUT_DIR = os.getenv("INPUT_DIR", r".\factures")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", r".\output")
