# Invoice Processor

Pipeline d'extraction de données depuis des PDF de factures via LLM local (LM Studio).

## Architecture

- **Extracteurs texte** : `pdfplumber`, `pymupdf`, `mineru` (depuis fichiers `.md`)
- **Extracteur vision** : rendu PDF → images base64 → LLM vision
- **LLMs** : Qwen 3.6-35b, Gemma 4-e4b (via LM Studio sur `localhost:1234`)
- **Stockage** : SQLite (`output/invoices.db`) avec traçabilité complète des runs
- **Consensus** : vote majoritaire pondéré par champ avec tolérance sur les montants

## Utilisation

```bash
# 1. Extraire les textes (une fois)
python seed_texts.py

# 2. Lancer un run LLM
python run_llm.py --model configs/qwen.json --extractor pdfplumber
python run_llm.py --model configs/qwen.json --extractor mineru
python run_llm.py --model configs/qwen.json --extractor vision

# 3. Comparer et générer le consensus final
python compare.py
```

## Configuration

Les variables d'environnement suivantes sont supportées (voir `config.py`) :
- `OPENAI_BASE_URL` (défaut: `http://localhost:1234/v1`)
- `MODEL_QWEN`, `MODEL_GEMMA`
- `MAX_CONCURRENT`, `PDF_DPI`

## Structure

```
├── configs/           # Configurations par modèle (prompt + sampling)
├── factures/          # PDFs source (organisés par dossier vendeur)
├── utils/             # pdf_utils.py, llm_utils.py, consensus.py
├── output/            # CSV + SQLite
├── run_llm.py         # Lanceur de runs
├── compare.py         # Moteur de consensus
└── seed_texts.py      # Extraction texte one-shot
```
