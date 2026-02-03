# FinRAG 📄🔎 — Financial Reports RAG Assistant (SEC 10-K + Annual Reports)

A production-style **Retrieval-Augmented Generation (RAG)** project that lets you ask questions over real financial filings (PDF + SEC iXBRL HTML) and get **answers with citations**.

This repo is built to demonstrate:  
- clean ingestion + chunking + indexing  
- reliable retrieval + citations  
- CLI workflows (index / ask / eval)  
- lightweight evaluation metrics for RAG quality

---

## Why this exists

Financial filings are long, messy, and inconsistent across formats:
- **PDFs** can be text-extractable or “garbage” depending on encoding
- **SEC iXBRL HTML** contains huge amounts of structured tags, namespaces, and hidden facts

This project normalizes all that into a searchable vector index, then answers questions using retrieved evidence + citation formatting.

---

## Features

✅ **Multi-format ingestion**
- PDF via PyMuPDF  
- SEC iXBRL HTML/HTM: extracts human text, removes tables/scripts/XBRL noise, and splits into **Item sections** (Item 1, Item 1A, etc.)

✅ **Vector search (FAISS)**
- Chunking with overlap  
- FAISS local index saved under `indices/`

✅ **Cited answers**
- Answers include citations like:
  - `[aapl-20230930.htm, Item 1A]`
  - `[NVIDIA-2025-Annual-Report.pdf, p.106]`

✅ **Evaluation pipeline**
- Loads `eval/cases.jsonl`
- Tracks:
  - `hit_rate@k`
  - `MRR`
  - `% answers with citations`
  - `% citations that match retrieved sources`
- Writes `reports/eval_report.json`

---

## Project layout

finrag/
├─ configs/
│ └─ index.yaml
├─ data/
│ └─ raw/ # put PDFs/HTML filings here (gitignored)
├─ eval/
│ └─ cases.jsonl # eval questions + expected sources
├─ indices/
│ └─ faiss/ # saved FAISS index (gitignored)
├─ reports/
│ └─ eval_report.json # eval output (optional to commit)
├─ src/finrag/
│ ├─ cli.py
│ ├─ config.py
│ ├─ ingestion/
│ ├─ indexing/
│ ├─ retrieval/
│ ├─ generation/
│ └─ eval/
└─ tests/


---

## Setup

### 1) Create a venv & install
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
2) Add your OpenAI key
Create .env:

OPENAI_API_KEY=your_key_here
Never commit .env. This repo includes .env.example.

Usage
Index documents
Put your filings in data/raw/, then run:

python -m finrag.cli index --config configs/index.yaml
You should see:

pages loaded per file

index saved under indices/faiss

Ask questions
python -m finrag.cli ask --config configs/index.yaml --q "Summarize the main risk factors and cite sources."
Run evaluation
python -m finrag.cli eval --config configs/index.yaml --eval-file eval/cases.jsonl --k 6 --out reports/eval_report.json
Example output:

hit_rate@k

MRR

citation quality metrics

Evaluation results (example)
Your run may vary by documents and chunking settings. Example from a real run:

num_cases: 6

hit_rate@k: 0.33

mrr: 0.069

answers_with_citations: 0.83

citations_match_retrieved: 0.66

If you want higher scores:

tune chunk_size / chunk_overlap

increase k

improve SEC HTML extraction (reduce noise, keep headings)

add better retrieval (MMR, BM25 hybrid)

Notes / Limitations 
SEC HTML is not “page-based” like PDFs; citations use section labels.

PDF extraction quality depends on how the PDF is embedded/encoded.

FAISS index is local and intended for demo + portfolio (not cloud scale).