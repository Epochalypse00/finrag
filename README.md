# FinRAG — Financial Reports RAG System (SEC 10-K + Annual Reports)

FinRAG is a production-style Retrieval-Augmented Generation (RAG) system for financial reports (PDF and SEC iXBRL HTML). It enables question answering over real filings with citation-grounded responses.

The system is designed not only for retrieval and generation, but also for building reliable, structured, and evaluation-driven AI pipelines.

---

## Why this exists

Financial filings are long, complex, and inconsistent across formats:

- PDFs vary in extraction quality depending on encoding
- SEC iXBRL HTML contains large amounts of structured tags, namespaces, and hidden data

This project normalizes these sources into a unified, searchable representation and generates answers grounded in retrieved evidence.

---

## Current Development Direction

FinRAG is being extended beyond a standard RAG pipeline into a more structured multi-stage system focused on improving reliability and interpretability.

Ongoing work includes:

- Attribute normalization to standardize heterogeneous financial report formats  
- Graph-based reasoning to capture relationships between entities  
- Perspective-driven query generation inspired by STORM-style approaches  
- Hybrid retrieval combining semantic embeddings and structured querying  
- Multi-level evaluation for retrieval quality, grounding, and reliability  

Target pipeline:

Raw Reports  
→ Attribute Normalization  
→ Graph Construction  
→ Perspective Queries  
→ Hybrid Retrieval  
→ Answer Generation with Citations  
→ Evaluation  

---

## Features

Multi-format ingestion  
- PDF via PyMuPDF  
- SEC iXBRL HTML/HTM: extracts human-readable text, removes tables/scripts/XBRL noise, and splits into Item sections (Item 1, Item 1A, etc.)

Vector search (FAISS)  
- Chunking with overlap  
- Local FAISS index stored under `indices/`

Cited answers  
- Answers include references such as:  
  - [aapl-20230930.htm, Item 1A]  
  - [NVIDIA-2025-Annual-Report.pdf, p.106]

Evaluation pipeline  
- Loads `eval/cases.jsonl`  
- Tracks:
  - hit_rate@k  
  - MRR  
  - percentage of answers with citations  
  - citation alignment with retrieved sources  
- Outputs results to `reports/eval_report.json`

Rapid iteration focus  
- Designed to support fast experimentation and improvement of retrieval and answer quality

---

## Project layout

finrag/  
├─ configs/  
│  └─ index.yaml  
├─ data/  
│  └─ raw/  
├─ eval/  
│  └─ cases.jsonl  
├─ indices/  
│  └─ faiss/  
├─ reports/  
│  └─ eval_report.json  
├─ src/finrag/  
│  ├─ cli.py  
│  ├─ config.py  
│  ├─ ingestion/  
│  ├─ indexing/  
│  ├─ retrieval/  
│  ├─ generation/  
│  └─ eval/  
└─ tests/  

---

## Setup

### 1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
