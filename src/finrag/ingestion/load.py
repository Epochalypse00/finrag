from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document


def _load_pdf(pdf_path: Path) -> List[Document]:
    docs: List[Document] = []
    pdf = fitz.open(pdf_path)
    try:
        for page_idx in range(pdf.page_count):
            page = pdf.load_page(page_idx)
            text = page.get_text("text") or ""
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "file_name": pdf_path.name,
                        "page": page_idx + 1,
                        "source": str(pdf_path),
                        "file_type": "pdf",
                    },
                )
            )
    finally:
        pdf.close()
    return docs


def _load_text_file(path: Path) -> List[Document]:
    # Keep raw content; HTML will be parsed into clean text in parse.py
    content = path.read_text(encoding="utf-8", errors="ignore")
    return [
        Document(
            page_content=content,
            metadata={
                "file_name": path.name,
                "page": 1,
                "source": str(path),
                "file_type": path.suffix.lower().lstrip("."),
            },
        )
    ]


def load_pdfs(raw_dir: str) -> List[Document]:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dir does not exist: {raw_dir}")

    docs: List[Document] = []

    # PDFs
    for pdf_path in sorted(raw_path.glob("*.pdf")):
        docs.extend(_load_pdf(pdf_path))

    # Text / HTML
    for path in sorted(
        list(raw_path.glob("*.txt"))
        + list(raw_path.glob("*.html"))
        + list(raw_path.glob("*.htm"))
    ):
        docs.extend(_load_text_file(path))

    if not docs:
        raise FileNotFoundError(
            f"No supported files found in: {raw_dir} (pdf/txt/html/htm)")

    return docs
