from __future__ import annotations

import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


ITEM_HEADER_RE = re.compile(
    r"(?im)^\s*item\s+(\d{1,2}[a-z]?)\s*[\.\-:]*\s*(.+)?\s*$"
)

# Common financial section aliases that show up in 10-K / annual reports
SECTION_NORMALIZE = {
    "1a": "Item 1A - Risk Factors",
    "7": "Item 7 - MD&A",
    "7a": "Item 7A - Quantitative and Qualitative Disclosures",
    "8": "Item 8 - Financial Statements",
}


def _detect_section(line: str) -> Optional[str]:
    m = ITEM_HEADER_RE.match(line)
    if not m:
        return None
    item = m.group(1).lower()
    title = (m.group(2) or "").strip()

    if item in SECTION_NORMALIZE:
        return SECTION_NORMALIZE[item]
    # fallback: keep whatever title is present
    if title:
        return f"Item {item.upper()} - {title}"
    return f"Item {item.upper()}"


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    out: List[Document] = []

    for d in docs:
        text = d.page_content or ""
        meta = dict(d.metadata or {})

        # Track section by scanning lines (especially useful for HTML filings)
        current_section: Optional[str] = meta.get("section")

        lines = text.splitlines()
        for line in lines:
            sec = _detect_section(line)
            if sec:
                current_section = sec

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            ch_meta = dict(meta)
            if current_section:
                ch_meta["section"] = current_section

            # stable chunk id
            fn = ch_meta.get("file_name", "unknown")
            page = ch_meta.get("page", "?")
            ch_meta["chunk_id"] = f"{fn}::p{page}::c{i}"

            out.append(Document(page_content=chunk, metadata=ch_meta))

    return out
