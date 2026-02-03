from __future__ import annotations

from typing import Iterable
from langchain_core.documents import Document


def format_source(d: Document) -> str:
    """
    Returns a consistent citation string.

    Priority:
      1) section (if present)  -> [file, Item 1A]
      2) page (if present)     -> [file, p.105]
      3) fallback              -> [file]
    """
    meta = d.metadata or {}
    file_name = meta.get("file_name") or meta.get("source") or "unknown"
    section = meta.get("section")
    page = meta.get("page")

    if section:
        return f"[{file_name}, {section}]"
    if page is not None:
        return f"[{file_name}, p.{page}]"
    return f"[{file_name}]"


def build_context(docs: Iterable[Document]) -> str:
    """
    Builds a context block for prompting. Each chunk is preceded by SOURCE i [citation].
    """
    blocks: list[str] = []
    for i, d in enumerate(docs, start=1):
        src = format_source(d)
        text = (d.page_content or "").strip()
        blocks.append(f"SOURCE {i} {src}\n{text}")
    return "\n\n".join(blocks)
