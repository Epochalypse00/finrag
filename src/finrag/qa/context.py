from __future__ import annotations

from typing import Iterable
from langchain_core.documents import Document


def format_source(d: Document) -> str:
    meta = d.metadata or {}
    fn = meta.get("file_name", "unknown")
    section = meta.get("section")
    page = meta.get("page", "?")

    if section:
        return f"[{fn}, {section}]"
    return f"[{fn}, p.{page}]"


def build_context(docs: Iterable[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        src = format_source(d)
        # Keep each chunk clearly separated for the model
        lines.append(f"SOURCE {i} {src}\n{d.page_content}")
    return "\n\n".join(lines)
