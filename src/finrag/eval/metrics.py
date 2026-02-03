from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from langchain_core.documents import Document


# matches [aapl..., Item 1A] or [file, p.12]
_CIT_RE = re.compile(r"\[([^\[\]]+?)\]")


def doc_id(d: Document) -> str:
    meta = d.metadata or {}
    fn = meta.get("file_name", "unknown")
    section = meta.get("section")
    page = meta.get("page")
    if section:
        return f"{fn}::{section}"
    if page is not None:
        return f"{fn}::p{page}"
    return f"{fn}"


def extract_citations(answer: str) -> List[str]:
    """
    Extract bracket citations like:
      [aapl-20230930.htm, Item 1A]
      [NVIDIA-2025-Annual-Report.pdf, p.105]
    Returns raw strings inside brackets.
    """
    return [m.group(1).strip() for m in _CIT_RE.finditer(answer or "")]


def normalize_expected(
    file_name: str,
    section: Optional[str] = None,
    page: Optional[int] = None,
) -> str:
    if section:
        return f"{file_name}::{section}"
    if page is not None:
        return f"{file_name}::p{page}"
    return f"{file_name}"


def hit_at_k(retrieved: List[Document], expected_ids: Iterable[str], k: int) -> bool:
    expected = set(expected_ids)
    top = retrieved[:k]
    return any(doc_id(d) in expected for d in top)


def mrr(retrieved: List[Document], expected_ids: Iterable[str]) -> float:
    expected = set(expected_ids)
    for i, d in enumerate(retrieved, start=1):
        if doc_id(d) in expected:
            return 1.0 / i
    return 0.0


def citations_in_retrieved(answer: str, retrieved: List[Document]) -> Tuple[bool, List[str]]:
    """
    Checks if any extracted citation string roughly matches retrieved sources.
    We match by file name presence (best-effort, robust to formatting differences).
    """
    cits = extract_citations(answer)
    if not cits:
        return False, []

    retrieved_fns = {(d.metadata or {}).get("file_name", "")
                     for d in retrieved}
    ok = []
    for c in cits:
        # if citation contains a filename that is among retrieved filenames
        if any(fn and fn in c for fn in retrieved_fns):
            ok.append(c)
    return (len(ok) > 0), ok
