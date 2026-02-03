from __future__ import annotations

from typing import List, Optional

from langchain.schema import Document


def _infer_file_filter(query: str) -> Optional[str]:
    q = query.lower()

    # Apple triggers
    if "apple" in q or "aapl" in q:
        return "aapl-20230930"

    # NVIDIA triggers
    if "nvidia" in q:
        return "nvidia"

    return None


def retrieve(vs, query: str, k: int) -> List[Document]:
    """
    Retrieve top-k docs from vectorstore.
    If query explicitly targets a company, filter retrieved docs to that company's filing.
    """
    file_filter = _infer_file_filter(query)

    # Pull extra candidates, then filter down.
    candidates = vs.similarity_search(query, k=max(20, k * 4))

    if file_filter:
        candidates = [
            d for d in candidates
            if file_filter in (d.metadata.get("file_name", "").lower())
        ]

    return candidates[:k]
