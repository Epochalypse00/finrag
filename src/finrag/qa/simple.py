from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def retrieve(vs: FAISS, query: str, k: int = 4) -> List[Document]:
    """Basic similarity search retriever."""
    return vs.similarity_search(query, k=k)
