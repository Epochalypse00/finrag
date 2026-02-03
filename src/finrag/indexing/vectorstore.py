from __future__ import annotations
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS


def build_faiss_index(chunks: List[Document], embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


def save_faiss(vs: FAISS, index_dir: str) -> None:
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(index_dir)


def load_faiss(index_dir: str, embeddings) -> FAISS:
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
