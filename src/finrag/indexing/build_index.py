from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from finrag.ingestion.load import load_pdfs
from finrag.ingestion.parse import parse_and_clean
from finrag.ingestion.chunk import chunk_documents
from finrag.indexing.embed import make_embeddings
from finrag.indexing.vectorstore import build_faiss_index, save_faiss


def build_index(
    raw_dir: str,
    index_dir: str,
    manifest_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> None:
    # 1) Load
    docs = load_pdfs(raw_dir)
    print(f"Loaded {len(docs)} pages total")
    counts = Counter(d.metadata.get("file_name") for d in docs)
    print("Pages per file:", dict(counts))

    # 2) Parse + clean (THIS is where Apple HTML becomes readable text)
    docs = parse_and_clean(docs)

    # 3) Sanity check AFTER parse
    apple = [d for d in docs if (d.metadata.get(
        "file_name") or "").lower() == "aapl-20230930.htm"]
    print("Apple pages loaded:", len(apple))
    if apple:
        print("Apple sample chars:", len(apple[0].page_content))
        print("Apple preview:", apple[0].page_content[:350].replace("\n", " "))

    # 4) Chunk
    chunks = chunk_documents(docs, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap)

    # 5) Embed + index
    embeddings = make_embeddings(embedding_model)
    vs = build_faiss_index(chunks, embeddings)
    save_faiss(vs, index_dir)

    # 6) Manifest
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "raw_dir": raw_dir,
        "index_dir": index_dir,
        "num_pages": len(docs),
        "num_chunks": len(chunks),
        "embedding_model": embedding_model,
    }
    Path(manifest_path).write_text(json.dumps(
        manifest, indent=2), encoding="utf-8")
