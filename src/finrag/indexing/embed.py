from __future__ import annotations

from langchain_openai import OpenAIEmbeddings


def make_embeddings(model: str) -> OpenAIEmbeddings:
    # chunk_size controls how many texts are sent per embeddings API request
    # Lower = safer for token limits
    return OpenAIEmbeddings(model=model, chunk_size=64)
