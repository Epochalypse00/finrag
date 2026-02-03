from __future__ import annotations
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from finrag.generation.prompt import SYSTEM_PROMPT
from finrag.generation.citations import build_context


def answer_question(
    question: str,
    retrieved_docs: List[Document],
    chat_model: str,
) -> Dict[str, Any]:
    llm = ChatOpenAI(model=chat_model, temperature=0)
    context = build_context(retrieved_docs)

    user_msg = f"""QUESTION:
{question}

CONTEXT:
{context}

Return answer with citations per rule #2."""
    resp = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
    )

    return {
        "answer": resp.content,
        "sources": [
            {
                "file_name": d.metadata.get("file_name"),
                "page": d.metadata.get("page"),
                "chunk_id": d.metadata.get("chunk_id"),
            }
            for d in retrieved_docs
        ],
    }
