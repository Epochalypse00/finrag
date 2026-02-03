from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from finrag.qa.context import build_context
from finrag.qa.simple import retrieve


SYSTEM_PROMPT = """You are a financial filings assistant.
Answer only using the provided SOURCES.
If the answer is not in SOURCES, say: "Not found in provided documents."
When you cite, copy the bracketed citation EXACTLY as shown in the SOURCE line.
Do not invent page numbers or sections.
"""

USER_PROMPT = """Question:
{question}

SOURCES:
{context}

Return:
1) A concise answer
2) Bullet points if multiple items
3) Citations inline like: [file, section] or [file, p.#] exactly from sources
"""


def answer_question(
    vs: Any,
    question: str,
    k: int = 6,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> Tuple[str, List[Dict[str, Any]]]:
    docs: List[Document] = retrieve(vs, question, k=k)
    context = build_context(docs)

    llm = ChatOpenAI(model=model, temperature=temperature)
    msg = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                question=question, context=context)},
        ]
    )

    debug_sources = [
        {
            "file_name": d.metadata.get("file_name"),
            "page": d.metadata.get("page"),
            "section": d.metadata.get("section"),
            "chunk_id": d.metadata.get("chunk_id"),
        }
        for d in docs
    ]

    return str(msg.content), debug_sources
