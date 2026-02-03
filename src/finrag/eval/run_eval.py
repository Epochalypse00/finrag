from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from finrag.indexing.embed import make_embeddings
from finrag.indexing.vectorstore import load_faiss
from finrag.retrieval.simple import retrieve
from finrag.generation.answer import answer_question

from .metrics import citations_in_retrieved, hit_at_k, mrr, normalize_expected


@dataclass
class EvalCase:
    q: str
    expected: List[str]  # list of normalized expected ids


def load_cases(path: str) -> List[EvalCase]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    cases: List[EvalCase] = []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        q = obj["q"]
        exp: List[str] = []
        for e in obj.get("expected", []):
            exp.append(
                normalize_expected(
                    file_name=e["file_name"],
                    section=e.get("section"),
                    page=e.get("page"),
                )
            )
        cases.append(EvalCase(q=q, expected=exp))
    return cases


def run_eval(
    index_dir: str,
    embedding_model: str,
    eval_path: str,
    k: int = 6,
    out_path: Optional[str] = None,
    chat_model: str = "gpt-4o-mini",
) -> Dict:
    embeddings = make_embeddings(embedding_model)
    vs = load_faiss(index_dir, embeddings)

    cases = load_cases(eval_path)

    total = len(cases)
    hit = 0
    mrr_sum = 0.0
    has_cit = 0
    cit_ok = 0

    rows = []

    for c in cases:
        retrieved: List[Document] = retrieve(vs, c.q, k=k)
        out = answer_question(c.q, retrieved, chat_model=chat_model)
        ans_text = out["answer"]

        h = hit_at_k(retrieved, c.expected, k=k)
        r = mrr(retrieved, c.expected)

        if h:
            hit += 1
        mrr_sum += r

        citations_present = ("[" in ans_text and "]" in ans_text)
        if citations_present:
            has_cit += 1

        ok, _ = citations_in_retrieved(ans_text, retrieved)
        if ok:
            cit_ok += 1

        rows.append(
            {
                "q": c.q,
                "hit@k": h,
                "mrr": r,
                "citations_present": citations_present,
                "citations_match_retrieved": ok,
                "expected": c.expected,
                "retrieved_top": [
                    {
                        "file_name": (d.metadata or {}).get("file_name"),
                        "section": (d.metadata or {}).get("section"),
                        "page": (d.metadata or {}).get("page"),
                    }
                    for d in retrieved[: min(k, len(retrieved))]
                ],
                "answer": ans_text,
            }
        )

    summary = {
        "num_cases": total,
        "k": k,
        "hit_rate@k": hit / total if total else 0.0,
        "mrr": mrr_sum / total if total else 0.0,
        "answers_with_citations": has_cit / total if total else 0.0,
        "citations_match_retrieved": cit_ok / total if total else 0.0,
    }

    report = {"summary": summary, "cases": rows}

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(
            report, indent=2), encoding="utf-8")

    return report
