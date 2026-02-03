from __future__ import annotations

import argparse
from dotenv import load_dotenv

from finrag.config import load_config
from finrag.indexing.build_index import build_index
from finrag.indexing.embed import make_embeddings
from finrag.indexing.vectorstore import load_faiss
from finrag.retrieval.simple import retrieve
from finrag.generation.answer import answer_question
from finrag.eval.run_eval import run_eval


def cmd_index(config_path: str) -> None:
    cfg = load_config(config_path)
    build_index(
        raw_dir=cfg.paths.raw_dir,
        index_dir=cfg.paths.index_dir,
        manifest_path=cfg.paths.manifest_path,
        chunk_size=cfg.ingestion.chunk_size,
        chunk_overlap=cfg.ingestion.chunk_overlap,
        embedding_model=cfg.openai.embedding_model,
    )
    print("✅ Index built:", cfg.paths.index_dir)


def cmd_ask(config_path: str, question: str) -> None:
    cfg = load_config(config_path)
    embeddings = make_embeddings(cfg.openai.embedding_model)
    vs = load_faiss(cfg.paths.index_dir, embeddings)

    docs = retrieve(vs, question, k=cfg.retrieval.k)
    out = answer_question(question, docs, chat_model=cfg.openai.chat_model)

    print("\n=== ANSWER ===\n")
    print(out["answer"])
    print("\n=== SOURCES (debug) ===")
    for s in out["sources"]:
        print(s)


def cmd_eval(config_path: str, eval_file: str, k: int, out_path: str) -> None:
    cfg = load_config(config_path)
    report = run_eval(
        index_dir=cfg.paths.index_dir,
        embedding_model=cfg.openai.embedding_model,
        eval_path=eval_file,
        k=k,
        out_path=out_path,
    )
    print("✅ Eval done.")
    print("Summary:", report["summary"])
    print("Saved:", out_path)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="finrag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index")
    p_index.add_argument("--config", required=True)

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("--config", required=True)
    p_ask.add_argument("--q", required=True)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--eval-file", required=True)
    p_eval.add_argument("--k", type=int, default=6)
    p_eval.add_argument("--out", default="reports/eval_report.json")

    args = parser.parse_args()

    if args.cmd == "index":
        cmd_index(args.config)
    elif args.cmd == "ask":
        cmd_ask(args.config, args.q)
    elif args.cmd == "eval":
        cmd_eval(args.config, args.eval_file, args.k, args.out)


if __name__ == "__main__":
    main()
