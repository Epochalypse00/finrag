from __future__ import annotations
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from finrag.config import load_config
from finrag.indexing.embed import make_embeddings
from finrag.indexing.vectorstore import load_faiss
from finrag.retrieval.simple import retrieve
from finrag.generation.answer import answer_question

load_dotenv()
app = FastAPI(title="FinRAG")

CFG = load_config("configs/serve.yaml")
EMB = make_embeddings(CFG.openai.embedding_model)
VS = load_faiss(CFG.paths.index_dir, EMB)


class AskReq(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskReq):
    docs = retrieve(VS, req.question, k=CFG.retrieval.k)
    return answer_question(req.question, docs, chat_model=CFG.openai.chat_model)
