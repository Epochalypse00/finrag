from __future__ import annotations
from pydantic import BaseModel
import yaml


class Paths(BaseModel):
    raw_dir: str = "data/raw"
    index_dir: str = "indices/faiss"
    manifest_path: str = "indices/faiss/manifest.json"


class IngestionCfg(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 150


class OpenAICfg(BaseModel):
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"


class RetrievalCfg(BaseModel):
    k: int = 6


class AppConfig(BaseModel):
    paths: Paths = Paths()
    ingestion: IngestionCfg = IngestionCfg()
    openai: OpenAICfg = OpenAICfg()
    retrieval: RetrievalCfg = RetrievalCfg()


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig(**data)
