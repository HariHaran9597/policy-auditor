"""Central configuration, loaded from environment / .env via pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # ---- LLM (Groq) ----
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_expansion_model: str = "openai/gpt-oss-20b"
    # separate non-thinking judge for RAGAS (thinking models leak <think> blocks
    # through langchain, which would corrupt metric parsing)
    groq_judge_model: str = "openai/gpt-oss-120b"

    # ---- Vector store (Milvus) ----
    # MILVUS_URI: http endpoint of a Milvus server (docker compose up -d).
    # MILVUS_LOCAL_DB: filename for embedded milvus-lite (no Docker needed).
    # pymilvus also parses MILVUS_URI from .env, so it must stay a valid http
    # URI — use MILVUS_LOCAL_DB for the embedded fallback instead.
    milvus_uri: str = "http://localhost:19530"
    milvus_local_db: str = ""
    milvus_collection: str = "policy_chunks"

    @property
    def milvus_endpoint(self) -> str:
        return self.milvus_local_db if self.milvus_local_db else self.milvus_uri

    # ---- Local models ----
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    reranker_model: str = "BAAI/bge-reranker-base"

    # ---- Retrieval tuning ----
    expansion_count: int = 3
    retrieval_top_k: int = 8
    rerank_pool_size: int = 12
    rerank_top_k: int = 3

    # ---- Chunking ----
    chunk_target_chars: int = 900
    chunk_max_chars: int = 1300
    chunk_overlap_lines: int = 2

    # ---- Generation ----
    answer_temperature: float = 0.1
    answer_max_tokens: int = 4096

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def policies_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "policies"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()
