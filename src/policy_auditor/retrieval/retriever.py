"""Hybrid retrieval with cross-query RRF fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from policy_auditor.config import get_settings
from policy_auditor.indexing.embedder import get_embedder
from policy_auditor.indexing.milvus_store import MilvusStore

RRF_K = 60


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    line_start: int
    line_end: int
    section: str
    contains_table: bool
    rrf_score: float
    n_queries_hit: int = 1
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        if self.line_start == self.line_end:
            return f"{self.source}, Page {self.page}, Line {self.line_start}"
        return f"{self.source}, Page {self.page}, Lines {self.line_start}-{self.line_end}"

    @property
    def chunk_key(self) -> tuple[str, int, int]:
        return (self.source, self.page, self.line_start)


def fuse_rrf(per_query_hits: list[list[dict[str, Any]]], pool_size: int) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion across expanded queries (pure function — unit tested)."""
    scores: dict[tuple[str, int, int], float] = {}
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    queries_hit: dict[tuple[str, int, int], int] = {}

    for hits in per_query_hits:
        for rank, hit in enumerate(hits):
            key = (hit["source"], hit["page"], hit["line_start"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_key.setdefault(key, hit)
            queries_hit[key] = queries_hit.get(key, 0) + 1

    fused = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:pool_size]:
        hit = by_key[key]
        fused.append(
            RetrievedChunk(
                text=hit["text"],
                source=hit["source"],
                page=int(hit["page"]),
                line_start=int(hit["line_start"]),
                line_end=int(hit["line_end"]),
                section=hit.get("section") or "",
                contains_table=bool(hit.get("contains_table", False)),
                rrf_score=round(score, 6),
                n_queries_hit=queries_hit[key],
            )
        )
    return fused


class HybridRetriever:
    def __init__(self, store: MilvusStore | None = None):
        self.store = store or MilvusStore()
        self.embedder = get_embedder()
        self.settings = get_settings()

    def retrieve(self, queries: list[str], top_k: int | None = None, pool_size: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieval_top_k
        pool_size = pool_size or self.settings.rerank_pool_size

        per_query_hits = []
        for q in queries:
            dense = self.embedder.encode_query(q)
            per_query_hits.append(self.store.hybrid_search(q, dense, limit=top_k))
        return fuse_rrf(per_query_hits, pool_size)
