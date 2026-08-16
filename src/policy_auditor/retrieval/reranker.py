"""Cross-encoder reranking with BAAI/bge-reranker-base.

The retriever's first stage (BM25 + dense + RRF) is fast and broad; the cross-
encoder then reads each (query, passage) pair jointly and scores true relevance,
which is what buys the precision this project is about.
"""

from __future__ import annotations

import math

from policy_auditor.config import get_settings


class Reranker:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder  # lazy: keeps tests torch-free

        self.model_name = model_name or get_settings().reranker_model
        self._model = CrossEncoder(self.model_name, max_length=512)

    def rerank(
        self,
        query: str,
        chunks: list,
        top_k: int | None = None,
    ) -> list:
        """Score chunks against the ORIGINAL user question; return the top k."""
        if not chunks:
            return []
        top_k = top_k or get_settings().rerank_top_k

        pairs = [(query, c.text) for c in chunks]
        logits = self._model.predict(pairs, show_progress_bar=False)
        for chunk, logit in zip(chunks, logits):
            chunk.rerank_score = round(1.0 / (1.0 + math.exp(-float(logit))), 4)

        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]


_default_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = Reranker()
    return _default_reranker
