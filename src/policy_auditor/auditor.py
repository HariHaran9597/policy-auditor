"""PolicyAuditor — orchestrates the full pipeline:

question -> query expansion (Groq) -> hybrid BM25+dense retrieval per variant (Milvus)
         -> RRF fusion across variants -> cross-encoder reranking (BGE) -> top-3 chunks
         -> grounded generation (Groq) with [source, Page X, Lines Y-Z] citations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from policy_auditor.config import Settings, get_settings
from policy_auditor.generation.answerer import Answerer, extract_citations
from policy_auditor.retrieval.query_expander import QueryExpander
from policy_auditor.retrieval.reranker import get_reranker
from policy_auditor.retrieval.retriever import HybridRetriever, RetrievedChunk


@dataclass
class AuditResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    expanded_queries: list[str]
    citations: list[dict] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def cited_chunks(self) -> list[RetrievedChunk]:
        """Chunks whose (source, page, line-range) were actually cited in the answer."""
        used = []
        for cit in self.citations:
            for c in self.chunks:
                if (
                    c.source == cit["source"]
                    and c.page == cit["page"]
                    and c.line_start <= cit["line_end"]
                    and c.line_end >= cit["line_start"]
                ):
                    if c not in used:
                        used.append(c)
        return used


class PolicyAuditor:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.expander = QueryExpander()
        self.retriever = None
        self.reranker = None
        self.answerer = Answerer()
        # Load order matters on Windows + embedded milvus-lite: torch models must
        # initialize BEFORE the milvus-lite CGO library opens, or the process
        # can segfault during CrossEncoder load (verified empirically, 3/3 runs).
        self._get_reranker()
        self.retriever = HybridRetriever()

    def _get_reranker(self):
        if self.reranker is None:
            self.reranker = get_reranker()
        return self.reranker

    def audit(self, question: str, expand: bool = True) -> AuditResult:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        queries = self.expander.expand(question) if expand else [question]
        if not queries:
            queries = [question]
        timings["expansion_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        pool = self.retriever.retrieve(queries)
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        top = self._get_reranker().rerank(question, pool)
        timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        t0 = time.perf_counter()
        answer = self.answerer.answer(question, top)
        timings["generation_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        return AuditResult(
            question=question,
            answer=answer,
            chunks=top,
            expanded_queries=queries,
            citations=extract_citations(answer),
            timings_ms=timings,
        )
