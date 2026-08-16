"""RRF fusion + citation parsing — pure functions, no Milvus / torch needed."""

from policy_auditor.generation.answerer import extract_citations
from policy_auditor.retrieval.retriever import fuse_rrf


def _hit(source, page, line_start, text="x"):
    return {
        "source": source,
        "page": page,
        "line_start": line_start,
        "line_end": line_start + 5,
        "text": text,
        "section": "",
        "contains_table": False,
    }


def test_rrf_prefers_chunks_hit_by_multiple_queries():
    q1 = [_hit("a.pdf", 1, 10), _hit("a.pdf", 2, 20), _hit("b.pdf", 3, 5)]
    q2 = [_hit("a.pdf", 2, 20), _hit("b.pdf", 3, 5), _hit("b.pdf", 9, 1)]
    fused = fuse_rrf([q1, q2], pool_size=10)

    top_two_keys = {(c.source, c.page, c.line_start) for c in fused[:2]}
    assert top_two_keys == {("a.pdf", 2, 20), ("b.pdf", 3, 5)}
    by_key = {(c.source, c.page, c.line_start): c for c in fused}
    assert by_key[("a.pdf", 2, 20)].n_queries_hit == 2


def test_rrf_pool_size_is_respected():
    q1 = [_hit("a.pdf", 1, i) for i in range(20)]
    fused = fuse_rrf([q1], pool_size=5)
    assert len(fused) == 5


def test_extract_citations_line_and_range():
    answer = (
        "Leave is 24 days [Acme_Employee_Handbook_2026.pdf, Page 4, Lines 12-18]. "
        "Notice is 14 days [Handbook.pdf, Page 5, Line 3]."
    )
    cits = extract_citations(answer)
    assert cits[0] == {
        "source": "Acme_Employee_Handbook_2026.pdf",
        "page": 4,
        "line_start": 12,
        "line_end": 18,
    }
    assert cits[1]["line_start"] == 3 and cits[1]["line_end"] == 3
    assert len(cits) == 2
