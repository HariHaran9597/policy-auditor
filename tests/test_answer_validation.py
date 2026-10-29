from policy_auditor.generation.answerer import (
    INSUFFICIENT,
    UNVERIFIED,
    Answerer,
    citations_resolve,
    extract_citations,
)
from policy_auditor.retrieval.retriever import RetrievedChunk


def _chunk(source="handbook.pdf", page=2, line_start=4, line_end=9):
    return RetrievedChunk(
        text="The policy evidence.",
        source=source,
        page=page,
        line_start=line_start,
        line_end=line_end,
        section="Leave",
        contains_table=False,
        rrf_score=0.02,
    )


def test_citation_must_resolve_inside_a_retrieved_chunk():
    citations = extract_citations("Leave is 24 days [handbook.pdf, Page 2, Lines 5-6].")

    assert citations_resolve(citations, [_chunk()])


def test_citation_with_wrong_source_page_or_lines_is_rejected():
    wrong_source = extract_citations("Leave is 24 days [other.pdf, Page 2, Lines 5-6].")
    wrong_page = extract_citations("Leave is 24 days [handbook.pdf, Page 3, Lines 5-6].")
    outside_chunk = extract_citations("Leave is 24 days [handbook.pdf, Page 2, Lines 3-10].")

    assert not citations_resolve(wrong_source, [_chunk()])
    assert not citations_resolve(wrong_page, [_chunk()])
    assert not citations_resolve(outside_chunk, [_chunk()])


def test_answerer_refuses_when_retrieval_returns_no_evidence():
    answer = Answerer(api_key="unused").answer("What is the policy?", [])

    assert answer.startswith(INSUFFICIENT)


def test_answerer_does_not_need_an_api_key_to_return_no_evidence_refusal():
    answer = Answerer(api_key="").answer("What is the policy?", [])

    assert answer.startswith(INSUFFICIENT)


def test_answerer_rejects_a_citation_not_in_retrieved_evidence(monkeypatch):
    answerer = Answerer(api_key="unused")
    monkeypatch.setattr(
        answerer,
        "_generate_with_retry",
        lambda question, contexts: "Leave is 24 days [wrong.pdf, Page 2, Lines 5-6].",
    )

    assert answerer.answer("How much leave?", [_chunk()]) == UNVERIFIED


def test_answerer_returns_an_answer_with_resolved_citations(monkeypatch):
    answerer = Answerer(api_key="unused")
    valid_answer = "Leave is 24 days [handbook.pdf, Page 2, Lines 5-6]."
    monkeypatch.setattr(answerer, "_generate_with_retry", lambda question, contexts: valid_answer)

    assert answerer.answer("How much leave?", [_chunk()]) == valid_answer
