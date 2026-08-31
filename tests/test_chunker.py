from policy_auditor.ingestion.chunker import chunk_parsed_doc, fixed_chunk_parsed_doc
from policy_auditor.ingestion.pdf_parser import parse_pdf


def test_chunks_respect_limits_and_pages(sample_pdf):
    parsed = parse_pdf(sample_pdf)
    chunks = chunk_parsed_doc(parsed)
    assert len(chunks) >= 3
    for c in chunks:
        assert c.text.strip()
        assert c.line_start <= c.line_end
        assert c.page >= 1
        assert len(c.text) <= 2000  # hard ceiling
        assert c.bboxes and all(pg == c.page for pg, _ in c.bboxes)


def test_tables_stay_atomic(sample_pdf):
    parsed = parse_pdf(sample_pdf)
    chunks = chunk_parsed_doc(parsed)
    table_chunks = [c for c in chunks if c.contains_table]
    assert table_chunks, "the leave table must land inside a chunk"
    for c in table_chunks:
        table_lines = [l for l in c.text.split("\n") if "|" in l]
        assert len(table_lines) >= 3, "all table rows must stay in one chunk"


def test_citation_format(sample_pdf):
    parsed = parse_pdf(sample_pdf)
    chunks = chunk_parsed_doc(parsed)
    c = chunks[0]
    assert f"Page {c.page}" in c.citation and c.source in c.citation
    assert ("Lines" if c.line_start != c.line_end else "Line") in c.citation


def test_fixed_chunks_are_page_local_and_support_overlap(sample_pdf):
    parsed = parse_pdf(sample_pdf)
    chunks = fixed_chunk_parsed_doc(parsed, chunk_chars=120, overlap_lines=1)
    assert len(chunks) > 5
    for chunk in chunks:
        assert chunk.text.strip()
        assert all(page == chunk.page for page, _ in chunk.bboxes)
    assert any(
        first.page == second.page and first.line_end >= second.line_start
        for first, second in zip(chunks, chunks[1:])
    )
