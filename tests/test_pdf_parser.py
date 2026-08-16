from policy_auditor.ingestion.pdf_parser import body_font_size, parse_pdf


def test_parses_pages_and_lines(sample_pdf):
    parsed = parse_pdf(sample_pdf)
    assert len(parsed.pages) >= 2
    for page in parsed.pages:
        numbers = [l.number for l in page.lines]
        assert numbers == sorted(numbers), "line numbers must be monotonic"
        assert numbers and numbers[0] == 1 and numbers[-1] == len(numbers)
        assert all(l.text.strip() for l in page.lines)


def test_table_detected_and_flattened(sample_pdf):
    parsed = parse_pdf(sample_pdf)
    assert parsed.n_tables >= 1
    table_lines = [l for p in parsed.pages for l in p.lines if l.is_table]
    assert len(table_lines) >= 3, "each table row should be its own line"
    assert any("Annual" in l.text and "24" in l.text for l in table_lines)
    assert any("|" in l.text for l in table_lines)


def test_table_text_not_duplicated(sample_pdf):
    parsed = parse_pdf(sample_pdf)
    non_table = [l.text for p in parsed.pages for l in p.lines if not l.is_table]
    # the raw 'Annual' cell text inside the table region must not also appear
    # as a standalone non-table line
    assert not any(t.strip() in ("Annual", "24", "14 days") for t in non_table)


def test_body_font_size(sample_pdf):
    parsed = parse_pdf(sample_pdf)
    assert 9 <= body_font_size(parsed) <= 12
