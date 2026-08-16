"""End-to-end ingestion: PDF -> parsed lines -> chunks -> embeddings -> Milvus."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from policy_auditor.indexing.embedder import get_embedder
from policy_auditor.indexing.milvus_store import MilvusStore
from policy_auditor.ingestion.chunker import chunk_parsed_doc
from policy_auditor.ingestion.pdf_parser import parse_pdf


@dataclass
class IngestReport:
    source: str
    pages: int
    tables: int
    chunks: int
    seconds: float


def ingest_pdf(store: MilvusStore, pdf_path: str | Path) -> IngestReport:
    t0 = time.perf_counter()
    pdf_path = Path(pdf_path)
    parsed = parse_pdf(pdf_path)
    chunks = chunk_parsed_doc(parsed)

    embedder = get_embedder()
    store.delete_source(parsed.source)  # idempotent re-ingest

    vectors = embedder.encode_passages([c.text for c in chunks])
    rows = [
        {
            "text": c.text,
            "dense": dense,
            "source": c.source,
            "page": c.page,
            "line_start": c.line_start,
            "line_end": c.line_end,
            "section": (c.section or "")[:500],
            "contains_table": c.contains_table,
        }
        for c, dense in zip(chunks, vectors)
    ]
    inserted = store.insert_chunks(rows)

    return IngestReport(
        source=parsed.source,
        pages=len(parsed.pages),
        tables=parsed.n_tables,
        chunks=inserted,
        seconds=round(time.perf_counter() - t0, 2),
    )


def ingest_directory(store: MilvusStore, directory: str | Path, rebuild: bool = False) -> list[IngestReport]:
    directory = Path(directory)
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {directory}")

    store.ensure_collection(rebuild=rebuild)
    reports = []
    for pdf in pdfs:
        reports.append(ingest_pdf(store, pdf))
    return reports
