"""Structure-aware chunking of parsed PDFs.

Rules:
  - chunks never cross a page boundary (citations stay unambiguous)
  - a detected heading starts a new chunk (policy sections are natural units)
  - a table is atomic — all of its rows stay in one chunk together
  - small tail overlap carries context into the next chunk
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from policy_auditor.config import get_settings
from policy_auditor.ingestion.pdf_parser import ParsedDoc, body_font_size

_TRAILING_PUNCT = re.compile(r"[.:;]$")


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    line_start: int
    line_end: int
    bboxes: list[tuple[int, tuple[float, float, float, float]]] = field(default_factory=list)
    section: str = ""
    contains_table: bool = False

    @property
    def citation(self) -> str:
        if self.line_start == self.line_end:
            return f"{self.source}, Page {self.page}, Line {self.line_start}"
        return f"{self.source}, Page {self.page}, Lines {self.line_start}-{self.line_end}"


def _is_heading(line, body_size: float) -> bool:
    if line.is_table or not line.text.strip():
        return False
    text = line.text.strip()
    if len(text) > 90:
        return False
    bigger_font = line.size >= body_size + 1.2 and line.size > 0
    numbered_section = re.match(r"^(\d+(\.\d+)*)\s+\S", text) and len(text) < 70
    short_bold = line.bold and len(text) < 70 and not _TRAILING_PUNCT.search(text)
    return bigger_font or numbered_section or short_bold


def chunk_parsed_doc(
    parsed: ParsedDoc,
    target_chars: int | None = None,
    max_chars: int | None = None,
    overlap_lines: int | None = None,
) -> list[Chunk]:
    s = get_settings()
    target = target_chars or s.chunk_target_chars
    limit = max_chars or s.chunk_max_chars
    overlap = overlap_lines if overlap_lines is not None else s.chunk_overlap_lines

    body_size = body_font_size(parsed)
    chunks: list[Chunk] = []
    current_section = ""

    for page in parsed.pages:
        pending: list = []  # DocLines of the chunk being built
        pending_table_ids: set[int] = set()

        def flush(carry_overlap: bool = True) -> None:
            nonlocal pending
            if pending:
                chunks.append(_build_chunk(parsed.source, page.page, pending, current_section))
                # tail overlap (only plain lines, never split tables)
                if carry_overlap:
                    plain_lines = [ln for ln in pending if not ln.is_table]
                    tail = plain_lines[-overlap:] if overlap else []
                    pending = list(tail)
                else:
                    pending = []
                pending_table_ids.clear()

        for line in page.lines:
            if _is_heading(line, body_size):
                if sum(len(l.text) for l in pending) >= 150:
                    flush()
                current_section = line.text.strip()
                pending.append(line)
                continue

            # tables are atomic: if adding all remaining rows of this table would
            # overflow, flush first, then take the whole table
            if line.is_table:
                table_id = id(line.bbox)
                if table_id not in pending_table_ids:
                    if sum(len(l.text) for l in pending) + len(line.text) > limit:
                        flush()
                    pending_table_ids.add(table_id)
                pending.append(line)
                if sum(len(l.text) for l in pending) > limit * 1.5:
                    flush()  # pathological oversized table — keep it alone
                continue

            projected = sum(len(l.text) for l in pending) + len(line.text) + len(pending)
            if projected > limit and any(not l.is_table for l in pending):
                flush()
            if sum(len(l.text) for l in pending) + len(line.text) > target and not line.is_table:
                flush()
            pending.append(line)

        # Never carry lines into the next page: page-local chunks are required
        # for unambiguous citation mapping and evidence highlighting.
        flush(carry_overlap=False)

    return chunks


def fixed_chunk_parsed_doc(
    parsed: ParsedDoc,
    chunk_chars: int = 512,
    overlap_lines: int = 0,
) -> list[Chunk]:
    """Build a simple fixed-size baseline for retrieval experiments.

    The baseline still keeps chunks on one page and keeps complete extracted
    lines, so its results can be compared with the production chunker using
    the same citation-aware gold spans.  Unlike :func:`chunk_parsed_doc`, it
    deliberately ignores headings, table boundaries, and semantic sections.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be greater than zero")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be zero or greater")

    chunks: list[Chunk] = []
    for page in parsed.pages:
        pending: list = []

        def flush() -> None:
            nonlocal pending
            if not pending:
                return
            chunks.append(_build_chunk(parsed.source, page.page, pending, ""))
            pending = list(pending[-overlap_lines:]) if overlap_lines else []

        for line in page.lines:
            projected = len("\n".join(item.text for item in pending))
            if pending and projected + 1 + len(line.text) > chunk_chars:
                flush()
            pending.append(line)

        flush()

    return chunks


def _build_chunk(source: str, page: int, lines, section: str) -> Chunk:
    first, last = lines[0], lines[-1]
    return Chunk(
        text="\n".join(l.text for l in lines),
        source=source,
        page=page,
        line_start=first.number,
        line_end=last.number,
        bboxes=[(page, tuple(l.bbox)) for l in lines],
        section=section,
        contains_table=any(l.is_table for l in lines),
    )
