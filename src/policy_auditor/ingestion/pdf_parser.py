"""PDF parsing with PyMuPDF.

Every extracted line keeps:
  - its 1-based page number
  - its 1-based line number within the page
  - its bounding box (for the UI evidence overlay)
  - whether it belongs to a table

Tables detected via ``page.find_tables()`` are flattened into one markdown-style
row per source row, each occupying exactly one line number — so a citation can
point at a single table row: [Page 7, Line 14].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class DocLine:
    page: int  # 1-based
    number: int  # 1-based within the page
    text: str
    bbox: tuple[float, float, float, float]
    size: float  # dominant font size (heading detection)
    bold: bool
    is_table: bool = False


@dataclass
class ParsedPage:
    page: int  # 1-based
    lines: list[DocLine] = field(default_factory=list)
    n_tables: int = 0


@dataclass
class ParsedDoc:
    source: str  # file name, used in citations
    path: Path
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def n_tables(self) -> int:
        return sum(p.n_tables for p in self.pages)

    @property
    def n_lines(self) -> int:
        return sum(len(p.lines) for p in self.pages)


def _bbox_center(bbox) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2, (y0 + y1) / 2


def _inside(point, bbox, margin: float = 2.0) -> bool:
    x, y = point
    return bbox[0] - margin <= x <= bbox[2] + margin and bbox[1] - margin <= y <= bbox[3] + margin


def _table_to_lines(page, table, words) -> list[str]:
    """Flatten a detected table into one ' | '-joined line per row.

    Cell text is rebuilt by assigning each word to the cell whose bbox contains
    the word's center — robust to wrapped multi-line cells, unlike extract().
    """
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if cell is None:
                cells.append("")
                continue
            x0, y0, x1, y1 = cell
            cell_words = [
                (round(w[1]), w[0], w[4])
                for w in words
                if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1
            ]
            cell_words.sort(key=lambda t: (t[0], t[1]))
            cells.append(" ".join(w[2] for w in cell_words))
        line = " | ".join(cells).strip()
        if line.replace("|", "").strip():
            rows.append(line)
    return rows


def _is_bold(spans) -> bool:
    if not spans:
        return False
    return all("bold" in (s.get("font") or "").lower() for s in spans if s.get("text", "").strip())


def parse_pdf(path: str | Path) -> ParsedDoc:
    path = Path(path)
    doc = fitz.open(path)
    parsed = ParsedDoc(source=path.name, path=path)

    try:
        for page_index, page in enumerate(doc, start=1):
            parsed_page = ParsedPage(page=page_index)

            # Detected tables: rows flattened, text lines inside their bbox suppressed
            tables = []
            try:
                finder = page.find_tables()
                tables = list(finder.tables)
            except Exception:
                tables = []

            table_lines: dict[int, list[str]] = {}
            table_bbox: dict[int, tuple] = {}
            words = page.get_text("words")
            for t_index, table in enumerate(tables):
                lines = _table_to_lines(page, table, words)
                if not lines:
                    continue
                table_lines[t_index] = lines
                table_bbox[t_index] = tuple(table.bbox)
                parsed_page.n_tables += 1

            emitted_tables: set[int] = set()
            line_no = 0

            def _emit_table(t_index: int) -> None:
                nonlocal line_no
                for row in table_lines[t_index]:
                    line_no += 1
                    parsed_page.lines.append(
                        DocLine(
                            page=page_index,
                            number=line_no,
                            text=row,
                            bbox=table_bbox[t_index],
                            size=0.0,
                            bold=False,
                            is_table=True,
                        )
                    )

            raw = page.get_text("dict", sort=True)
            for block in raw.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    center = _bbox_center(line["bbox"])

                    # emit any table whose region we have entered (reading order)
                    for t_index, tb in table_bbox.items():
                        if t_index not in emitted_tables and _inside(center, tb):
                            emitted_tables.add(t_index)
                            _emit_table(t_index)

                    # suppress original text lines that fall inside a table region
                    if any(_inside(center, tb) for tb in table_bbox.values()):
                        continue

                    line_no += 1
                    size = max(s.get("size", 0.0) for s in spans)
                    parsed_page.lines.append(
                        DocLine(
                            page=page_index,
                            number=line_no,
                            text=text,
                            bbox=tuple(line["bbox"]),
                            size=round(float(size), 2),
                            bold=_is_bold(spans),
                        )
                    )

            # tables never intersected by text flow (e.g. page with only a table)
            for t_index in table_lines:
                if t_index not in emitted_tables:
                    _emit_table(t_index)
                    emitted_tables.add(t_index)

            parsed.pages.append(parsed_page)
    finally:
        doc.close()

    return parsed


def body_font_size(parsed: ParsedDoc) -> float:
    """Most common line font size across the document — the baseline for headings."""
    counts: dict[float, int] = {}
    for p in parsed.pages:
        for line in p.lines:
            if not line.is_table:
                counts[line.size] = counts.get(line.size, 0) + 1
    if not counts:
        return 10.0
    return max(counts.items(), key=lambda kv: kv[1])[0]
