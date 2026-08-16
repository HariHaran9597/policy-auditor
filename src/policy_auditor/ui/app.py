"""Enterprise Policy Auditor — Streamlit UI.

Left:  the grounded answer, parsed citations, and the reranked evidence table.
Right: the actual source PDF page with the cited lines highlighted in red
       (other retrieved chunks outlined in blue).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from policy_auditor.config import PROJECT_ROOT, get_settings  # noqa: E402

st.set_page_config(page_title="Enterprise Policy Auditor", page_icon="🛡️", layout="wide")

SAMPLE_QUESTIONS = [
    "How many days of paternity leave do I get and when must it be used?",
    "What is the hotel ceiling per night for a Grade L3 employee travelling in India?",
    "What are the acknowledgement and resolution targets for a Sev2 security incident?",
    "What is the monthly internet allowance for a fully-remote Grade L1 employee as of March 2026?",
    "Can I carry unused sick leave into next year?",
]


@st.cache_resource(show_spinner="Loading models (first run downloads them) ...")
def get_auditor():
    from policy_auditor.auditor import PolicyAuditor

    return PolicyAuditor()


@st.cache_data(show_spinner=False)
def render_page(source: str, page: int, focus_key: str, other_keys: tuple) -> bytes:
    """Render a PDF page as PNG with focus chunk highlighted, others outlined."""
    import fitz

    path = PROJECT_ROOT / "data" / "policies" / source
    doc = fitz.open(path)
    try:
        pg = doc[page - 1]
        if focus_key:
            src, p, ls, le = focus_key.split("|")
            if int(p) == page:
                rects = _chunk_rects(doc, page - 1, src, int(ls), int(le))
                for r in rects:
                    pg.draw_rect(fitz.Rect(r), color=(0.9, 0.1, 0.1), width=2.0)
        for key in other_keys:
            src, p, ls, le = key.split("|")
            if int(p) == page:
                for r in _chunk_rects(doc, page - 1, src, int(ls), int(le)):
                    pg.draw_rect(fitz.Rect(r), color=(0.2, 0.4, 0.9), width=1.0)
        pix = pg.get_pixmap(matrix=fitz.Matrix(2.2, 2.2))
        return pix.tobytes("png")
    finally:
        doc.close()


def _chunk_rects(doc, page_index: int, source: str, line_start: int, line_end: int):
    """Locate the bbox of a cited line range by re-parsing just that page."""
    from policy_auditor.ingestion.pdf_parser import parse_pdf

    parsed = parse_pdf(PROJECT_ROOT / "data" / "policies" / source)
    page = parsed.pages[page_index]
    rects = []
    for line in page.lines:
        if line.is_table and line_start <= line.number <= line_end:
            rects.append(tuple(line.bbox))
        elif line_start <= line.number <= line_end:
            rects.append(tuple(line.bbox))
    # merge vertically overlapping plain-text rects into bands to reduce clutter
    merged: list[list[float]] = []
    for r in sorted(rects, key=lambda r: r[1]):
        if merged and r[1] <= merged[-1][3] + 2 and abs(r[0] - merged[-1][0]) < 40:
            merged[-1][3] = max(merged[-1][3], r[3])
            merged[-1][2] = max(merged[-1][2], r[2])
        else:
            merged.append([r[0], r[1], r[2], r[3]])
    return [tuple(m) for m in merged]


def _doc_pages(source: str) -> int:
    import fitz

    doc = fitz.open(PROJECT_ROOT / "data" / "policies" / source)
    n = doc.page_count
    doc.close()
    return n


# ---------------------------------------------------------------- sidebar
settings = get_settings()
with st.sidebar:
    st.title("🛡️ Policy Auditor")
    st.caption("Hybrid retrieval • Cross-encoder reranking • Cited answers")

    st.divider()
    st.subheader("Index status")
    try:
        from policy_auditor.indexing.milvus_store import MilvusStore

        stats = MilvusStore().stats()
        if stats.get("exists"):
            st.metric("Indexed chunks", stats.get("entities", 0))
            sources = MilvusStore().list_sources()
            for s in sources:
                st.markdown(f"- 📄 `{s}`")
        else:
            st.warning("Collection not found. Run `python scripts/ingest.py` first.")
    except Exception as exc:
        st.error(f"Milvus unreachable: {exc}")

    st.divider()
    if not settings.groq_api_key:
        st.error("**GROQ_API_KEY missing** — set it in `.env` and restart.")

    st.divider()
    st.caption(
        f"answer model: `{settings.groq_model}`\n\n"
        f"expansion model: `{settings.groq_expansion_model}`\n\n"
        f"embeddings: `{settings.embed_model}`\n\n"
        f"reranker: `{settings.reranker_model}`"
    )

# ---------------------------------------------------------------- main
st.header("Ask the policy corpus")
st.caption("Answers cite every claim as [document, Page N, Lines A-B] — open the evidence on the right.")

question = st.text_input(
    "Question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is the notice period for a Grade L3 resignation?",
)
asked = st.button("Audit the policy", type="primary", use_container_width=True)

with st.expander("Try a sample question"):
    cols = st.columns(2)
    for i, sq in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 2].button(sq, key=f"sq{i}"):
            st.session_state["question"] = sq
            st.rerun()

result = None
if asked and question.strip():
    if not settings.groq_api_key:
        st.error("GROQ_API_KEY is not set — generation is disabled. Add it to .env and restart.")
        st.stop()

    auditor = get_auditor()
    with st.status("Auditing ...", expanded=True) as status:
        import time as _t

        st.write("🔄 Expanding the question with the LLM ...")
        t0 = _t.perf_counter()
        queries = auditor.expander.expand(question)
        st.write(f"{len(queries)} retrieval variants ready")

        st.write("🔎 Hybrid search (BM25 + dense, fused with RRF) ...")
        pool = auditor.retriever.retrieve(queries)
        st.write(f"{len(pool)} candidate chunks fused")

        st.write("🎚️ Cross-encoder reranking ...")
        top = auditor._get_reranker().rerank(question, pool)

        st.write("✍️ Generating the grounded answer ...")
        from policy_auditor.auditor import AuditResult
        from policy_auditor.generation.answerer import extract_citations

        answer = auditor.answerer.answer(question, top)
        result = AuditResult(
            question=question,
            answer=answer,
            chunks=top,
            expanded_queries=queries,
            citations=extract_citations(answer),
            timings_ms={
                "expansion_ms": round((_t.perf_counter() - t0) * 1000, 1),
            },
        )
        status.update(label="Done", state="complete", expanded=False)
elif not asked:
    st.info("Ask a question above, or pick a sample below the input box.")

if result is not None:
    left, right = st.columns([7, 5], gap="large")

    with left:
        st.subheader("Answer")
        st.markdown(result.answer)

        if result.citations:
            st.markdown("**Cited evidence** (click to view the source page →)")
            cited_chunks = result.cited_chunks or result.chunks[:1]
            for i, c in enumerate(cited_chunks):
                if st.button(f"📍 {c.citation}", key=f"cit{i}"):
                    st.session_state["focus"] = f"{c.source}|{c.page}|{c.line_start}|{c.line_end}"
                    st.session_state["view_doc"] = c.source
                    st.session_state["view_page"] = c.page

        ev = st.expander("Retrieval detail (top chunks after reranking)")
        with ev:
            rows = [
                {
                    "rank": i + 1,
                    "citation": c.citation,
                    "rerank": c.rerank_score,
                    "rrf": c.rrf_score,
                    "queries_hit": c.n_queries_hit,
                    "table": "✓" if c.contains_table else "",
                }
                for i, c in enumerate(result.chunks)
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(
                "rerank = BGE cross-encoder relevance • rrf = fused hybrid rank • "
                "queries_hit = how many expanded queries retrieved this chunk"
            )
            with st.expander("Expanded queries"):
                for q in result.expanded_queries:
                    st.markdown(f"- {q}")

    with right:
        st.subheader("Source evidence")
        sources = sorted({c.source for c in result.chunks}) or ["Acme_Employee_Handbook_2026.pdf"]
        view_doc = st.selectbox(
            "Document",
            sources,
            index=sources.index(st.session_state.get("view_doc")) if st.session_state.get("view_doc") in sources else 0,
        )
        pages = _doc_pages(view_doc)
        default_page = st.session_state.get("view_page", 1)
        view_page = st.number_input("Page", min_value=1, max_value=pages, value=min(default_page, pages))

        focus = st.session_state.get("focus", "")
        focus_key = focus if focus.startswith(view_doc + "|") else ""
        other_keys = tuple(
            f"{c.source}|{c.page}|{c.line_start}|{c.line_end}"
            for c in result.chunks
            if not (focus_key and c.source == view_doc and c.page == view_page)
        )[:3]
        try:
            png = render_page(view_doc, int(view_page), focus_key, other_keys)
            st.image(io.BytesIO(png), use_container_width=True)
            st.caption(
                "🔴 highlighted = cited lines &nbsp;·&nbsp; 🔵 outline = other top-ranked chunks"
            )
        except FileNotFoundError:
            st.error(f"PDF {view_doc} not found under data/policies — drop it there to enable the viewer.")

        with st.expander("Chunk text"):
            key = focus_key or next(
                (f"{c.source}|{c.page}|{c.line_start}|{c.line_end}" for c in result.chunks if c.source == view_doc),
                "",
            )
            match = next(
                (c for c in result.chunks if f"{c.source}|{c.page}|{c.line_start}|{c.line_end}" == key),
                None,
            )
            if match:
                st.text(match.text)
