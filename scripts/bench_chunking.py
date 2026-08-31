"""Benchmark chunking and reranking with retrieval hit-rate.

The benchmark uses the 16-question golden set but deliberately skips query
expansion and answer generation.  Each configuration gets its own temporary
Milvus collection, so the comparison measures retrieval changes rather than
LLM variability.

Usage:
  python scripts/bench_chunking.py
  python scripts/bench_chunking.py --limit 5 --keep-collections
  python scripts/bench_chunking.py --top-k 5 --json-out reports/my_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy_auditor.config import get_settings  # noqa: E402
from policy_auditor.evaluation.dataset import EVAL_QUERIES  # noqa: E402
from policy_auditor.indexing.embedder import get_embedder  # noqa: E402
from policy_auditor.indexing.milvus_store import MilvusStore  # noqa: E402
from policy_auditor.ingestion.chunker import (  # noqa: E402
    chunk_parsed_doc,
    fixed_chunk_parsed_doc,
)
from policy_auditor.ingestion.pdf_parser import ParsedDoc, parse_pdf  # noqa: E402
from policy_auditor.retrieval.reranker import get_reranker  # noqa: E402
from policy_auditor.retrieval.retriever import HybridRetriever  # noqa: E402


Chunker = Callable[[ParsedDoc], list]


@dataclass(frozen=True)
class BenchmarkConfig:
    key: str
    label: str
    chunker: Chunker
    rerank: bool = False


@dataclass
class BenchmarkResult:
    key: str
    label: str
    rerank: bool
    chunks: int
    answerable_questions: int
    full_evidence_hits: int
    hit_rate: float
    target_coverage: float
    elapsed_s: float
    per_question: list[dict]


def _configs() -> list[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            "fixed-512",
            "Fixed 512 chars",
            lambda doc: fixed_chunk_parsed_doc(doc, chunk_chars=512),
        ),
        BenchmarkConfig(
            "semantic",
            "Semantic structure-aware",
            lambda doc: chunk_parsed_doc(doc, overlap_lines=0),
        ),
        BenchmarkConfig(
            "semantic-overlap",
            "Semantic + 2-line overlap",
            lambda doc: chunk_parsed_doc(doc, overlap_lines=2),
        ),
        BenchmarkConfig(
            "semantic-overlap-reranker",
            "Semantic + overlap + BGE reranker",
            lambda doc: chunk_parsed_doc(doc, overlap_lines=2),
            rerank=True,
        ),
    ]


def _covers(chunk, evidence: dict) -> bool:
    return (
        chunk.source == evidence["source"]
        and chunk.page == evidence["page"]
        and chunk.line_start <= evidence["line_end"]
        and chunk.line_end >= evidence["line_start"]
    )


def _score_question(question: dict, top_chunks: list) -> dict:
    evidence = question.get("gold_evidence", [])
    covered = [any(_covers(chunk, target) for chunk in top_chunks) for target in evidence]
    return {
        "question": question["question"],
        "answerable": bool(evidence),
        "covered_targets": sum(covered),
        "total_targets": len(evidence),
        "full_evidence_hit": bool(evidence) and all(covered),
        "top_chunks": [chunk.citation for chunk in top_chunks],
    }


def _build_index(store: MilvusStore, parsed_docs: list[ParsedDoc], config: BenchmarkConfig) -> int:
    chunks = [chunk for doc in parsed_docs for chunk in config.chunker(doc)]
    if not chunks:
        raise RuntimeError(f"{config.label} produced no chunks")

    embedder = get_embedder()
    vectors = embedder.encode_passages([chunk.text for chunk in chunks])
    rows = [
        {
            "text": chunk.text,
            "dense": vector,
            "source": chunk.source,
            "page": chunk.page,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "section": (chunk.section or "")[:500],
            "contains_table": chunk.contains_table,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    store.ensure_collection(rebuild=True)
    return store.insert_chunks(rows)


def _run_config(
    config: BenchmarkConfig,
    parsed_docs: list[ParsedDoc],
    questions: list[dict],
    collection: str,
    top_k: int,
    first_stage_k: int,
    keep_collection: bool,
) -> BenchmarkResult:
    store = MilvusStore(collection=collection)
    started = time.perf_counter()
    inserted = _build_index(store, parsed_docs, config)
    retriever = HybridRetriever(store=store)
    reranker = get_reranker() if config.rerank else None
    scored = []

    for question in questions:
        # Expansion and generation are intentionally excluded: the benchmark
        # isolates the chunking and reranking stages and is deterministic apart
        # from the local embedding/reranker model.
        pool = retriever.retrieve([question["question"]], top_k=first_stage_k)
        if reranker is not None:
            top = reranker.rerank(question["question"], pool, top_k=top_k)
        else:
            top = pool[:top_k]
        scored.append(_score_question(question, top))

    answerable = [item for item in scored if item["answerable"]]
    full_hits = sum(item["full_evidence_hit"] for item in answerable)
    total_targets = sum(item["total_targets"] for item in answerable)
    covered_targets = sum(item["covered_targets"] for item in answerable)
    result = BenchmarkResult(
        key=config.key,
        label=config.label,
        rerank=config.rerank,
        chunks=inserted,
        answerable_questions=len(answerable),
        full_evidence_hits=full_hits,
        hit_rate=round(full_hits / len(answerable), 4) if answerable else 0.0,
        target_coverage=round(covered_targets / total_targets, 4) if total_targets else 0.0,
        elapsed_s=round(time.perf_counter() - started, 2),
        per_question=scored,
    )

    if not keep_collection:
        store.client.drop_collection(collection)
    return result


def _markdown_table(results: list[BenchmarkResult], top_k: int, n_questions: int) -> str:
    answerable = results[0].answerable_questions if results else 0
    lines = [
        f"## Chunking / reranking benchmark (full-evidence hit rate@{top_k})",
        "",
        "A question is a hit only when every annotated evidence span has at least one overlapping retrieved chunk. "
        f"The {answerable} answerable questions form the denominator; designed refusals are excluded.",
        "",
        "| Configuration | Chunks | Reranker | Full-evidence hits | Hit rate | Δ vs previous | Target coverage |",
        "|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for index, result in enumerate(results):
        delta = "—" if index == 0 else f"{(result.hit_rate - results[index - 1].hit_rate) * 100:+.1f} pp"
        lines.append(
            f"| {result.label} | {result.chunks} | {'yes' if result.rerank else 'no'} | "
            f"{result.full_evidence_hits}/{result.answerable_questions} | "
            f"{result.hit_rate:.1%} | {delta} | {result.target_coverage:.1%} |"
        )

    lines.extend([
        "",
        "### Per-question results",
        "",
        "| Question | " + " | ".join(result.key for result in results) + " |",
        "|---|" + "---:|" * len(results),
    ])
    for index, question in enumerate(results[0].per_question):
        cells = []
        for result in results:
            item = result.per_question[index]
            cells.append("hit" if item["full_evidence_hit"] else ("—" if not item["answerable"] else "miss"))
        lines.append(f"| {question['question']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _print_table(results: list[BenchmarkResult], top_k: int, n_questions: int) -> None:
    answerable = results[0].answerable_questions if results else 0
    print(f"\nRetrieval hit-rate benchmark (full-evidence hit@{top_k}; {answerable} answerable / {n_questions} total)\n")
    print("Configuration                         Chunks  Rerank  Hits       Hit rate  Δ prev     Coverage")
    print("------------------------------------  ------  ------  ---------  --------  --------  --------")
    for index, result in enumerate(results):
        delta = "—" if index == 0 else f"{(result.hit_rate - results[index - 1].hit_rate) * 100:+.1f} pp"
        print(
            f"{result.label:<36}  {result.chunks:>6}  {'yes' if result.rerank else 'no':>6}  "
            f"{result.full_evidence_hits:>2}/{result.answerable_questions:<6}  "
            f"{result.hit_rate:>7.1%}  {delta:>9}  {result.target_coverage:>7.1%}"
        )
    print("\nDesigned refusal: excluded from retrieval denominator (no evidence span is expected).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="benchmark only the first N golden questions")
    parser.add_argument("--top-k", type=int, default=1, help="number of final chunks counted for a hit (default: 1)")
    parser.add_argument("--first-stage-k", type=int, default=8, help="hybrid candidates per query")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="JSON output path (default: reports/chunking_benchmark.json)",
    )
    parser.add_argument("--keep-collections", action="store_true", help="keep temporary benchmark collections")
    args = parser.parse_args()
    if args.top_k <= 0 or args.first_stage_k <= 0:
        parser.error("--top-k and --first-stage-k must be greater than zero")

    settings = get_settings()
    pdfs = sorted(settings.policies_dir.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: no PDFs found in {settings.policies_dir}", file=sys.stderr)
        return 1
    questions = EVAL_QUERIES[: args.limit] if args.limit else EVAL_QUERIES
    if any("gold_evidence" not in question for question in questions):
        print("ERROR: every benchmark question needs a gold_evidence annotation", file=sys.stderr)
        return 1

    print(f"Parsing {len(pdfs)} policy PDFs ...")
    parsed_docs = [parse_pdf(path) for path in pdfs]
    prefix = f"policy_bench_{os.getpid()}"
    results = []
    collections = []
    try:
        for config in _configs():
            collection = f"{prefix}_{config.key.replace('-', '_')}"
            collections.append(collection)
            print(f"Indexing {config.label} ...")
            results.append(
                _run_config(
                    config,
                    parsed_docs,
                    questions,
                    collection,
                    args.top_k,
                    args.first_stage_k,
                    args.keep_collections,
                )
            )
    finally:
        if not args.keep_collections:
            # Best-effort cleanup also covers an exception after collection
            # creation, without touching the user's production collection.
            for collection in collections:
                try:
                    MilvusStore(collection=collection).client.drop_collection(collection)
                except Exception:
                    pass

    _print_table(results, args.top_k, len(questions))
    output = args.json_out or (settings.reports_dir / "chunking_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "first_stage_k": args.first_stage_k,
        "n_questions": len(questions),
        "n_answerable": sum(bool(q.get("gold_evidence")) for q in questions),
        "configs": [asdict(result) for result in results],
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown_table(results, args.top_k, len(questions)), encoding="utf-8")
    print(f"\nSaved: {output} (+ .md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
