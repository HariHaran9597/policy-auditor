"""RAGAS evaluation: Faithfulness + Response Relevancy over the golden set.

Judge LLM: Groq (llama-3.3-70b, non-thinking) wrapped via LangchainLLMWrapper.
Embeddings for relevancy: the same local BGE model used for retrieval.

Measurement hygiene:
  - citation brackets are stripped from answers before scoring — faithfulness
    must measure content grounding, not how `[doc, Page N, Lines A-B]` tokens
    decompose into pseudo-claims (citations are verified separately by exact
    page/line mapping)
  - designed-refusal questions are excluded from the relevancy mean (RAGAS
    scores any refusal 0 by construction) and reported as their own line
"""

from __future__ import annotations

import json
import re
from typing import Any

from policy_auditor.config import get_settings

_CITATION_BRACKET = re.compile(r"\[[^\[\]]*Page \d+[^\[\]]*\]")

# questions whose correct answer is a refusal
REFUSAL_QUESTIONS = {"tuition reimbursement"}


def _is_refusal(question: str) -> bool:
    q = question.lower()
    return any(marker in q for marker in REFUSAL_QUESTIONS)


def _strip_citations(text: str) -> str:
    return _CITATION_BRACKET.sub("", text).replace("  ", " ").strip()


def _is_na(value) -> bool:
    return value != value  # NaN is the only value not equal to itself


class _BGEAdapter:
    """Minimal LangChain Embeddings facade over our local BGE embedder."""

    def __init__(self) -> None:
        from policy_auditor.indexing.embedder import get_embedder

        self._embedder = get_embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode_passages(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode_query(text)


def build_ragas_wrappers():
    from langchain_groq import ChatGroq
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    s = get_settings()
    judge = LangchainLLMWrapper(
        ChatGroq(api_key=s.groq_api_key, model=s.groq_judge_model, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(_BGEAdapter())
    return judge, embeddings


def run_ragas(records: list[dict[str, Any]]) -> dict[str, Any]:
    """records: question / answer / contexts (list[str]) / reference."""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=_strip_citations(r["answer"]),
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in records
    ]
    judge, embeddings = build_ragas_wrappers()
    run_config = RunConfig(timeout=180, max_workers=2, max_retries=2)
    result = evaluate(
        dataset=EvaluationDataset(samples),
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
        show_progress=True,
    )

    df = result.to_pandas()
    per_sample = []
    for i, r in enumerate(records):
        row = df.iloc[i]
        f = None if _is_na(row["faithfulness"]) else float(row["faithfulness"])
        a = None if _is_na(row["answer_relevancy"]) else float(row["answer_relevancy"])
        per_sample.append({**r, "faithfulness": f, "answer_relevancy": a, "refusal": _is_refusal(r["question"])})

    def _mean(key: str, skip_refusal: bool = False) -> float | None:
        vals = [p[key] for p in per_sample if p[key] is not None and not (skip_refusal and p["refusal"])]
        return round(sum(vals) / len(vals), 4) if vals else None

    n_refusal = sum(p["refusal"] for p in per_sample)
    refusal_ok = all(
        "not contain sufficient information" in p["answer"].lower() for p in per_sample if p["refusal"]
    )
    return {
        "n": len(per_sample),
        "mean_faithfulness": _mean("faithfulness"),
        "mean_answer_relevancy": _mean("answer_relevancy", skip_refusal=True),
        "n_refusal_questions": n_refusal,
        "refusal_behaviour_correct": refusal_ok,
        "per_sample": per_sample,
    }


def save_report(report: dict[str, Any], out_path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# RAGAS Evaluation — Policy Auditor",
        "",
        f"- Samples: **{report['n']}** (of which {report['n_refusal_questions']} designed-refusal)",
        f"- Mean Faithfulness: **{report['mean_faithfulness']}**",
        f"- Mean Answer Relevancy (answerable questions): **{report['mean_answer_relevancy']}**",
        f"- Refusal behaviour correct: **{report['refusal_behaviour_correct']}**",
        "",
        "| # | Question | Faithfulness | Relevancy | Refusal |",
        "|---|----------|-------------:|----------:|---------|",
    ]
    for i, p in enumerate(report["per_sample"], 1):
        md.append(
            f"| {i} | {p['question'][:70]} | {p['faithfulness']} | {p['answer_relevancy']} |"
            f" {'yes' if p['refusal'] else ''} |"
        )
    out_path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
