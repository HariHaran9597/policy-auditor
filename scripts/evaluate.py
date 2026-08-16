"""Run the golden-set evaluation: full pipeline per question, then RAGAS scoring.

Usage:
  python scripts/evaluate.py [--limit 5] [--no-expand]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy_auditor.auditor import PolicyAuditor  # noqa: E402
from policy_auditor.config import get_settings  # noqa: E402
from policy_auditor.evaluation.dataset import EVAL_QUERIES  # noqa: E402
from policy_auditor.evaluation.ragas_eval import run_ragas, save_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--no-expand", action="store_true", help="skip LLM query expansion")
    parser.add_argument("--resume", action="store_true", help="reuse checkpointed answers, skip done questions")
    parser.add_argument("--ragas-only", action="store_true", help="score the checkpointed records without re-running the pipeline")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY is not set (put it in .env). Evaluation needs the LLM.", file=sys.stderr)
        return 1

    queries = EVAL_QUERIES[: args.limit] if args.limit else EVAL_QUERIES
    checkpoint = settings.reports_dir / "eval_records.json"
    records: list[dict] = []
    if checkpoint.exists() and (args.resume or args.ragas_only):
        records = json.loads(checkpoint.read_text(encoding="utf-8"))
        print(f"Loaded {len(records)} checkpointed records from {checkpoint}")

    if not args.ragas_only:
        print(f"Evaluating {len(queries)} questions with full pipeline ...\n")
        auditor = PolicyAuditor()
        done = {r["question"] for r in records}
        for i, q in enumerate(queries, 1):
            if q["question"] in done:
                continue
            t0 = time.perf_counter()
            result = auditor.audit(q["question"], expand=not args.no_expand)
            records.append(
                {
                    "question": q["question"],
                    "answer": result.answer,
                    "contexts": [c.text for c in result.chunks],
                    "reference": q["ground_truth"],
                    "citations": [
                        f"{c.source}, Page {c.page}, Lines {c.line_start}-{c.line_end}"
                        for c in result.cited_chunks
                    ],
                    "timings_ms": result.timings_ms,
                    "wall_s": round(time.perf_counter() - t0, 2),
                }
            )
            settings.reports_dir.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  [{i}/{len(queries)}] {q['question'][:60]}... ({records[-1]['wall_s']}s)")
            time.sleep(4)  # stay under Groq free-tier TPM limits (thinking models burn fast)

    print("\nRunning RAGAS metrics (judge = Groq llama-3.3-70b) ...")
    report = run_ragas(records)

    out = settings.reports_dir / "ragas_report.json"
    save_report(report, out)

    print(f"\n{'=' * 60}")
    print(f"RAGAS over {report['n']} questions")
    print(f"  Faithfulness      : {report['mean_faithfulness']}")
    print(f"  Answer Relevancy  : {report['mean_answer_relevancy']}")
    print(f"{'=' * 60}")
    print(f"\nFull report: {out} (+ .md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
