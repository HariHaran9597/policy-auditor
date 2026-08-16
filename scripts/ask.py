"""Ask a policy question from the command line (smoke test / quick demo).

Usage:
  python scripts/ask.py "How many days of paternity leave do I get?"
  python scripts/ask.py "..." --no-expand --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy_auditor.auditor import PolicyAuditor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--no-expand", action="store_true", help="skip LLM query expansion")
    parser.add_argument("--verbose", action="store_true", help="show pipeline internals")
    args = parser.parse_args()

    auditor = PolicyAuditor()
    result = auditor.audit(args.question, expand=not args.no_expand)

    if args.verbose:
        print("Expanded queries:")
        for q in result.expanded_queries:
            print(f"  - {q}")
        print("\nTop chunks after reranking:")
        for c in result.chunks:
            print(
                f"  [{c.rerank_score:.4f}] {c.citation} "
                f"(rrf={c.rrf_score:.4f}, hit_by={c.n_queries_hit} queries, table={c.contains_table})"
            )
        print(f"\nTimings: {result.timings_ms}")
        print()

    print("ANSWER:\n" + result.answer)

    if result.citations:
        print("\nCitations parsed from answer:")
        for cit in result.citations:
            print(f"  - {cit['source']}, Page {cit['page']}, Lines {cit['line_start']}-{cit['line_end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
