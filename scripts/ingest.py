"""Ingest policy PDFs into the Milvus hybrid index.

Usage:
  python scripts/ingest.py                       # ingest everything in data/policies
  python scripts/ingest.py --dir data/policies --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy_auditor.config import get_settings  # noqa: E402
from policy_auditor.indexing.milvus_store import MilvusStore  # noqa: E402
from policy_auditor.indexing.pipeline import ingest_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=None, help="directory of PDFs (default: data/policies)")
    parser.add_argument("--rebuild", action="store_true", help="drop and recreate the collection")
    args = parser.parse_args()

    settings = get_settings()
    directory = Path(args.dir) if args.dir else settings.policies_dir

    print(f"Connecting to Milvus at {settings.milvus_uri} ...")
    store = MilvusStore()

    print(f"Ingesting PDFs from {directory}\n")
    total = 0
    for report in ingest_directory(store, directory, rebuild=args.rebuild):
        total += report.chunks
        print(
            f"  {report.source}: {report.pages} pages, {report.tables} tables, "
            f"{report.chunks} chunks indexed ({report.seconds}s)"
        )

    stats = store.stats()
    print(f"\nDone. {total} chunks inserted; collection now holds {stats.get('entities', '?')} entities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
