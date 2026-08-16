"""Milvus collection with native hybrid search.

One collection holds both signals per chunk:
  - ``dense``  : FLOAT_VECTOR (BGE embeddings, HNSW / COSINE)
  - ``sparse`` : SPARSE_FLOAT_VECTOR, produced *server-side* by a Milvus BM25
                 Function over the raw ``text`` field (english analyzer)

Retrieval issues both searches in one ``hybrid_search`` call and fuses the two
rankings with Reciprocal Rank Fusion (RRF, k=60).
"""

from __future__ import annotations

import hashlib
from typing import Any

from policy_auditor.config import get_settings

OUTPUT_FIELDS = ["chunk_id", "text", "source", "page", "line_start", "line_end", "section", "contains_table"]


def chunk_pk(source: str, page: int, line_start: int, text: str) -> int:
    digest = hashlib.md5(f"{source}|{page}|{line_start}|{text[:80]}".encode("utf-8")).hexdigest()
    return int(digest[:14], 16)  # fits comfortably in INT64


class MilvusStore:
    def __init__(self, uri: str | None = None, collection: str | None = None, dim: int | None = None):
        from pymilvus import MilvusClient  # lazy import

        s = get_settings()
        self.uri = uri or s.milvus_endpoint
        self.collection = collection or s.milvus_collection
        self.dim = dim or s.embed_dim
        self.client = MilvusClient(uri=self.uri)

    # ---------- schema ----------

    def ensure_collection(self, rebuild: bool = False) -> bool:
        """Create the hybrid collection if missing. Returns True if it exists after the call."""
        from pymilvus import DataType, Function, FunctionType

        if self.client.has_collection(self.collection):
            if not rebuild:
                return True
            self.client.drop_collection(self.collection)

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.INT64, is_primary=True)
        # milvus-lite ships only the 'standard' analyzer; full Milvus uses 'english'
        # (stemming + stopwords) — auto-select by backend.
        analyzer = "standard" if self.uri.endswith(".db") else "english"
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=8192,
            enable_analyzer=True,
            analyzer_params={"type": analyzer},
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("page", DataType.INT64)
        schema.add_field("line_start", DataType.INT64)
        schema.add_field("line_end", DataType.INT64)
        schema.add_field("section", DataType.VARCHAR, max_length=512)
        schema.add_field("contains_table", DataType.BOOL)

        # server-side BM25: sparse vector is derived from `text` on insert
        schema.add_function(
            Function(
                name="text_bm25_emb",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        self.client.create_collection(self.collection, schema=schema, index_params=index_params)
        return True

    def _ensure_loaded(self) -> None:
        """milvus-lite persists a 'released' state across processes; load fixes it."""
        try:
            self.client.load_collection(self.collection)
        except Exception:
            pass  # already loaded / server mode handles it

    # ---------- ingestion ----------

    def delete_source(self, source: str) -> None:
        self.client.delete(self.collection, filter=f'source == "{source}"')

    def insert_chunks(self, rows: list[dict[str, Any]]) -> int:
        """rows: dicts with text/dense/source/page/line_start/line_end/section/contains_table."""
        for r in rows:
            r["chunk_id"] = chunk_pk(r["source"], r["page"], r["line_start"], r["text"])
        res = self.client.insert(self.collection, data=rows)
        return int(res.get("insert_count", 0))

    # ---------- retrieval ----------

    def hybrid_search(self, query_text: str, dense_vec: list[float], limit: int = 8) -> list[dict[str, Any]]:
        """BM25 + dense search fused with RRF. Returns chunk dicts with ``rrf_score``."""
        from pymilvus import AnnSearchRequest, RRFRanker

        self._ensure_loaded()
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=limit,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=limit,
        )
        results = self.client.hybrid_search(
            self.collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(60),
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )

        hits: list[dict[str, Any]] = []
        for hit in results[0]:
            row = {f: hit.entity.get(f) for f in OUTPUT_FIELDS}
            row["rrf_score"] = hit.distance
            hits.append(row)
        return hits

    # ---------- misc ----------

    def stats(self) -> dict[str, Any]:
        if not self.client.has_collection(self.collection):
            return {"exists": False}
        stats = self.client.get_collection_stats(self.collection)
        return {"exists": True, "entities": stats.get("row_count", 0)}

    def list_sources(self) -> list[str]:
        if not self.client.has_collection(self.collection):
            return []
        self._ensure_loaded()
        res = self.client.query(
            self.collection,
            filter='chunk_id >= 0',
            output_fields=["source"],
            limit=16384,
        )
        return sorted({r["source"] for r in res})
