"""BGE embeddings via sentence-transformers (local, CPU-friendly)."""

from __future__ import annotations

from functools import lru_cache

from policy_auditor.config import get_settings

# BGE v1.5 models recommend this instruction prefix for retrieval queries.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: keeps tests torch-free

        self.model_name = model_name or get_settings().embed_model
        self._model = SentenceTransformer(self.model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]

    def encode_query(self, text: str) -> list[float]:
        if "bge" in self.model_name.lower():
            text = _QUERY_PREFIX + text
        vec = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )[0]
        return vec.tolist()


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
