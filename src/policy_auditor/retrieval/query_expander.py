"""LLM query expansion: rewrite the user question into N retrieval-oriented variants.

Diverse rewrites (paraphrase / keyword-focused / synonym+entity) each hit the
hybrid index, and the per-query rankings are fused — this is what lifts recall
for awkwardly-phrased HR questions.
"""

from __future__ import annotations

import json
import re

from policy_auditor.config import get_settings

_EXPANSION_PROMPT = """You expand a question about corporate HR / compliance policy documents \
into {n} different search queries for a retrieval system.

Variants must be genuinely different from each other and from the original:
1. a keyword-focused version (exact policy terms, codes, numbers kept verbatim)
2. a plain-language paraphrase a different employee might use
3. a version enriched with likely synonyms/related policy vocabulary

Rules:
- Keep every number, code, job grade, and proper noun EXACTLY as in the original.
- Each query must be a single line, max 30 words.
- Never answer the question; only rewrite it.

Original question: {question}

Respond with ONLY a JSON object: {{"queries": ["...", "...", "..."]}}"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in response")
    return json.loads(match.group(0))


class QueryExpander:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        s = get_settings()
        self.api_key = api_key or s.groq_api_key
        self.model = model or s.groq_expansion_model
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _groq(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def expand(self, question: str, n: int | None = None) -> list[str]:
        """Return the original question plus up to `n` rewritten variants."""
        n = n or get_settings().expansion_count
        if not self.available:
            return [question]

        try:
            completion = self._groq().chat.completions.create(
                model=self.model,
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a precise query-rewriting engine."},
                    {"role": "user", "content": _EXPANSION_PROMPT.format(n=n, question=question)},
                ],
            )
            data = _extract_json(completion.choices[0].message.content or "")
            queries = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        except Exception:
            return [question]

        seen, unique = {question.lower()}, [question]
        for q in queries:
            if q.lower() not in seen and len(q) < 250:
                unique.append(q)
                seen.add(q.lower())
        return unique[: 1 + n]
