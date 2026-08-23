"""Grounded answer generation via Groq, with citation extraction."""

from __future__ import annotations

import re

from policy_auditor.config import get_settings
from policy_auditor.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from policy_auditor.retrieval.retriever import RetrievedChunk

CITATION_RE = re.compile(
    r"\[(?P<source>[^\[\],]+?),\s*Page\s+(?P<page>\d+),\s*Lines?\s+(?P<a>\d+)(?:\s*[-–]\s*(?P<b>\d+))?\]"
)

INSUFFICIENT = "The provided policy documents do not contain sufficient information"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Thinking models (e.g. Qwen3) may leak <think> blocks — drop them."""
    return _THINK_BLOCK.sub("", text).strip()


class Answerer:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        s = get_settings()
        self.api_key = api_key or s.groq_api_key
        self.model = model or s.groq_model
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _groq(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        contexts = []
        for i, c in enumerate(chunks, start=1):
            header = (
                f"source={c.source} | page={c.page} | lines={c.line_start}-{c.line_end}"
                f" | section={c.section or '-'}"
            )
            contexts.append(f"{header}\n{c.text}")

        return _strip_reasoning(self._generate_with_retry(question, contexts))

    def _generate_with_retry(self, question: str, contexts: list[str]) -> str:
        """Groq free tier enforces tight TPM limits; thinking models burn tokens
        invisibly — honor 429 backoff windows instead of failing the query."""
        import time

        from groq import RateLimitError

        prompt = build_user_prompt(question, contexts)
        last_error: Exception | None = None
        for _ in range(6):
            try:
                completion = self._groq().chat.completions.create(
                    model=self.model,
                    temperature=get_settings().answer_temperature,
                    max_tokens=get_settings().answer_max_tokens,
                    # GPT-OSS keeps reasoning separate from the visible answer.
                    include_reasoning=False,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return (completion.choices[0].message.content or "").strip()
            except RateLimitError as exc:
                last_error = exc
                wait = 15.0
                match = re.search(r"try again in ([\d.]+)s", str(exc))
                if match:
                    wait = float(match.group(1)) + 1.5
                time.sleep(min(wait, 60.0))
        raise RuntimeError(f"Groq rate limit persisted after retries: {last_error}")


def extract_citations(answer: str) -> list[dict]:
    """Parse [source, Page N, Lines A-B] markers back out of a generated answer."""
    found = []
    for m in CITATION_RE.finditer(answer):
        found.append(
            {
                "source": m.group("source").strip(),
                "page": int(m.group("page")),
                "line_start": int(m.group("a")),
                "line_end": int(m.group("b") or m.group("a")),
            }
        )
    return found
