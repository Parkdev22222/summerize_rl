"""LLM-extracted key sentences for the reward (source-side, cached).

The frozen LLM is prompted once per source to pick the N most important
sentences of the report. Those key sentences depend only on the source, so they
are cached and reused across every rollout and training step touching that
source — the extraction cost is paid once per unique document, not per reward.

The reward then measures how much of the key-sentence content the summary
reflects (see rewards.key_sentence_coverage).
"""

from __future__ import annotations

import re

from .config import RewardConfig
from .llm_backend import LLMBackend

# Split model output into candidate sentences: newlines or sentence enders,
# then strip list markers ("1.", "-", "•") and surrounding whitespace.
_SPLIT = re.compile(r"[\n]+|(?<=[.!?。])\s+")
_LEAD = re.compile(r"^\s*(?:\d+[.)]|[-*•·])\s*")


def _parse_sentences(text: str, n: int) -> list[str]:
    out = []
    for piece in _SPLIT.split(text or ""):
        s = _LEAD.sub("", piece).strip()
        if len(s) >= 2:
            out.append(s)
        if len(out) >= n:
            break
    return out


class KeySentenceExtractor:
    """Extracts and caches the source's key sentences via the frozen LLM."""

    def __init__(self, config: RewardConfig):
        self.n = config.keysent_n
        self.max_new_tokens = config.keysent_max_new_tokens
        self._cache: dict[str, list[str]] = {}

    def _prompt(self, source: str) -> str:
        return (
            f"다음 보고서에서 가장 중요한 핵심 문장 {self.n}개를 원문에서 골라 "
            "한 줄에 하나씩 출력하시오.\n\n"
            f"[보고서]\n{source}\n\n[핵심 문장]\n"
        )

    def extract(self, backend: LLMBackend, source: str) -> list[str]:
        """Return the cached key sentences for `source`, extracting on first use."""
        if source in self._cache:
            return self._cache[source]
        try:
            text = backend.generate_text(self._prompt(source), self.max_new_tokens)
            sents = _parse_sentences(text, self.n)
        except NotImplementedError:
            sents = []  # backend can't generate text -> reward skips this term
        self._cache[source] = sents
        return sents
