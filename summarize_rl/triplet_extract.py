"""Inference-time triplet extraction (source-side, cached).

Training feeds the SQ branch corpus triplets, and the trained policy leans on
that branch heavily (large c). At inference the caller usually has only the raw
report and no triplets, so the SQ branch is empty and the policy — routing much
of its weight to an empty branch — fabricates. This module fills that gap: the
frozen LLM extracts (head, relation, tail) triplets from the source, so the SQ
branch carries real content the way it did during training.

The triplets depend only on the source, so they are cached per unique source.
"""

from __future__ import annotations

import re

from .branches import Triplet
from .llm_backend import LLMBackend

_LEAD = re.compile(r"^\s*(?:\(?\d+\)?[.)]?|[가-힣][.)]|[-*•·])\s*")


def _parse_triplets(text: str, n: int) -> list[Triplet]:
    """Parse '주체 | 관계 | 대상' (or '주체, 관계, 대상') lines into Triplets."""
    out: list[Triplet] = []
    for line in (text or "").splitlines():
        s = _LEAD.sub("", line).strip().strip("()").strip()
        if not s:
            continue
        if s.count("|") == 2:
            parts = [p.strip() for p in s.split("|")]
        elif s.count(",") == 2:
            parts = [p.strip() for p in s.split(",")]
        else:
            continue
        if all(parts) and all(len(p) <= 40 for p in parts):
            out.append(Triplet(*parts))
        if len(out) >= n:
            break
    return out


class TripletExtractor:
    """Extract and cache a source's (head, relation, tail) triplets via the LLM."""

    def __init__(self, n: int = 12, max_new_tokens: int = 384):
        self.n = n
        self.max_new_tokens = max_new_tokens
        self._cache: dict[str, list[Triplet]] = {}

    def _prompt(self, source: str) -> str:
        return (
            f"다음 보고서에서 핵심 사실을 '주체 | 관계 | 대상' 형식의 triplet으로 "
            f"최대 {self.n}개 추출하라. 원문에 있는 부대·수치·지명만 사용하고, "
            "각 triplet을 한 줄에 하나씩 '주체 | 관계 | 대상' 형식으로만 출력하라.\n\n"
            f"[보고서]\n{source}\n\n[Triplet]\n"
        )

    def extract(self, backend: LLMBackend, source: str) -> list[Triplet]:
        """Return cached triplets for `source`, extracting on first use."""
        if source in self._cache:
            return self._cache[source]
        try:
            text = backend.generate_text(self._prompt(source), self.max_new_tokens)
            triplets = _parse_triplets(text, self.n)
        except NotImplementedError:
            triplets = []  # backend can't generate -> leave SQ branch empty
        self._cache[source] = triplets
        return triplets
