"""LLM-as-judge reward: an accuracy score from a language model (optional).

The lexical reward terms (faithfulness, coverage, key sentences) cannot catch
*semantic* errors — e.g. writing 소대 where the source said 소총중대, or swapping
which unit took casualties — because the wrong word is still lexically present in
the source. An LLM judge reads the source and the summary and scores how
faithfully the summary reports it, catching those errors.

``JudgeModel`` is the pluggable interface (like ``FaithfulnessModel``). Two ways
to supply one:

* :class:`BackboneJudge` — reuse the *local* frozen backbone as the judge. No
  external API, so it can run inside the RL loop, at the cost of one extra
  generation per scored summary (opt in via ``RewardConfig.w_judge > 0``; it is
  off by default because it is expensive).
* Your own object with ``score(source, summary) -> float | None`` — e.g. wrap an
  external API (Gemini). Prefer this for *offline* checkpoint evaluation rather
  than the per-rollout training loop: an external call per rollout is far too
  slow / rate-limited to train against.

The rubric deliberately scores *accuracy and non-fabrication*, not fluency, so a
grammatical-but-wrong summary does not win points.
"""

from __future__ import annotations

import re
from typing import Protocol

from .config import RewardConfig
from .llm_backend import LLMBackend


class JudgeModel(Protocol):
    """Score how faithfully `summary` reports `source`, in [0, 1] (or None)."""

    def score(self, source: str, summary: str) -> float | None: ...


_SCORE_RE = re.compile(r"\d{1,3}")


class BackboneJudge:
    """LLM judge backed by the local frozen backbone (via ``generate_text``).

    Prompts the model for a 0-100 accuracy score and maps it to [0, 1]. Returns
    None when the backend cannot generate text (so the reward simply skips the
    judge term) or when no number can be parsed. Cached per (source, summary).
    """

    def __init__(self, backend: LLMBackend, config: RewardConfig):
        self.backend = backend
        self.max_new_tokens = config.judge_max_new_tokens
        self._cache: dict[tuple[str, str], float | None] = {}

    def _prompt(self, source: str, summary: str) -> str:
        return (
            "당신은 군사 보고서 요약을 채점하는 심사관이다. 아래 [요약]이 [원문]을 "
            "얼마나 정확히 반영했는지 0~100 정수 하나로만 평가하라.\n"
            "채점 기준: (1) 부대·수치·지명·시간을 정확히 반영했는가, "
            "(2) 원문에 없는 부대·사건·숫자를 지어내지 않았는가, "
            "(3) 각 제대의 상황과 핵심 조치·건의를 빠짐없이 담았는가.\n"
            "정확할수록 100, 지어내거나 틀릴수록 0. 숫자만 출력하라.\n\n"
            f"[원문]\n{source}\n\n[요약]\n{summary}\n\n[점수(0-100)]\n"
        )

    def score(self, source: str, summary: str) -> float | None:
        key = (source, summary)
        if key in self._cache:
            return self._cache[key]
        try:
            text = self.backend.generate_text(self._prompt(source, summary), self.max_new_tokens)
        except NotImplementedError:
            self._cache[key] = None
            return None
        m = _SCORE_RE.search(text or "")
        val = None if m is None else max(0.0, min(1.0, int(m.group()) / 100.0))
        self._cache[key] = val
        return val
