"""Key sentences for the reward, weighted by military importance (cached).

Each source sentence is scored for *military importance* by rule: sentences
describing a military event (교전/파괴/기동/…), an echelon's status (중대/대대/
진지/…), casualties (전사/부상/…), a request (재보급/공중지원 요청/…), the enemy,
or carrying quantities (전차 4대, 40발). Any sentence that clears the importance
threshold is ALWAYS taken as a key sentence, and its importance becomes its
weight, so :func:`rewards.key_sentence_coverage` — a weighted average — makes the
summary's reflection of the *important* events count more than routine lines.

Optionally the frozen LLM is still queried for salient sentences and merged in
(``keysent_use_llm``); its picks are weighted by the same importance rule. Key
sentences depend only on the source, so the result is cached per unique source —
the extraction cost is paid once per document, not per rollout.
"""

from __future__ import annotations

import re

from .config import RewardConfig
from .llm_backend import LLMBackend

# Split text into candidate sentences: newlines or sentence enders, then strip
# list/enumeration markers ("1.", "가.", "(1)", "-", "•") and surrounding space.
_SPLIT = re.compile(r"[\n]+|(?<=[.!?。])\s+")
_LEAD = re.compile(r"^\s*(?:\(?\d+\)?[.)]?|[가-힣][.)]|[-*•·])\s*")

# Military-importance keyword categories: (keywords, weight). A sentence's
# importance is the sum of the weights of the categories it hits, plus a bonus
# for carrying a quantity. Tuned so a combat event with casualties/quantities
# scores well above a routine administrative line.
_IMPORTANCE_KEYWORDS: list[tuple[tuple[str, ...], float]] = [
    # 군사적 사건 (전투 행위)
    (("교전", "공격", "방어", "파괴", "격퇴", "저지", "침투", "기동", "확보", "점령",
      "타격", "사격", "피격", "전개", "우회", "돌파", "공세", "남하", "포착", "식별",
      "매복", "제압", "소탕", "강습", "상륙", "도하", "증원"), 2.0),
    # 피해/손실
    (("전사", "부상", "사상", "피해", "경파", "손실", "파손"), 2.0),
    # 소요/건의 (지휘관 결심에 직결)
    (("요청", "건의", "재보급", "보급", "지원", "소요", "긴급", "상신"), 1.5),
    # 제대/부대 상태
    (("대대", "중대", "소대", "여단", "연대", "사단", "진지", "예비대", "화기중대",
      "기계화", "부대"), 1.0),
    # 적 관련
    (("적군", "적은", "적의", "적이", "적 ", "적을"), 1.0),
]
_QUANT = re.compile(r"\d+\s*(?:대|명|발|문|km|m|개|시|번|여단|대대|중대|소대)")


def importance(sentence: str) -> float:
    """Rule-based military importance of one sentence (>= 0). Higher = more."""
    w = 0.0
    for kws, wt in _IMPORTANCE_KEYWORDS:
        if any(k in sentence for k in kws):
            w += wt
    if _QUANT.search(sentence):
        w += 1.0
    return w


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for piece in _SPLIT.split(text or ""):
        s = _LEAD.sub("", piece).strip()
        if len(s) >= 4:
            out.append(s)
    return out


class KeySentenceExtractor:
    """Extract and cache a source's importance-weighted key sentences.

    Returns a list of ``(sentence, weight)`` pairs. Military-event / echelon /
    casualty / request sentences always clear the threshold and are included;
    the weight (``1 + importance``) makes the more important events dominate the
    key-sentence reward.
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self.n = config.keysent_n
        self.max_new_tokens = config.keysent_max_new_tokens
        self._cache: dict[str, list[tuple[str, float]]] = {}

    def _prompt(self, source: str) -> str:
        return (
            f"다음 보고서에서 가장 중요한 핵심 문장 {self.n}개를 원문에서 골라 "
            "한 줄에 하나씩 출력하시오.\n\n"
            f"[보고서]\n{source}\n\n[핵심 문장]\n"
        )

    def extract(self, backend: LLMBackend, source: str) -> list[tuple[str, float]]:
        """Return cached ``(sentence, weight)`` key sentences, extracting on first use."""
        if source in self._cache:
            return self._cache[source]
        cfg = self.config
        weighted: dict[str, float] = {}

        def _consider(sent: str) -> None:
            imp = importance(sent)
            if imp >= cfg.keysent_min_weight:  # military event / echelon / request
                weighted[sent] = max(weighted.get(sent, 0.0), 1.0 + imp)

        # 1) Rule-based: every military-important source sentence is a key sentence.
        if cfg.keysent_importance:
            for s in _split_sentences(source):
                _consider(s)

        # 2) Optional LLM augmentation: salient picks, weighted by the same rule.
        if cfg.keysent_use_llm:
            try:
                text = backend.generate_text(self._prompt(source), self.max_new_tokens)
                for s in _split_sentences(text)[: self.n]:
                    # LLM picks earn at least a base weight even if keyword-light.
                    weighted[s] = max(weighted.get(s, 0.0), 1.0 + importance(s))
            except NotImplementedError:
                pass  # backend can't generate -> rely on the rule-based set

        # 3) Keep the highest-importance sentences (cap keeps the reward focused).
        items = sorted(weighted.items(), key=lambda kv: -kv[1])[: cfg.keysent_max]
        result = [(s, w) for s, w in items]
        self._cache[source] = result
        return result
