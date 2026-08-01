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

import json
import re
from typing import Protocol

from .config import RewardConfig
from .llm_backend import LLMBackend


class JudgeModel(Protocol):
    """Score how faithfully `summary` reports `source`, in [0, 1] (or None).

    ``key_sentences`` (optional) are the source's important sentences; when given
    the judge should score whether the summary reflects them *semantically*
    (paraphrase counts), complementing the cheap lexical key-sentence reward.
    """

    def score(
        self, source: str, summary: str, key_sentences: object | None = None
    ) -> float | None: ...


_SCORE_RE = re.compile(r"\d{1,3}")


def _keysent_texts(key_sentences: object | None) -> list[str]:
    """Normalize key sentences (list[str] or list[(str, weight)]) to text strings."""
    if not key_sentences:
        return []
    out: list[str] = []
    for item in key_sentences:  # type: ignore[union-attr]
        sent = item[0] if isinstance(item, (tuple, list)) else item
        if isinstance(sent, str) and sent.strip():
            out.append(sent.strip())
    return out


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
        self._cmp_cache: dict[tuple[str, str, str], float | None] = {}

    def _prompt(self, source: str, summary: str, key_sentences: object | None = None) -> str:
        ks_block = ""
        crit3 = "(3) 각 제대의 상황과 핵심 조치·건의를 빠짐없이 담았는가."
        sents = _keysent_texts(key_sentences)
        if sents:
            listed = "\n".join(f"- {s}" for s in sents)
            ks_block = f"[핵심 문장]\n{listed}\n\n"
            crit3 = "(3) 아래 [핵심 문장]들의 내용을 (표현이 달라도) 의미상 빠짐없이 담았는가."
        return (
            "당신은 군사 보고서 요약을 채점하는 심사관이다. 아래 [요약]이 [원문]을 "
            "얼마나 정확히 반영했는지 0~100 정수 하나로만 평가하라.\n"
            "채점 기준: (1) 부대·수치·지명·시간을 정확히 반영했는가, "
            "(2) 원문에 없는 부대·사건·숫자를 지어내지 않았는가, "
            f"{crit3}\n"
            "정확할수록 100, 지어내거나 핵심을 빠뜨릴수록 0. 숫자만 출력하라.\n\n"
            f"[원문]\n{source}\n\n{ks_block}[요약]\n{summary}\n\n[점수(0-100)]\n"
        )

    def score(
        self, source: str, summary: str, key_sentences: object | None = None
    ) -> float | None:
        key = (source, summary)
        if key in self._cache:
            return self._cache[key]
        try:
            text = self.backend.generate_text(
                self._prompt(source, summary, key_sentences), self.max_new_tokens
            )
        except NotImplementedError:
            self._cache[key] = None
            return None
        val = _parse_score(text)
        self._cache[key] = val
        return val

    # -- comparative (pairwise vs. a reference) scoring ---------------------
    def _compare_prompt(self, source: str, a: str, b: str) -> str:
        return (
            "당신은 군사 보고서 요약을 채점하는 심사관이다. 아래 [원문]에 대한 두 요약 "
            "[요약 A]와 [요약 B] 중 어느 것이 원문을 더 잘 요약했는지 고르라.\n"
            "기준: (1) 부대·수치·지명·사상자를 정확히 반영했는가, (2) 원문에 없는 부대·"
            "사건·숫자를 지어내지 않았는가, (3) 핵심 상황과 조치·건의를 담았는가, "
            "(4) 군더더기 없이 간결한가.\n"
            "더 나은 쪽 문자 하나(A 또는 B)만 출력하라. 우열을 가리기 어려우면 T를 출력하라. "
            "다른 말은 하지 마라.\n\n"
            f"[원문]\n{source}\n\n[요약 A]\n{a}\n\n[요약 B]\n{b}\n\n[더 나은 요약]\n"
        )

    def compare(
        self, source: str, candidate: str, reference: str
    ) -> float | None:
        """Pairwise: is `candidate` a better summary than `reference`?

        Returns 1.0 if the judge prefers the candidate, 0.0 if it prefers the
        reference, 0.5 for a tie, or None if unparseable. The reference is the
        RAW frozen-LLM summary, so this directly measures "did the policy beat
        the base model" -- a discriminative per-rollout signal (unlike the
        near-constant absolute score). Presentation order is de-biased by a
        stable hash of the candidate so position bias averages out across
        rollouts while staying reproducible. Cached per (source, cand, ref).
        """
        key = (source, candidate, reference)
        if key in self._cmp_cache:
            return self._cmp_cache[key]
        cand_first = sum(map(ord, candidate)) % 2 == 0  # stable, de-biases order
        a, b = (candidate, reference) if cand_first else (reference, candidate)
        try:
            text = self.backend.generate_text(
                self._compare_prompt(source, a, b), self.max_new_tokens
            )
        except NotImplementedError:
            self._cmp_cache[key] = None
            return None
        val = _parse_winner(text, cand_first)
        self._cmp_cache[key] = val
        return val


_WINNER_RE = re.compile(r"[ABT]")


def _parse_winner(text: str | None, cand_first: bool) -> float | None:
    """Map the judge's A/B/T verdict to the candidate's win score.

    A/B is the first A/B/T letter in the output; it is translated back through
    the (possibly swapped) presentation order so the result is always from the
    candidate's perspective: 1.0 win, 0.0 loss, 0.5 tie, None if no verdict.
    """
    m = _WINNER_RE.search((text or "").upper())
    if not m:
        return None
    letter = m.group()
    if letter == "T":
        return 0.5
    cand_letter = "A" if cand_first else "B"
    return 1.0 if letter == cand_letter else 0.0


def _parse_score(text: str | None) -> float | None:
    """Pull a 0-100 score out of the judge's text -> [0,1], or None if absent.

    Takes the LAST 0-100 integer in the output (the score usually follows any
    preamble or an echoed "0-100"), so a chatty judge that says "…점수는 85"
    still parses, and an echoed range like "0~100" does not hijack the value.
    """
    nums = [int(x) for x in _SCORE_RE.findall(text or "")]
    nums = [n for n in nums if 0 <= n <= 100]
    if not nums:
        return None
    return max(0.0, min(1.0, nums[-1] / 100.0))


# ---------------------------------------------------------------------------
# Multi-axis diagnostic judge (Phase 1): used only at diagnosis boundaries to
# re-score failure candidates on distinct error axes, kept OFF the per-rollout
# reward path so training cost is unchanged.
# ---------------------------------------------------------------------------

AXES = ("hallucination", "numeric", "entity", "omission")
# hallucination: inventing units/events/places absent from the source
# numeric:       wrong troop counts / equipment / times (rule-based has priority;
#                the judge covers unit-conversion / "약 천여 명" style variants)
# entity:        entity confusion (소대 vs 소총중대, swapping who took casualties)
# omission:      dropping key situations / actions / recommendations


class AxisJudge(Protocol):
    """Score a summary on the four diagnostic AXES, each in [0, 1] (or None)."""

    def score_axes(self, source: str, summary: str) -> dict[str, float] | None: ...


def _axes_prompt(source: str, summary: str) -> str:
    return (
        "당신은 군사 보고서 요약을 채점하는 심사관이다. [요약]을 [원문]과 대조해 "
        "아래 4개 항목을 각각 0~100 정수로 채점하고, JSON 하나만 출력하라.\n"
        '{"hallucination": _, "numeric": _, "entity": _, "omission": _}\n'
        "- hallucination: 원문에 없는 부대·사건·장소를 지어내지 않았는가 (지어냈으면 0)\n"
        "- numeric: 병력 수·장비 대수·시각 등 수치가 정확한가\n"
        "- entity: 부대 명칭·피해 주체 등 개체를 혼동하지 않았는가\n"
        "- omission: 핵심 상황·조치·건의를 빠뜨리지 않았는가\n\n"
        f"[원문]\n{source}\n\n[요약]\n{summary}\n\n[JSON]\n"
    )


_AXIS_RE = {
    # Tolerate colon, equals, or bare whitespace as the axis→score separator
    # (e.g. "entity 70", 'entity": 70', "entity=70").
    axis: re.compile(rf'"?{axis}"?[\s:="]*(-?\d{{1,3}})') for axis in AXES
}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v / 100.0))


def _parse_axes_json(text: str | None) -> dict[str, float] | None:
    """Lenient parse of a per-axis score blob into {axis: [0,1]}.

    Tries a JSON object first (tolerating preamble/trailing text by extracting the
    first ``{...}``), then falls back to a per-axis regex for backbones that won't
    emit clean JSON. Values are clamped to [0, 100] then scaled to [0, 1]. Returns
    only the axes it could read, or None if none parsed.
    """
    if not text:
        return None
    out: dict[str, float] = {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict):
            for axis in AXES:
                if axis in obj:
                    try:
                        out[axis] = _clamp01(float(obj[axis]))
                    except (ValueError, TypeError):
                        pass
    # Regex fallback for any axis JSON didn't yield.
    for axis in AXES:
        if axis not in out:
            rm = _AXIS_RE[axis].search(text)
            if rm:
                out[axis] = _clamp01(float(int(rm.group(1))))
    return out or None


class MultiAxisJudge:
    """Per-axis diagnostic scorer backed by the local frozen backbone.

    Single JSON call per (source, summary) by default; parsing degrades to a
    per-axis regex when the model won't honour JSON. The ``numeric`` axis is a
    *secondary* signal — rewards.ungrounded_fact_penalty already catches literal
    numeric hallucination at rollout time; this judge covers only what lexical
    matching misses (unit conversions, approximate phrasings). Returns None when
    the backend cannot generate text. Cached per (source, summary).
    """

    def __init__(self, backend: LLMBackend, config: RewardConfig):
        self.backend = backend
        self.config = config
        self.max_new_tokens = 128  # room for the JSON object
        self._cache: dict[tuple[str, str], dict[str, float] | None] = {}

    def score_axes(self, source: str, summary: str) -> dict[str, float] | None:
        key = (source, summary)
        if key in self._cache:
            return self._cache[key]
        try:
            text = self.backend.generate_text(
                _axes_prompt(source, summary), self.max_new_tokens
            )
        except NotImplementedError:
            self._cache[key] = None
            return None
        val = _parse_axes_json(text)
        self._cache[key] = val
        return val


class GeminiAxisJudge:
    """AxisJudge over an external text-completion callable (e.g. make_gemini).

    Injectable at diagnosis time so held-out scoring can use a stronger external
    judge instead of the backbone judging its own output. ``call(prompt) -> str``
    matches examples/eval_gemini.make_gemini's adapter. Cached per (source, summary).
    """

    def __init__(self, call, max_new_tokens: int = 128):
        self.call = call
        self.max_new_tokens = max_new_tokens
        self._cache: dict[tuple[str, str], dict[str, float] | None] = {}

    def score_axes(self, source: str, summary: str) -> dict[str, float] | None:
        key = (source, summary)
        if key in self._cache:
            return self._cache[key]
        try:
            text = self.call(_axes_prompt(source, summary))
        except Exception:
            self._cache[key] = None
            return None
        val = _parse_axes_json(text)
        self._cache[key] = val
        return val
