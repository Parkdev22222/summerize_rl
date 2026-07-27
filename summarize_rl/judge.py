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
* :class:`GeminiJudge` — the external Gemini API as the judge. Removes the
  self-referential setup (the local model grading its own outputs) and is a
  stronger evaluator, but calls the API once per rollout, so training is slow /
  costs money / is rate-limited. Prefer a fast model and modest ``num_samples``.
* Your own object with ``score(source, summary) -> float | None`` (and optionally
  ``compare(source, candidate, reference)``) — any custom evaluator.

The rubric deliberately scores *accuracy and non-fabrication*, not fluency, so a
grammatical-but-wrong summary does not win points.
"""

from __future__ import annotations

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


def _cand_first(candidate: str) -> bool:
    """Whether the candidate is shown first (slot A); stable per-candidate de-bias."""
    return sum(map(ord, candidate)) % 2 == 0


def _absolute_prompt(source: str, summary: str, key_sentences: object | None = None) -> str:
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


def _comparative_prompt(source: str, a: str, b: str) -> str:
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
        return _absolute_prompt(source, summary, key_sentences)

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
        cand_first = _cand_first(candidate)  # stable, de-biases order
        a, b = (candidate, reference) if cand_first else (reference, candidate)
        try:
            text = self.backend.generate_text(
                _comparative_prompt(source, a, b), self.max_new_tokens
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


# -- Gemini judge (external API; independent of the local backbone) -----------
def _make_gemini_call(api_key: str, model_name: str, temperature: float = 0.0):
    """Return call(prompt)->str using whichever google SDK is installed."""
    try:
        from google import genai  # new SDK: pip install google-genai

        client = genai.Client(api_key=api_key)

        def call(prompt: str) -> str:
            resp = client.models.generate_content(
                model=model_name, contents=prompt,
                config={"temperature": temperature},
            )
            return resp.text or ""

        return call
    except ImportError:
        pass
    import google.generativeai as genai  # old SDK: pip install google-generativeai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    def call(prompt: str) -> str:
        return model.generate_content(
            prompt, generation_config={"temperature": temperature}
        ).text or ""

    return call


class GeminiJudge:
    """LLM judge backed by the Gemini API instead of the local backbone.

    Same ``score`` / ``compare`` interface as :class:`BackboneJudge`, but the
    verdict comes from Gemini. This removes the self-referential setup (the local
    model judging its own outputs) and uses a stronger evaluator, at the cost of
    a network call per rollout. Intended for the comparative "beat RAW" reward.

    WARNING: RL training calls the judge once per rollout -- potentially
    thousands of Gemini calls, subject to rate limits and cost. Use a fast model
    (gemini-2.5-flash / -flash-lite), keep num_samples modest, and set ``sleep``
    to stay under the rate limit. Verdicts are cached per (source, cand, ref).

    ``call`` may be injected (a function prompt->str) for testing without the SDK.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        retries: int = 4,
        sleep: float = 0.0,
        call=None,
    ):
        import os

        self.model_name = model_name
        self.temperature = temperature
        self.retries = retries
        self.sleep = sleep
        if call is not None:
            self._call = call
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise SystemExit(
                    "GeminiJudge: 환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY)를 설정하세요."
                )
            self._call = _make_gemini_call(key, model_name, temperature)
        self._cache: dict[tuple[str, str], float | None] = {}
        self._cmp_cache: dict[tuple[str, str, str], float | None] = {}

    def _generate(self, prompt: str) -> str | None:
        import time

        for attempt in range(self.retries):
            try:
                text = self._call(prompt) or ""
                if self.sleep:
                    time.sleep(self.sleep)
                return text
            except Exception as e:  # transient API/rate errors -> backoff and retry
                if attempt == self.retries - 1:
                    print(f"   [GeminiJudge 오류] {e}")
                    return None
                time.sleep(2 * (attempt + 1))
        return None

    def score(
        self, source: str, summary: str, key_sentences: object | None = None
    ) -> float | None:
        key = (source, summary)
        if key in self._cache:
            return self._cache[key]
        val = _parse_score(self._generate(_absolute_prompt(source, summary, key_sentences)))
        self._cache[key] = val
        return val

    def compare(self, source: str, candidate: str, reference: str) -> float | None:
        key = (source, candidate, reference)
        if key in self._cmp_cache:
            return self._cmp_cache[key]
        cand_first = _cand_first(candidate)
        a, b = (candidate, reference) if cand_first else (reference, candidate)
        val = _parse_winner(self._generate(_comparative_prompt(source, a, b)), cand_first)
        self._cmp_cache[key] = val
        return val
