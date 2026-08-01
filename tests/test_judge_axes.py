"""MultiAxisJudge: per-axis diagnostic scoring, separate from the reward path.

Parsing must be lenient (JSON first, per-axis regex fallback) because a frozen
backbone often won't honour a strict JSON contract. score_axes must cache and
degrade to None when the backend can't generate text.
"""

import pytest

from summarize_rl.config import RewardConfig
from summarize_rl.judge import (
    AXES,
    GeminiAxisJudge,
    MultiAxisJudge,
    _parse_axes_json,
)
from summarize_rl.llm_backend import MockBackend


def test_axes_are_the_four_diagnostic_axes():
    assert AXES == ("hallucination", "numeric", "entity", "omission")


def test_parse_clean_json():
    v = _parse_axes_json('{"hallucination": 90, "numeric": 80, "entity": 70, "omission": 60}')
    assert v == {"hallucination": 0.9, "numeric": 0.8, "entity": 0.7, "omission": 0.6}


def test_parse_json_with_preamble_and_trailing_text():
    v = _parse_axes_json('점수: {"hallucination": 100, "numeric": 50, "entity": 0, "omission": 25} 끝')
    assert v["hallucination"] == 1.0 and v["entity"] == 0.0 and v["omission"] == 0.25


def test_parse_regex_fallback_when_not_json():
    v = _parse_axes_json("hallucination: 90\nnumeric=80\nentity 70\nomission: 60")
    assert v == {"hallucination": 0.9, "numeric": 0.8, "entity": 0.7, "omission": 0.6}


def test_parse_partial_returns_only_available_axes():
    v = _parse_axes_json('{"hallucination": 40, "numeric": 55}')
    assert set(v) == {"hallucination", "numeric"}
    assert v["numeric"] == 0.55


def test_parse_clamps_out_of_range():
    v = _parse_axes_json('{"hallucination": 150, "numeric": -10, "entity": 70, "omission": 60}')
    assert v["hallucination"] == 1.0 and v["numeric"] == 0.0


def test_parse_garbage_returns_none():
    assert _parse_axes_json("죄송합니다. 채점할 수 없습니다.") is None
    assert _parse_axes_json("") is None


def test_score_axes_parses_and_caches():
    be = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    calls = {"n": 0}

    def fake(prompt, max_new_tokens=128):
        calls["n"] += 1
        return '{"hallucination": 80, "numeric": 90, "entity": 70, "omission": 60}'

    be.generate_text = fake
    judge = MultiAxisJudge(be, RewardConfig())
    v1 = judge.score_axes("원문", "요약")
    v2 = judge.score_axes("원문", "요약")  # cached: no second backend call
    assert v1 == v2 == {"hallucination": 0.8, "numeric": 0.9, "entity": 0.7, "omission": 0.6}
    assert calls["n"] == 1


def test_score_axes_none_when_backend_cannot_generate():
    be = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)

    def boom(prompt, max_new_tokens=128):
        raise NotImplementedError

    be.generate_text = boom
    judge = MultiAxisJudge(be, RewardConfig())
    assert judge.score_axes("원문", "요약") is None


def test_gemini_axis_judge_uses_external_call():
    seen = {}

    def call(prompt):
        seen["prompt"] = prompt
        return '{"hallucination": 100, "numeric": 100, "entity": 50, "omission": 50}'

    judge = GeminiAxisJudge(call)
    v = judge.score_axes("원문 텍스트", "요약 텍스트")
    assert v["entity"] == 0.5
    assert "원문 텍스트" in seen["prompt"] and "요약 텍스트" in seen["prompt"]
