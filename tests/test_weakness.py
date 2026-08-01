"""FailureLog: rollout-level failure accumulation with correct axis-sign handling.

The penalty axes (hallucination, copy_penalty, length_penalty) are stored as
positive magnitudes and subtracted from the reward, so a *high* value is a
failure; the content axes fail when *low*. These tests pin that asymmetry, the
JSONL round-trip, failure-rate accounting, and scenario metadata mining.
"""

import json

from summarize_rl.branches import Example, Triplet
from summarize_rl.rewards import RewardBreakdown
from summarize_rl.weakness import (
    DEFAULT_THRESHOLDS,
    PENALTY_AXES,
    FailureLog,
    FailureRecord,
    extract_meta,
    load_window,
)


def _bd(**over) -> RewardBreakdown:
    base = dict(
        faithfulness=0.9, coverage=0.9, key_sentence=0.9, contrast=1.0,
        length_penalty=0.0, copy_penalty=0.0, hallucination=0.0, judge=0.0,
        total=1.0,
    )
    base.update(over)
    return RewardBreakdown(**base)


def _ex(source="산악 지형에서 1중대 120명이 교전.", **over):
    kw = dict(id="10000", triplets=[Triplet("1중대", "규모", "120명")],
              keyfacts=["k1", "k2"], meta={"split": "train"})
    kw.update(over)
    return Example(source, **kw)


def test_penalty_axis_fails_when_above_threshold(tmp_path):
    log = FailureLog(str(tmp_path / "f.jsonl"),
                     thresholds={"hallucination": 0.3})
    log.record(5, _ex(), "요약A", _bd(hallucination=0.6))  # 0.6 > 0.3 -> fail
    log.record(5, _ex(), "요약B", _bd(hallucination=0.1))  # 0.1 < 0.3 -> ok
    assert log.failure_rates() == {"hallucination": 0.5}
    recs = load_window(str(tmp_path / "f.jsonl"), window_steps=100, now=5)
    assert len(recs) == 1 and recs[0].axis == "hallucination"


def test_good_axis_fails_when_below_threshold(tmp_path):
    log = FailureLog(str(tmp_path / "f.jsonl"), thresholds={"coverage": 0.4})
    log.record(1, _ex(), "s", _bd(coverage=0.2))  # low -> fail
    log.record(1, _ex(), "s", _bd(coverage=0.9))  # high -> ok
    assert log.failure_rates()["coverage"] == 0.5


def test_penalty_axes_membership():
    assert PENALTY_AXES == {"hallucination", "copy_penalty", "length_penalty"}
    # DEFAULT_THRESHOLDS must not carry key_sentence (disabled -> all-zero noise).
    assert "key_sentence" not in DEFAULT_THRESHOLDS


def test_reward_axis_vocabularies_are_valid_breakdown_fields():
    # The threshold / judge-mapping / synthesis-knob dicts must all key off real
    # RewardBreakdown fields, so a renamed axis can't silently drop out of the loop.
    from dataclasses import fields
    from summarize_rl.rewards import RewardBreakdown
    from summarize_rl.weakness import PENALTY_AXES, _REWARD_TO_JUDGE
    from data.gen_weakness_scenarios import AXIS_KNOBS

    valid = {f.name for f in fields(RewardBreakdown)}
    assert set(DEFAULT_THRESHOLDS) <= valid
    assert set(_REWARD_TO_JUDGE) <= valid
    assert set(AXIS_KNOBS) <= valid
    assert PENALTY_AXES <= valid


def test_jsonl_roundtrip_preserves_fields(tmp_path):
    p = str(tmp_path / "f.jsonl")
    log = FailureLog(p, thresholds={"coverage": 0.4})
    log.record(7, _ex(), "실패한 요약 텍스트", _bd(coverage=0.1))
    recs = load_window(p, window_steps=100, now=7)
    assert len(recs) == 1
    r = recs[0]
    assert isinstance(r, FailureRecord)
    assert r.step == 7 and r.example_id == "10000" and r.axis == "coverage"
    assert r.score == 0.1 and r.summary == "실패한 요약 텍스트"
    assert r.meta["n_keyfacts"] == 2


def test_load_window_filters_by_step(tmp_path):
    p = str(tmp_path / "f.jsonl")
    log = FailureLog(p, thresholds={"coverage": 0.4})
    log.record(1, _ex(), "old", _bd(coverage=0.1))
    log.record(50, _ex(), "new", _bd(coverage=0.1))
    recs = load_window(p, window_steps=10, now=50)  # keep step >= 40
    assert [r.step for r in recs] == [50]


def test_default_thresholds_adds_key_sentence_only_when_enabled():
    from summarize_rl.config import RewardConfig
    from summarize_rl.weakness import default_thresholds

    off = default_thresholds(RewardConfig(w_keysent=0.0))
    assert "key_sentence" not in off

    on = default_thresholds(RewardConfig(w_keysent=0.3))
    assert on["key_sentence"] == 0.4
    # base axes still present and unchanged
    assert on["coverage"] == DEFAULT_THRESHOLDS["coverage"]


def test_extract_meta_counts_numbers_units_terrain():
    ex = _ex(source="산악 지형. 1중대 120명, 예비대 1,200명. 전차 4대 투입.")
    m = extract_meta(ex)
    assert m["numeric_tokens"] >= 3     # 120명, 1,200명, 4대
    assert m["n_units"] == 1            # one unique triplet head
    assert m["n_keyfacts"] == 2
    assert m["source_len"] == len(ex.source)
    assert "산악" in m["terrain"]
