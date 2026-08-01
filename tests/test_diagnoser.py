"""WeaknessDiagnoser: aggregate failures, triage with a judge, confirm weak axes.

Axis vocabulary is the reward-breakdown field names (what FailureRecord.axis
holds). The three confirmation gates (absolute rate, consecutive rounds, judge
triage) and the RAW-ceiling reclassification are pinned here with synthetic
records and a stub judge — no real backbone needed.
"""

import pytest

from summarize_rl.weakness import (
    DiagnosisReport,
    FailureRecord,
    HeldoutReport,
    WeaknessDiagnoser,
    axis_failure_rates,
    failure_meta_profile,
    triage_with_judge,
)


def _rec(axis, ex="1", numeric=2, summary="요약"):
    return FailureRecord(
        step=10, example_id=ex, axis=axis, score=0.1, summary=summary,
        meta={"numeric_tokens": numeric, "n_units": 2, "n_keyfacts": 2,
              "source_len": 100, "terrain": ["산악"]},
    )


def test_axis_failure_rates_normalizes_by_total_rollouts():
    recs = [_rec("coverage"), _rec("coverage"), _rec("hallucination")]
    rates = axis_failure_rates(recs, total_rollouts=10)
    assert rates["coverage"] == 0.2
    assert rates["hallucination"] == 0.1


def test_meta_profile_severity_scales_with_numeric_density():
    # coverage failures are numeric-dense (10), hallucination failures sparse (1)
    recs = [_rec("coverage", numeric=10) for _ in range(3)] + \
           [_rec("hallucination", numeric=1) for _ in range(3)]
    prof = failure_meta_profile(recs)
    assert prof.severity("coverage") > prof.severity("hallucination")
    assert prof.severity("coverage") >= 1.0


def test_triage_confirms_when_judge_agrees():
    # coverage maps to the judge's 'omission' axis. Low omission score => confirmed.
    recs = [_rec("coverage", ex=str(i)) for i in range(4)]
    lookup = {str(i): f"원문 {i}" for i in range(4)}

    class StubJudge:  # judge scores omission low (bad) => confirms the failure
        def score_axes(self, source, summary):
            return {"hallucination": 0.9, "numeric": 0.9, "entity": 0.9, "omission": 0.1}

    conf = triage_with_judge(recs, lookup, StubJudge(), k_per_axis=20)
    assert conf["coverage"] == 1.0  # all sampled failures confirmed


def test_triage_rejects_when_judge_disagrees():
    recs = [_rec("coverage", ex=str(i)) for i in range(4)]
    lookup = {str(i): f"원문 {i}" for i in range(4)}

    class StubJudge:  # judge says omission is fine (high) => lexical false alarm
        def score_axes(self, source, summary):
            return {"omission": 0.95, "hallucination": 0.9, "numeric": 0.9, "entity": 0.9}

    conf = triage_with_judge(recs, lookup, StubJudge(), k_per_axis=20)
    assert conf["coverage"] == 0.0


def _diag(min_rate=0.15, consecutive=2, min_confirm=0.5):
    return WeaknessDiagnoser(min_rate=min_rate, consecutive=consecutive,
                             min_confirm=min_confirm)


def test_consecutive_rule_requires_two_rounds():
    d = _diag()
    recs = [_rec("coverage") for _ in range(3)]  # rate 0.3 > 0.15
    # Round 1: candidate but no history -> not yet confirmed weak.
    r1 = d.diagnose(recs, total_rollouts=10, history=[])
    assert "coverage" in r1.candidates
    assert "coverage" not in r1.weak_axes
    # Round 2: candidate again, previous round also had it -> weak.
    r2 = d.diagnose(recs, total_rollouts=10, history=[r1])
    assert "coverage" in r2.weak_axes


def test_below_min_rate_is_not_weak():
    d = _diag()
    recs = [_rec("coverage")]  # rate 0.1 < 0.15
    r1 = d.diagnose(recs, total_rollouts=10, history=[])
    prev = DiagnosisReport(weak_axes=[], backbone_limited=[], rates={"coverage": 0.2},
                           meta_stats=r1.meta_stats, confirmed={}, candidates=["coverage"])
    r2 = d.diagnose(recs, total_rollouts=10, history=[prev])
    assert "coverage" not in r2.weak_axes


def test_judge_triage_gate_blocks_unconfirmed_axis():
    d = _diag()
    recs = [_rec("coverage", ex=str(i)) for i in range(3)]
    lookup = {str(i): f"원문 {i}" for i in range(3)}
    prev = DiagnosisReport(weak_axes=[], backbone_limited=[], rates={"coverage": 0.3},
                           meta_stats=failure_meta_profile(recs), confirmed={},
                           candidates=["coverage"])

    class RejectJudge:
        def score_axes(self, source, summary):
            return {"omission": 0.95, "hallucination": 0.9, "numeric": 0.9, "entity": 0.9}

    r = d.diagnose(recs, total_rollouts=10, source_lookup=lookup,
                   multi_judge=RejectJudge(), history=[prev])
    assert "coverage" not in r.weak_axes  # judge confirm rate 0 < min_confirm


def test_raw_ceiling_moves_axis_to_backbone_limited():
    d = _diag()
    recs = [_rec("hallucination") for _ in range(3)]  # rate 0.3
    prev = DiagnosisReport(weak_axes=[], backbone_limited=[], rates={"hallucination": 0.3},
                           meta_stats=failure_meta_profile(recs), confirmed={},
                           candidates=["hallucination"])
    # RAW base also fails hallucination at 0.4 -> backbone ceiling, not a synth target.
    r = d.diagnose(recs, total_rollouts=10, history=[prev],
                   raw_axis_rates={"hallucination": 0.4})
    assert "hallucination" in r.backbone_limited
    assert "hallucination" not in r.weak_axes


def test_converged_when_all_rates_below_min():
    d = _diag()
    recs = [_rec("coverage")]  # 0.1
    r = d.diagnose(recs, total_rollouts=10, history=[])
    assert r.converged(min_rate=0.15) is True


def test_breakdown_failures_flags_correct_axes():
    from summarize_rl.rewards import RewardBreakdown
    from summarize_rl.weakness import breakdown_failures

    bd = RewardBreakdown(faithfulness=0.9, coverage=0.2, key_sentence=0.9, contrast=1.0,
                         length_penalty=0.0, copy_penalty=0.0, hallucination=0.6,
                         judge=0.0, total=1.0)
    fails = breakdown_failures(bd, {"coverage": 0.4, "hallucination": 0.3, "faithfulness": 0.5})
    assert fails == {"coverage", "hallucination"}  # low coverage, high hallucination


def test_eval_heldout_averages_judge_axes():
    from summarize_rl.branches import Example
    from summarize_rl.weakness import eval_heldout

    class FakeSummarizer:
        def summarize(self, source, *, query=None, triplets=None):
            class R:
                text = "요약"
            return R()

    class StubJudge:
        def score_axes(self, source, summary):
            return {"hallucination": 0.8, "numeric": 0.6, "entity": 0.4, "omission": 0.2}

    examples = [Example(f"원문{i}", []) for i in range(3)]
    report = eval_heldout(FakeSummarizer(), examples, StubJudge())
    assert report.axis_scores["hallucination"] == pytest.approx(0.8)
    assert report.axis_scores["omission"] == pytest.approx(0.2)


def test_heldout_regression_detects_axis_drop():
    prev = HeldoutReport(axis_scores={"omission": 0.80, "entity": 0.70})
    now = HeldoutReport(axis_scores={"omission": 0.60, "entity": 0.72})  # omission -0.20
    assert now.regressed(prev, margin=0.05) is True
    assert now.regressed(HeldoutReport(axis_scores={"omission": 0.62, "entity": 0.70}),
                         margin=0.05) is False
