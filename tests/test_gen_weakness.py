"""gen_weakness_scenarios: weak-axis-targeted synthesis over the template engine.

Reuses data.gen_test_scenarios pools/structure but modulates generation knobs
by the diagnosed weak axes (concept-level, not surface-copying the failures).
Tests pin: numeric knob raises numeric density, every triplet stays grounded in
the source, id/split tagging, and axis->knob mapping (incl. backbone_limited
exclusion).
"""

import random

from summarize_rl.rewards import _FACT_RE
from summarize_rl.weakness import DiagnosisReport, MetaProfile

from data.gen_weakness_scenarios import (
    ID_OFFSET,
    default_params,
    make_weakness_scenario,
    synthesis_budget,
    synthesis_params_from,
    triplets_grounded,
)


def _numeric_count(text):
    return sum(1 for m in _FACT_RE.findall(text) if any(c.isdigit() for c in m))


def test_numeric_knob_increases_numeric_density():
    rng1 = random.Random(0)
    rng2 = random.Random(0)
    base = make_weakness_scenario(rng1, 0, default_params(), round_k=1)
    dense = make_weakness_scenario(
        rng2, 0, {**default_params(), "numeric_density": 2.5}, round_k=1)
    assert _numeric_count(dense["source_text"]) > _numeric_count(base["source_text"])


def test_every_triplet_is_grounded_in_source():
    rng = random.Random(3)
    params = {**default_params(), "numeric_density": 2.5, "near_miss_names": True,
              "entity_crossover": True, "extra_recommendations": 2}
    rec = make_weakness_scenario(rng, 5, params, round_k=2)
    assert triplets_grounded(rec) is True


def test_id_offset_and_round_split_tag():
    rng = random.Random(1)
    rec = make_weakness_scenario(rng, 7, default_params(), round_k=3)
    assert rec["id"] == ID_OFFSET + 7
    assert rec["split"] == "synthetic_r3"


def _report(weak, backbone=None, sev=1.5):
    ms = MetaProfile(by_axis={a: {"numeric_median": 6, "source_len_median": 100,
                                  "n_units_median": 2, "count": 3} for a in weak},
                     overall_numeric_median=4)
    return DiagnosisReport(weak_axes=list(weak), backbone_limited=list(backbone or []),
                           rates={}, meta_stats=ms, confirmed={}, candidates=list(weak))


def test_synthesis_params_from_activates_weak_axis_knobs():
    p = synthesis_params_from(_report(["hallucination"]))
    assert p["near_miss_names"] is True
    # coverage knobs not activated
    assert p["late_key_events"] is False


def test_backbone_limited_axis_gets_no_knobs():
    # hallucination only in backbone_limited (not weak_axes) -> no near_miss knob
    p = synthesis_params_from(_report([], backbone=["hallucination"]))
    assert p["near_miss_names"] is False
    assert p == default_params()


def test_synthesis_budget_scales_with_weak_axis_count():
    r2 = _report(["faithfulness", "coverage"])
    b = synthesis_budget(r2, train_size=150, ratio=0.25, min_per_axis=8)
    assert b >= 8 * 2               # at least min per axis
    assert b >= round(0.25 * 150)   # at least the ratio floor
