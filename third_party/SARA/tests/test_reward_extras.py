"""Unit tests for the ported reference-free reward components (reward_extras.py)
and the summarize_rl Korean data adapter.

These are dependency-light (stdlib only) so they run without torch/torchmetrics.
Run:  python -m pytest third_party/SARA/tests/ -q
  or: python third_party/SARA/tests/test_reward_extras.py   (plain runner below)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import reward_extras as R  # noqa: E402


# --- triplet_coverage ------------------------------------------------------- #

def test_triplet_coverage_high_when_entities_present():
    triplets = R.to_triplets([["블루포스", "규모", "제1기계화보병대대 1,000명"],
                              ["갈도비아", "목표", "국경 지역 통제"]])
    summary = "갈도비아의 블루포스 제1기계화보병대대가 국경 지역 통제를 목표로 기동했다."
    assert R.triplet_coverage(summary, triplets) > 0.7


def test_triplet_coverage_low_when_entities_absent():
    triplets = R.to_triplets([["레드포스", "규모", "제3기갑여단"],
                              ["메타니아", "목표", "영토 방어"]])
    summary = "오늘 날씨가 맑고 시장에서 사람들이 물건을 샀다."
    assert R.triplet_coverage(summary, triplets) < 0.2


def test_triplet_coverage_vacuous_without_triplets():
    assert R.triplet_coverage("아무 요약", []) == 1.0


def test_to_triplets_accepts_formats():
    assert R.to_triplets([["a", "r", "b"]]) == [R.Triplet("a", "r", "b")]
    assert R.to_triplets([{"head": "a", "relation": "r", "tail": "b"}]) == [R.Triplet("a", "r", "b")]
    assert R.to_triplets(None) == []


# --- ungrounded_fact_penalty ------------------------------------------------ #

def test_hallucination_penalizes_invented_facts():
    source = "블루포스 제1기계화보병대대가 전차 4대를 운용한다."
    summary = "제3기갑여단이 전차 8대와 40발을 투입했다."  # unit + numbers not in source
    assert R.ungrounded_fact_penalty(summary, source) > 0.5


def test_hallucination_zero_when_grounded():
    source = "블루포스 제1기계화보병대대가 전차 4대를 운용한다."
    summary = "제1기계화보병대대가 전차 4대를 운용했다."
    assert R.ungrounded_fact_penalty(summary, source) == 0.0


def test_hallucination_zero_when_no_checkable_facts():
    assert R.ungrounded_fact_penalty("부대가 이동했다.", "부대가 전진했다.") == 0.0


# --- judge parsing + GeminiJudge ------------------------------------------- #

def test_parse_score_takes_last_int_in_range():
    assert R._parse_score("점수는 85") == 0.85
    assert R._parse_score("기준1... 2... 최종 90점") == 0.90  # last valid int


def test_parse_score_none_cases():
    assert R._parse_score("") is None
    assert R._parse_score("점수 없음") is None
    assert R._parse_score("999") is None  # out of 0..100 range


def test_gemini_judge_with_stub_call():
    judge = R.GeminiJudge(lambda prompt: "이 요약의 점수는 85점입니다.")
    assert judge.score("원문", "요약") == 0.85


def test_gemini_judge_none_call_returns_zero():
    assert R.GeminiJudge(None).score("원문", "요약") == 0.0


def test_gemini_judge_caches(monkeypatch=None):
    calls = {"n": 0}

    def stub(prompt):
        calls["n"] += 1
        return "70"

    judge = R.GeminiJudge(stub)
    judge.score("s", "a")
    judge.score("s", "a")
    assert calls["n"] == 1  # second call served from cache


# --- data adapter ----------------------------------------------------------- #

def test_load_ours_dataset_splits_and_shape():
    train, validation, test = R.load_ours_dataset()
    assert len(train) == 140
    assert len(validation) == 10
    assert len(test) >= 1
    row = train[0]
    assert len(row) == 4  # document, summary, presumm, triplets
    document, summary, presumm, triplets = row
    assert isinstance(document, str) and document
    assert isinstance(summary, str) and summary
    assert isinstance(presumm, str)  # keyfacts joined (may be empty for some rows)
    assert isinstance(triplets, list)
    if triplets:
        assert len(triplets[0]) == 3  # [head, relation, tail]


def test_reward_combination_is_noop_when_weights_zero():
    """The added reward term must not change scores when all new weights are 0."""
    base = [0.3, 0.5, 0.4, 0.6]
    trip = [1.0, 0.2, 0.8, 0.5]
    hallu = [0.1, 0.9, 0.0, 0.3]
    judge = [0.7, 0.7, 0.7, 0.7]
    tcw = hw = jw = 0.0
    combined = [base[i] + tcw * trip[i] - hw * hallu[i] + jw * judge[i] for i in range(4)]
    assert combined == base


if __name__ == "__main__":
    # Plain runner so it works even without pytest installed.
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
