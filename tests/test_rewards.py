import math

from summarize_rl.branches import Triplet
from summarize_rl.config import RewardConfig
from summarize_rl.rewards import (
    LexicalFaithfulness,
    compute_reward,
    extractive_copy,
    length_penalty,
    normalize_rewards,
    term_usage,
    triplet_coverage,
)


def test_lexical_faithfulness_full_and_zero():
    fm = LexicalFaithfulness()
    assert fm.score("적 부대 이동", "적 부대가 이동 중이다") == 1.0
    assert fm.score("완전히 무관한 단어들", "적 부대 이동") < 1.0
    assert fm.score("", "무언가") == 0.0


def test_triplet_coverage():
    triplets = [Triplet("1소대", "위치", "고지"), Triplet("적", "행동", "이동")]
    # summary mentions 3 of 4 entities (1소대, 고지, 이동) but not "적"
    cov = triplet_coverage("1소대가 고지로 이동했다", triplets)
    assert math.isclose(cov, 3 / 4, rel_tol=1e-6)


def test_triplet_coverage_empty_is_one():
    assert triplet_coverage("아무말", []) == 1.0


def test_term_usage():
    assert term_usage("기동과 정찰을 실시", ["기동", "정찰", "화력지원"]) == 2 / 3
    assert term_usage("아무말", []) == 1.0


def test_length_penalty_overlength():
    cfg = RewardConfig(target_length=10)
    # 20 tokens, all unique -> penalty ~ (20-10)/10 = 1.0
    text = " ".join(f"w{i}" for i in range(20))
    pen = length_penalty(20, cfg, text)
    assert pen >= 1.0


def test_length_penalty_repetition():
    cfg = RewardConfig(target_length=100, repeat_ngram=1)
    text = "spam spam spam spam"
    pen = length_penalty(4, cfg, text)
    # all unigrams identical -> repetition = 1 - 1/4 = 0.75
    assert math.isclose(pen, 0.75, rel_tol=1e-6)


def test_compute_reward_weighted_sum():
    cfg = RewardConfig(
        w_faithfulness=1.0, w_coverage=1.0, w_term=0.5, w_length=0.2, target_length=100
    )
    triplets = [Triplet("적", "행동", "이동")]
    r = compute_reward(
        summary="적이 이동했다",
        source="적이 이동했다",
        triplets=triplets,
        active_terms=["기동"],
        config=cfg,
    )
    # faithfulness=1, coverage=1 (적, 이동 present), term=0 (기동 absent)
    assert math.isclose(r.faithfulness, 1.0, rel_tol=1e-6)
    assert math.isclose(r.coverage, 1.0, rel_tol=1e-6)
    assert math.isclose(r.term_usage, 0.0, abs_tol=1e-9)
    expected = 1.0 * 1.0 + 1.0 * 1.0 + 0.5 * 0.0 - 0.2 * r.length_penalty
    assert math.isclose(r.total, expected, rel_tol=1e-6)


def test_extractive_copy_detects_verbatim():
    src = "적 부대가 능선을 따라 이동하였고 이후 고지를 장악하였다"
    # verbatim span -> high copy fraction
    assert extractive_copy("적 부대가 능선을 따라 이동하였고", src, n=4) == 1.0
    # recombined / novel wording -> low copy
    assert extractive_copy("부대가 고지를 장악", src, n=4) < 1.0
    # too short for the n-gram -> no copy signal
    assert extractive_copy("적 부대", src, n=4) == 0.0


def test_term_usage_only_credits_non_source_terms():
    # "기동" is NOT in the source -> standardization worth crediting
    assert term_usage("기동을 실시", ["기동"], source="적이 이동했다") == 1.0
    # standard term already verbatim in source -> nothing to add -> vacuous 1.0
    assert term_usage("무관", ["점령"], source="적이 점령했다") == 1.0
    # needed but absent from summary -> 0
    assert term_usage("적이 움직였다", ["기동"], source="적이 이동했다") == 0.0


def test_copy_penalty_lowers_reward_for_verbatim():
    cfg = RewardConfig(w_copy=1.0, w_faithfulness=1.0, w_coverage=0.0, w_term=0.0)
    src = "적 부대가 능선을 따라 이동하였고 이후 고지를 장악하였다"
    verbatim = compute_reward(src, src, [], [], cfg)  # summary == source
    novel = compute_reward("부대가 고지를 장악", src, [], [], cfg)
    assert verbatim.copy_penalty > novel.copy_penalty
    # copying is no longer a free lunch: its total is dragged down by the penalty
    assert verbatim.total < verbatim.faithfulness * cfg.w_faithfulness


def test_contrast_term_enters_total():
    cfg = RewardConfig(
        w_faithfulness=1.0, w_coverage=1.0, w_term=0.5, w_contrast=0.5, w_length=0.2
    )
    triplets = [Triplet("적", "행동", "이동")]
    kwargs = dict(
        summary="적이 이동했다", source="적이 이동했다", triplets=triplets,
        active_terms=["기동"], config=cfg,
    )
    base = compute_reward(contrast=0.0, **kwargs)
    withc = compute_reward(contrast=2.0, **kwargs)
    assert math.isclose(withc.contrast, 2.0, rel_tol=1e-9)
    # total gains exactly w_contrast * contrast
    assert math.isclose(withc.total - base.total, 0.5 * 2.0, rel_tol=1e-6)


def test_pluggable_faithfulness_model():
    class Always:
        def __init__(self, v):
            self.v = v

        def score(self, summary, source):
            return self.v

    cfg = RewardConfig()
    r = compute_reward("s", "src", [], [], cfg, faithfulness_model=Always(0.42))
    assert math.isclose(r.faithfulness, 0.42, rel_tol=1e-6)


def test_normalize_rewards():
    out = normalize_rewards([1.0, 2.0, 3.0])
    assert math.isclose(sum(out), 0.0, abs_tol=1e-6)
    # symmetric around zero
    assert out[0] < 0 < out[2]


def test_normalize_rewards_constant():
    # zero variance -> all zeros (no NaN)
    out = normalize_rewards([5.0, 5.0, 5.0])
    assert all(abs(x) < 1e-6 for x in out)
