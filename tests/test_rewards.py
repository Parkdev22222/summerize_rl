import math

from summarize_rl.branches import Triplet
from summarize_rl.config import RewardConfig
from summarize_rl.rewards import (
    LexicalFaithfulness,
    compute_reward,
    extractive_copy,
    key_sentence_coverage,
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


def test_faithfulness_drops_single_char_noise():
    fm = LexicalFaithfulness()
    # 1-char tokens are ignored; only content tokens (>=2 chars) count.
    # "고지" is absent from the source, so grounding is 0 (not diluted to ~1 by
    # trivially-present single characters).
    assert fm.score("고지 점령", "적 부대가 이동 중이다") == 0.0


def test_reward_prefers_on_topic_over_generic():
    # The reward must rank a summary about THIS source above a fluent but
    # off-topic one (the training->inference "military but wrong content" bug).
    cfg = RewardConfig()  # defaults: coverage up-weighted, term down-weighted
    triplets = [Triplet("갈도비아", "목표", "국경통제"), Triplet("블루포스", "규모", "대대")]
    source = "갈도비아 블루포스 대대가 국경통제 작전을 수행하며 정찰한다."
    on_topic = "갈도비아 블루포스 대대가 국경통제 작전을 수행한다"
    off_topic = "아군 부대가 고지를 점령하고 방어 진지를 구축하며 기동한다"  # generic military
    r_on = compute_reward(on_topic, source, triplets, ["기동", "정찰"], cfg)
    r_off = compute_reward(off_topic, source, triplets, ["기동", "정찰"], cfg)
    assert r_on.coverage > r_off.coverage
    assert r_on.total > r_off.total


def test_balance_content_gate_throttles_off_topic_fluency():
    # A fluent, on-genre but off-topic summary can still score high on the
    # gameable fluency terms (faith/term). With balance_content on, those terms
    # are gated by content (cov+keysent), so the off-topic summary loses most of
    # its fluency credit and the on-topic/off-topic total gap widens.
    triplets = [Triplet("갈도비아", "목표", "국경통제"), Triplet("블루포스", "규모", "대대")]
    source = "갈도비아 블루포스 대대가 국경통제 작전을 수행하며 정찰한다."
    on_topic = "갈도비아 블루포스 대대가 국경통제 작전을 수행한다"
    off_topic = "아군 부대가 고지를 점령하고 방어 진지를 구축하며 기동한다"
    terms = ["기동", "정찰"]

    plain = RewardConfig()
    gated = RewardConfig(balance_content=True)

    gap_plain = (compute_reward(on_topic, source, triplets, terms, plain).total
                 - compute_reward(off_topic, source, triplets, terms, plain).total)
    gap_gated = (compute_reward(on_topic, source, triplets, terms, gated).total
                 - compute_reward(off_topic, source, triplets, terms, gated).total)

    # Gating never hurts the ranking and strictly widens the separation.
    assert gap_gated > gap_plain

    # The off-topic summary's fluency credit is throttled toward the floor: its
    # faith/term contribution is scaled by the (near-zero) content gate.
    off_gated = compute_reward(off_topic, source, triplets, terms, gated)
    off_plain = compute_reward(off_topic, source, triplets, terms, plain)
    assert off_gated.total < off_plain.total


def test_balance_content_off_by_default_is_unchanged():
    # Default config must produce the exact additive reward (no gating), so the
    # flag is a pure opt-in and existing runs are bit-for-bit unaffected.
    cfg = RewardConfig()
    assert cfg.balance_content is False
    triplets = [Triplet("적", "행동", "이동")]
    bd = compute_reward("적이 이동했다", "적이 이동했다", triplets, ["기동"], cfg)
    expected = (
        cfg.w_faithfulness * bd.faithfulness
        + cfg.w_coverage * bd.coverage
        + cfg.w_term * bd.term_usage
        + cfg.w_keysent * bd.key_sentence
        + cfg.w_contrast * bd.contrast
        - cfg.w_length * bd.length_penalty
        - cfg.w_copy * bd.copy_penalty
    )
    assert math.isclose(bd.total, expected, rel_tol=1e-9)


def test_key_sentence_extractor_caches():
    from summarize_rl.config import RewardConfig
    from summarize_rl.keysent import KeySentenceExtractor
    from summarize_rl.llm_backend import MockBackend

    be = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    calls = {"n": 0}
    base = be.generate_text
    be.generate_text = lambda p, m=256: (calls.__setitem__("n", calls["n"] + 1) or base(p, m))

    ext = KeySentenceExtractor(RewardConfig(keysent_n=2))
    first = ext.extract(be, "원문 A")
    second = ext.extract(be, "원문 A")  # served from cache
    assert first == second
    assert calls["n"] == 1  # extracted once per unique source
    ext.extract(be, "원문 B")  # different source -> extracts again
    assert calls["n"] == 2


def test_key_sentence_coverage():
    keys = ["갈도비아 블루포스가 국경을 통제한다", "레드포스가 도시를 방어한다"]
    # reflects both key sentences' content (verb conjugation costs a little)
    full = "갈도비아 블루포스가 국경을 통제하고 레드포스가 도시를 방어한다"
    assert key_sentence_coverage(full, keys) > 0.8
    # reflects neither
    assert key_sentence_coverage("날씨가 맑고 새가 난다", keys) < 0.2
    # no key sentences -> vacuously complete
    assert key_sentence_coverage("아무말", []) == 1.0


def test_key_sentence_enters_total():
    cfg = RewardConfig(w_keysent=1.0)
    keys = ["갈도비아 블루포스", "레드포스 방어"]
    kw = dict(source="갈도비아 블루포스 레드포스 방어", triplets=[], active_terms=[], config=cfg)
    with_ks = compute_reward("갈도비아 블루포스 레드포스 방어", key_sentences=keys, **kw)
    without = compute_reward("갈도비아 블루포스 레드포스 방어", key_sentences=None, **kw)
    assert with_ks.key_sentence > 0.9
    assert without.key_sentence == 0.0  # None -> no contribution
    assert with_ks.total > without.total


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
