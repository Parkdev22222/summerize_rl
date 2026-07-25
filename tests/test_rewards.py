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
    triplets = [Triplet("1소대", "위치", "고지"), Triplet("적군", "행동", "이동")]
    # entities {1소대, 고지, 적군, 이동}: 3 fully present, 적군 absent -> 3/4
    cov = triplet_coverage("1소대가 고지로 이동했다", triplets)
    assert math.isclose(cov, 3 / 4, rel_tol=1e-6)


def test_triplet_coverage_space_insensitive():
    # KB stores the entity solid; the summary spaces it -> still grounded.
    trs = [Triplet("블루포스", "관계", "레드포스")]
    assert triplet_coverage("블루 포스와 레드 포스가 교전한다", trs) == 1.0
    assert triplet_coverage("날씨가 맑고 바람이 분다", trs) == 0.0


def test_triplet_coverage_partial_credit():
    # A long descriptive tail earns fractional credit for the tokens present,
    # instead of collapsing to 0 unless every token appears verbatim.
    trs = [Triplet("갈도비아", "목표", "국경 지역 통제")]
    # 갈도비아 present (1.0); tail tokens {국경, 지역, 통제}: 국경+통제 present,
    # 지역 absent -> 2/3.  mean(1.0, 2/3) ~= 0.833
    cov = triplet_coverage("갈도비아가 국경 통제를 시도한다", trs)
    assert 0.8 < cov < 0.86


def test_triplet_coverage_gold_beats_offtopic_with_phrase_tails():
    # Regression for the flat-zero-coverage bug: a summary that names the
    # entities must clear a generic off-topic one by a wide margin, and must NOT
    # be pinned near zero the way all-or-nothing phrase matching did.
    trs = [Triplet("갈도비아", "목표", "국경 지역 통제"),
           Triplet("블루포스", "규모", "제1기계화보병대대")]
    on = "갈도비아 블루 포스가 제1기계화보병대대로 국경 지역 통제를 수행한다"
    off = "아군 부대가 고지를 점령하고 방어 진지를 구축한다"
    assert triplet_coverage(on, trs) > 0.9
    assert triplet_coverage(off, trs) < 0.1


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


def test_ungrounded_fact_penalty_flags_invented_units_and_numbers():
    from summarize_rl.rewards import ungrounded_fact_penalty
    source = "제1기계화보병대대가 전차 2대를 파괴하였고 부상 2명이 발생함"
    # faithful: every unit/quantity is in the source -> no penalty
    assert ungrounded_fact_penalty("제1기계화보병대대가 전차 2대를 파괴", source) == 0.0
    # hallucinated: invents a unit and a quantity absent from the source
    p = ungrounded_fact_penalty("제3기갑여단이 전차 8대를 격파", source)
    assert p > 0.0
    # no checkable facts -> no penalty
    assert ungrounded_fact_penalty("적이 이동 중이다", source) == 0.0
    # fabricated year (2026 -> 2023) is flagged; keeping the real year is not
    assert ungrounded_fact_penalty("SITREP-2023 보고", "보고번호 SITREP-2026") > 0.0
    assert ungrounded_fact_penalty("SITREP-2026 보고", "보고번호 SITREP-2026") == 0.0


def test_hallucination_lowers_reward_for_invented_units():
    cfg = RewardConfig()
    source = "제1기계화보병대대가 전차 2대를 파괴하였음"
    faithful = compute_reward("제1기계화보병대대가 전차 2대를 파괴", source, [], [], cfg)
    invented = compute_reward("제3기갑여단이 전차 8대를 격파", source, [], [], cfg)
    assert invented.hallucination > faithful.hallucination
    assert invented.total < faithful.total


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
        - cfg.w_hallucination * bd.hallucination
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


def test_key_sentence_coverage_weighted_prioritizes_important():
    # (sentence, weight) pairs -> weighted average. Reflecting the high-weight
    # military event scores higher than reflecting only the low-weight line.
    ks = [("적 전차 4대를 파괴하였음", 3.0), ("보고번호는 SITREP 이다", 1.0)]
    hit_important = key_sentence_coverage("아군이 적 전차 파괴", ks)
    hit_trivial = key_sentence_coverage("보고번호 SITREP", ks)
    assert hit_important > hit_trivial
    # plain strings still work (equal weight) -> unchanged behavior
    assert key_sentence_coverage("아무말", []) == 1.0


def test_key_sentence_extractor_weights_military_events():
    from summarize_rl.config import RewardConfig
    from summarize_rl.keysent import KeySentenceExtractor, importance
    from summarize_rl.llm_backend import MockBackend

    # importance: a combat+casualty+quantity sentence outranks an admin line.
    assert importance("1중대는 적 전차 2대를 파괴하였고 아군 1명이 전사함") > importance("보고번호 SITREP-2026")

    cfg = RewardConfig(keysent_use_llm=False)  # rules-only -> deterministic
    be = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    src = ("보고번호 SITREP-2026.\n"
           "1중대는 적 전차 2대를 파괴하였고 아군 1명이 전사함.\n"
           "날씨는 맑음.")
    ks = KeySentenceExtractor(cfg).extract(be, src)
    sents = [s for s, _ in ks]
    # military event sentence is always included; non-military lines are dropped
    assert any("전차" in s for s in sents)
    assert not any("날씨" in s for s in sents)
    assert not any("보고번호" in s for s in sents)
    # its weight exceeds the base (1.0)
    assert max(w for s, w in ks if "전차" in s) > 1.0


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
