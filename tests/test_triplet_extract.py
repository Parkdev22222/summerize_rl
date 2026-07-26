from summarize_rl.branches import Triplet
from summarize_rl.config import Config
from summarize_rl.llm_backend import MockBackend
from summarize_rl.policy import WeightPolicy
from summarize_rl.infer import Summarizer
from summarize_rl.triplet_extract import TripletExtractor, _parse_triplets


def test_parse_triplets_pipe_and_comma():
    text = (
        "블루포스 | 규모 | 1000명\n"
        "- 레드포스 | 무장 | M4\n"
        "(갈도비아, 목표, 국경통제)\n"
        "잡음 줄 (형식 아님)\n"
    )
    trs = _parse_triplets(text, n=10)
    assert Triplet("블루포스", "규모", "1000명") in trs
    assert Triplet("레드포스", "무장", "M4") in trs
    assert Triplet("갈도비아", "목표", "국경통제") in trs
    assert len(trs) == 3  # the malformed line is skipped


def test_triplet_extractor_caches():
    be = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    calls = {"n": 0}
    be.generate_text = lambda p, m=384: (
        calls.__setitem__("n", calls["n"] + 1) or "적 | 행동 | 이동"
    )
    ext = TripletExtractor()
    first = ext.extract(be, "원문 A")
    second = ext.extract(be, "원문 A")  # cached
    assert first == second == [Triplet("적", "행동", "이동")]
    assert calls["n"] == 1
    ext.extract(be, "원문 B")
    assert calls["n"] == 2


def test_summarizer_uses_extractor_when_triplets_absent():
    cfg = Config()
    cfg.policy.llm_hidden_size = 8
    cfg.decode.eos_token_id = 1
    cfg.decode.max_new_tokens = 6
    cfg.decode.min_new_tokens = 2
    be = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)

    class StubExtractor:
        def __init__(self):
            self.called_with = None

        def extract(self, backend, source):
            self.called_with = source
            return [Triplet("적", "행동", "이동")]

    stub = StubExtractor()
    s = Summarizer(be, WeightPolicy(cfg.policy), cfg, triplet_extractor=stub)
    s.summarize("적이 이동 중이다")  # no triplets passed -> extractor should fire
    assert stub.called_with == "적이 이동 중이다"

    # explicit triplets -> extractor NOT used
    stub.called_with = None
    s.summarize("적이 이동 중이다", triplets=[Triplet("x", "y", "z")])
    assert stub.called_with is None
