"""Integration test: the --compile StaticCache decode path == the dynamic path.

Builds a tiny *Llama* locally (StaticCache-compatible; no download) and checks
that summaries from the compiled/StaticCache backend are token-identical to the
default DynamicCache backend. This is the correctness guarantee behind
``HFBackend(compile_decode=True)`` — the CUDA-graph speedup is a GPU runtime
property, but the output must not change.

Skips gracefully if transformers/tokenizers are unavailable.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def tiny_llama_dir(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers

    d = str(tmp_path_factory.mktemp("tiny_llama"))
    vocab = {
        "<pad>": 0, "<eos>": 1, "source": 2, "core": 3, "gloss": 4,
        "summarize": 5, "text": 6, "here": 7, "info": 8,
        "적": 9, "부대": 10, "이동": 11, "고지": 12, "점령": 13,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<pad>"
    )
    wrapped.save_pretrained(d)

    cfg = LlamaConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=1,
    )
    LlamaForCausalLM(cfg).save_pretrained(d)
    return d


def _summarize_with(backend, hidden_size):
    from summarize_rl.config import Config, DecodeConfig, PolicyConfig
    from summarize_rl.infer import Summarizer
    from summarize_rl.policy import WeightPolicy

    torch.manual_seed(0)  # identical policy init for both backends
    cfg = Config()
    cfg.policy = PolicyConfig(llm_hidden_size=hidden_size, hidden_dim=16)
    cfg.decode = DecodeConfig(
        max_new_tokens=8, min_new_tokens=3,
        eos_token_id=backend.eos_token_id, pad_token_id=backend.pad_token_id,
    )
    policy = WeightPolicy(cfg.policy)
    summarizer = Summarizer(backend, policy, cfg)
    return summarizer.summarize("적 부대가 이동 고지 점령")


def test_static_compiled_matches_dynamic(tiny_llama_dir):
    from summarize_rl.llm_backend import HFBackend

    dynamic = HFBackend(tiny_llama_dir, device="cpu", dtype="float32")
    static = HFBackend(
        tiny_llama_dir, device="cpu", dtype="float32",
        compile_decode=True, max_seq_len=64,
    )

    r_dyn = _summarize_with(dynamic, dynamic.hidden_size)
    r_sta = _summarize_with(static, static.hidden_size)

    # Same greedy decode -> identical summary text and same mean weights.
    assert r_sta.text == r_dyn.text
    for a, b in zip(r_sta.mean_weights, r_dyn.mean_weights):
        assert a == pytest.approx(b, abs=1e-4)


def test_static_cache_uncompiled_matches_dynamic(tiny_llama_dir):
    """StaticCache path without torch.compile also matches (isolates the cache)."""
    from summarize_rl.llm_backend import HFBackend

    dynamic = HFBackend(tiny_llama_dir, device="cpu", dtype="float32")
    static = HFBackend(tiny_llama_dir, device="cpu", dtype="float32",
                       compile_decode=True, max_seq_len=64)
    # Bypass the compiled model to test the StaticCache logic in isolation.
    static._decode_model = static.model

    assert _summarize_with(static, static.hidden_size).text == \
        _summarize_with(dynamic, dynamic.hidden_size).text


def test_prompt_too_long_raises(tiny_llama_dir):
    from summarize_rl.llm_backend import HFBackend

    be = HFBackend(tiny_llama_dir, device="cpu", dtype="float32",
                   compile_decode=True, max_seq_len=4)
    with pytest.raises(ValueError, match="max_seq_len"):
        be.start({"XQ": "source text here info core", "SQ": "core", "GQ": "gloss", "Q": "summarize"})
