"""Integration test: HFBackend against a REAL transformers model.

The model is a tiny GPT-2 built and saved locally (no network/hub download), so
this genuinely exercises the transformers forward pass, KV cache, hidden-state
extraction and left padding — the boundary the MockBackend cannot cover.

Skips gracefully if transformers/tokenizers are unavailable.
"""

import torch
import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers

    d = str(tmp_path_factory.mktemp("tiny_gpt2"))
    vocab = {
        "<pad>": 0, "<eos>": 1, "source": 2, "core": 3, "gloss": 4,
        "summarize": 5, "text": 6, "here": 7, "info": 8,
        "적": 9, "부대": 10, "이동": 11,
    }
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<pad>"
    )
    wrapped.save_pretrained(d)

    cfg = GPT2Config(
        vocab_size=len(vocab), n_positions=64, n_embd=16, n_layer=2, n_head=2,
        bos_token_id=1, eos_token_id=1,
    )
    GPT2LMHeadModel(cfg).save_pretrained(d)
    return d


def _branch_texts():
    return {"XQ": "source text here", "SQ": "core info", "GQ": "gloss", "Q": "summarize"}


def test_hf_backend_shapes_and_freeze(tiny_model_dir):
    from summarize_rl.llm_backend import HFBackend

    be = HFBackend(tiny_model_dir, device="cpu", dtype="float32")
    assert be.hidden_size == 16
    assert be.vocab_size == 12
    # All LLM parameters are frozen.
    for p in be.model.parameters():
        assert not p.requires_grad

    state, step = be.start(_branch_texts())
    assert step.logits.shape == (4, be.vocab_size)
    assert step.hidden.shape == (4, be.hidden_size)
    out = be.step(state, 2)
    assert out.logits.shape == (4, be.vocab_size)
    assert out.hidden.shape == (4, be.hidden_size)


def test_hf_backend_generate_and_policy_grad(tiny_model_dir):
    from summarize_rl.config import DecodeConfig, PolicyConfig
    from summarize_rl.decoder import generate
    from summarize_rl.llm_backend import HFBackend
    from summarize_rl.policy import WeightPolicy

    be = HFBackend(tiny_model_dir, device="cpu", dtype="float32")
    policy = WeightPolicy(PolicyConfig(llm_hidden_size=be.hidden_size, hidden_dim=16))
    cfg = DecodeConfig(max_new_tokens=5, min_new_tokens=1, eos_token_id=be.eos_token_id)
    gen = torch.Generator().manual_seed(0)

    r = generate(be, _branch_texts(), policy, cfg, generator=gen)
    assert 1 <= r.length <= 5

    loss = -r.sum_logp()
    loss.backward()
    # Gradient reaches the policy...
    assert policy.head.weight.grad is not None
    assert torch.any(policy.head.weight.grad != 0)
    # ...and never the frozen LLM.
    for p in be.model.parameters():
        assert p.grad is None


def test_hf_backend_full_scst_step(tiny_model_dir):
    import copy
    from summarize_rl.branches import Example, Triplet
    from summarize_rl.config import Config, DecodeConfig, PolicyConfig, TrainConfig
    from summarize_rl.glossary import Glossary
    from summarize_rl.llm_backend import HFBackend
    from summarize_rl.policy import WeightPolicy
    from summarize_rl.train import SCSTTrainer

    be = HFBackend(tiny_model_dir, device="cpu", dtype="float32")
    cfg = Config()
    cfg.policy = PolicyConfig(llm_hidden_size=be.hidden_size, hidden_dim=16)
    cfg.decode = DecodeConfig(max_new_tokens=5, min_new_tokens=1, eos_token_id=be.eos_token_id)
    cfg.train = TrainConfig(num_samples=3, total_steps=5, grad_accum_steps=1, seed=0)

    policy = WeightPolicy(cfg.policy)
    glossary = Glossary({"기동": ["이동"]})
    gen = torch.Generator().manual_seed(0)
    trainer = SCSTTrainer(policy, be, cfg, glossary=glossary, generator=gen)

    ex = Example(source="적 부대 이동", triplets=[Triplet("적", "행동", "이동")], query="summarize")
    before = copy.deepcopy(policy.state_dict())
    m = trainer.train_step([ex])
    after = policy.state_dict()

    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert m.step == 1
    for p in be.model.parameters():
        assert p.grad is None  # frozen throughout
