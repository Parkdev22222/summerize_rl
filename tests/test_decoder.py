import torch

from summarize_rl.config import DecodeConfig, PolicyConfig
from summarize_rl.decoder import combine_logits, generate, pmi_contrast, _top_p_mask
from summarize_rl.llm_backend import MockBackend, StepOutput
from summarize_rl.policy import WeightPolicy, Weights


def test_pmi_contrast_sign():
    # Token 0 is favored by the source branches (XQ/SQ/GQ) over the prior Q;
    # token 2 is favored by the prior. PMI must be positive for the former.
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0],  # XQ
            [5.0, 0.0, 0.0],  # SQ
            [5.0, 0.0, 0.0],  # GQ
            [0.0, 0.0, 5.0],  # Q (prior)
        ]
    )
    step = StepOutput(logits=logits, hidden=torch.zeros(4, 2))
    assert pmi_contrast(step, 0) > 0.0  # source-specific token
    assert pmi_contrast(step, 2) < 0.0  # prior-driven token


def _weights(a, b, c, d):
    return Weights(
        a=torch.tensor([a]),
        b=torch.tensor([b]),
        c=torch.tensor([c]),
        d=torch.tensor([d]),
    )


def test_combine_formula_matches_hand_computation():
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0],  # XQ
            [0.0, 1.0, 0.0],  # SQ
            [2.0, 0.0, 1.0],  # GQ
            [1.0, 1.0, 1.0],  # Q
        ]
    )
    step = StepOutput(logits=logits, hidden=torch.zeros(4, 2))
    a, b, c, d = 0.5, 0.2, 0.3, 0.5
    w = _weights(a, b, c, d)
    out = combine_logits(step, w)
    pos = b * logits[0] + c * logits[1] + d * logits[2]
    expected = (1 + a) * pos - a * logits[3]
    assert torch.allclose(out, expected, atol=1e-6)


def test_combine_gradient_flows_to_weights():
    logits = torch.randn(4, 5)
    step = StepOutput(logits=logits, hidden=torch.zeros(4, 2))
    a = torch.tensor([0.5], requires_grad=True)
    b = torch.tensor([0.2], requires_grad=True)
    c = torch.tensor([0.3], requires_grad=True)
    d = torch.tensor([0.5], requires_grad=True)
    out = combine_logits(step, Weights(a, b, c, d))
    out.sum().backward()
    assert a.grad is not None and b.grad is not None


def test_top_p_mask_keeps_top_token():
    logits = torch.tensor([5.0, 1.0, 0.5, 0.1])
    keep = _top_p_mask(logits, top_p=0.5)
    assert keep[0].item() is True
    # Very low-prob tail dropped.
    assert keep[-1].item() is False


def test_top_p_full_keeps_all():
    logits = torch.randn(10)
    keep = _top_p_mask(logits, top_p=1.0)
    assert bool(keep.all())


def _policy():
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    return WeightPolicy(cfg)


def _branch_texts():
    return {"XQ": "source", "SQ": "core", "GQ": "gloss", "Q": "q"}


def test_generate_respects_min_and_max():
    backend = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    policy = _policy()
    cfg = DecodeConfig(max_new_tokens=10, min_new_tokens=3, eos_token_id=1)
    gen = torch.Generator().manual_seed(0)
    r = generate(backend, _branch_texts(), policy, cfg, generator=gen)
    assert r.length <= 10
    # EOS cannot appear before min_new_tokens.
    for tok in r.token_ids[: cfg.min_new_tokens]:
        assert tok != 1


def test_generate_greedy_deterministic():
    backend = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    policy = _policy()
    policy.eval()
    cfg = DecodeConfig(max_new_tokens=6, min_new_tokens=2, eos_token_id=1)
    r1 = generate(backend, _branch_texts(), policy, cfg, greedy=True)
    r2 = generate(backend, _branch_texts(), policy, cfg, greedy=True)
    assert r1.token_ids == r2.token_ids


def test_generate_logp_carries_grad():
    backend = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    policy = _policy()
    cfg = DecodeConfig(max_new_tokens=5, min_new_tokens=1, eos_token_id=1)
    gen = torch.Generator().manual_seed(1)
    r = generate(backend, _branch_texts(), policy, cfg, generator=gen)
    loss = -r.sum_logp()
    loss.backward()
    assert policy.head.weight.grad is not None
    assert torch.any(policy.head.weight.grad != 0)


def test_weight_trace_recorded():
    backend = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    policy = _policy()
    cfg = DecodeConfig(max_new_tokens=5, min_new_tokens=1, eos_token_id=1)
    gen = torch.Generator().manual_seed(2)
    r = generate(backend, _branch_texts(), policy, cfg, generator=gen)
    assert len(r.weight_trace) == r.length
    for a, b, c, d in r.weight_trace:
        assert 0 < a < 1
        assert abs((b + c + d) - 1.0) < 1e-4
