import torch

from summarize_rl.config import DecodeConfig, PolicyConfig
from summarize_rl.decoder import (
    combine_logits,
    generate,
    generate_batch,
    pmi_contrast,
    score_tokens,
    score_tokens_batch,
    _plausibility_mask,
    _top_p_mask,
)
from summarize_rl.llm_backend import MockBackend, StepOutput
from summarize_rl.policy import WeightPolicy, Weights


def _batch_setup(max_new=10, min_new=3):
    torch.manual_seed(0)
    policy = WeightPolicy(PolicyConfig(llm_hidden_size=8, hidden_dim=16))
    policy.eval()
    backend = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    cfg = DecodeConfig(max_new_tokens=max_new, min_new_tokens=min_new, eos_token_id=1)
    bt = {"XQ": "source text", "SQ": "core", "GQ": "gloss", "Q": "q"}
    return backend, bt, policy, cfg


def test_generate_batch_greedy_matches_single():
    # MockBackend logits are history-independent, so every batched greedy
    # rollout must equal the single greedy decode token-for-token.
    backend, bt, policy, cfg = _batch_setup()
    single = generate(backend, bt, policy, cfg, greedy=True)
    batch = generate_batch(backend, bt, policy, cfg, n=4, greedy=True)
    assert len(batch) == 4
    for r in batch:
        assert r.token_ids == single.token_ids
        assert len(r.logps) == len(r.token_ids)
        assert len(r.contrasts) == len(r.token_ids)


def test_score_tokens_batch_matches_single():
    backend, bt, policy, cfg = _batch_setup()
    seqs = [[2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12, 13]]  # different lengths
    batched = score_tokens_batch(backend, bt, policy, seqs, cfg)
    assert len(batched) == 3
    for r, ids in enumerate(seqs):
        single = score_tokens(backend, bt, policy, ids, cfg)
        assert len(batched[r].logps) == len(ids)
        lp_single = torch.stack(single.logps)
        lp_batch = torch.stack(batched[r].logps)
        assert torch.allclose(lp_single, lp_batch, atol=1e-5)


def test_score_tokens_batch_carries_grad():
    backend, bt, policy, cfg = _batch_setup()
    batched = score_tokens_batch(backend, bt, policy, [[2, 3, 4], [5, 6]], cfg)
    torch.stack(batched[0].logps).sum().backward()
    g = policy.head.weight.grad
    assert g is not None and bool((g != 0).any())


def test_mock_backend_batch_shapes():
    backend = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    bt = {"XQ": "a", "SQ": "b", "GQ": "c", "Q": "d"}
    state, step = backend.start_batch(bt, n=3)
    assert step.logits.shape == (3, 4, 16)
    assert step.hidden.shape == (3, 4, 8)
    step2 = backend.step_batch(state, [2, 3, 4])
    assert step2.logits.shape == (3, 4, 16)


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


def test_plausibility_mask_prunes_implausible_tokens():
    # XQ (base, branch 0) concentrates on tokens 0,1; tokens 2,3 ~0 probability.
    # With alpha=0.1 the base-implausible tokens 2,3 are masked to -inf.
    branch = torch.tensor([
        [5.0, 4.0, -10.0, -10.0],  # XQ base
        [0.0, 0.0, 0.0, 0.0],      # SQ
        [0.0, 0.0, 0.0, 0.0],      # GQ
        [0.0, 0.0, 0.0, 0.0],      # Q
    ])
    combined = torch.zeros(4)
    out = _plausibility_mask(combined, branch, alpha=0.1)
    assert torch.isfinite(out[0]) and torch.isfinite(out[1])
    assert out[2] == float("-inf") and out[3] == float("-inf")


def test_plausibility_mask_noop_when_disabled():
    branch = torch.randn(4, 6)
    combined = torch.randn(6)
    out = _plausibility_mask(combined, branch, alpha=0.0)
    assert torch.equal(out, combined)  # exactly unchanged (same object semantics)


def test_plausibility_mask_never_empty():
    # The base's own top token always survives (log alpha < 0), even at tiny alpha.
    branch = torch.randn(4, 20)
    combined = torch.zeros(20)
    out = _plausibility_mask(combined, branch, alpha=1e-6)
    top = int(torch.argmax(branch[0]).item())
    assert torch.isfinite(out[top])
    assert bool(torch.isfinite(out).any())


def test_plausibility_mask_batched_shape():
    branch = torch.randn(3, 4, 7)  # [N,4,V]
    combined = torch.zeros(3, 7)
    out = _plausibility_mask(combined, branch, alpha=0.2)
    assert out.shape == (3, 7)
    # every row keeps at least its base top token
    for r in range(3):
        top = int(torch.argmax(branch[r, 0]).item())
        assert torch.isfinite(out[r, top])


def test_generate_with_plausibility_batch_matches_single():
    # With the constraint on, batched greedy still equals single greedy
    # (MockBackend is history-independent; both apply the identical XQ mask).
    backend, bt, policy, _cfg = _batch_setup()
    cfg = DecodeConfig(max_new_tokens=10, min_new_tokens=3, eos_token_id=1,
                       plausibility_alpha=0.1)
    single = generate(backend, bt, policy, cfg, greedy=True)
    batch = generate_batch(backend, bt, policy, cfg, n=3, greedy=True)
    for r in batch:
        assert r.token_ids == single.token_ids


def test_score_tokens_batch_plausibility_matches_single():
    backend, bt, policy, _cfg = _batch_setup()
    cfg = DecodeConfig(max_new_tokens=10, min_new_tokens=3, eos_token_id=1,
                       plausibility_alpha=0.1)
    seqs = [[2, 3, 4, 5], [6, 7, 8]]
    batched = score_tokens_batch(backend, bt, policy, seqs, cfg)
    for r, ids in enumerate(seqs):
        single = score_tokens(backend, bt, policy, ids, cfg)
        lp_single = torch.stack(single.logps)
        lp_batch = torch.stack(batched[r].logps)
        assert torch.allclose(lp_single, lp_batch, atol=1e-5)


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
