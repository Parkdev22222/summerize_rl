import torch

from summarize_rl.config import PolicyConfig
from summarize_rl.policy import BranchHidden, WeightPolicy


def _hidden(batch, dim):
    return BranchHidden(
        h_xq=torch.randn(batch, dim),
        h_sq=torch.randn(batch, dim),
        h_gq=torch.randn(batch, dim),
        h_q=torch.randn(batch, dim),
    )


def test_input_dim_with_and_without_contrast():
    c = PolicyConfig(llm_hidden_size=8, use_contrast_features=False)
    assert c.input_dim == 4 * 8
    c2 = PolicyConfig(llm_hidden_size=8, use_contrast_features=True)
    assert c2.input_dim == 7 * 8


def test_features_shape():
    cfg = PolicyConfig(llm_hidden_size=8, use_contrast_features=True)
    feats = _hidden(3, 8).as_features(use_contrast=True)
    assert feats.shape == (3, cfg.input_dim)


def test_weight_constraints():
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    policy = WeightPolicy(cfg)
    feats = _hidden(5, 8).as_features(cfg.use_contrast_features)
    w = policy(feats)
    # a in (0,1)
    assert torch.all(w.a > 0) and torch.all(w.a < 1)
    # b + c + d = 1
    total = w.b + w.c + w.d
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    # each in (0,1)
    for t in (w.b, w.c, w.d):
        assert torch.all(t > 0) and torch.all(t < 1)


def test_warm_start_neutral():
    # Zero-bias, tiny-weight head -> near a=0.5, b=c=d=1/3.
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16, init_std=0.0)
    policy = WeightPolicy(cfg)
    policy.eval()
    feats = torch.zeros(2, cfg.input_dim)
    w = policy(feats)
    assert torch.allclose(w.a, torch.full_like(w.a, 0.5), atol=1e-4)
    third = torch.full_like(w.b, 1 / 3)
    assert torch.allclose(w.b, third, atol=1e-4)
    assert torch.allclose(w.c, third, atol=1e-4)
    assert torch.allclose(w.d, third, atol=1e-4)


def test_policy_is_fp32():
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    policy = WeightPolicy(cfg)
    for p in policy.parameters():
        assert p.dtype == torch.float32


def test_gradient_flows_to_policy():
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    policy = WeightPolicy(cfg)
    feats = _hidden(4, 8).as_features(cfg.use_contrast_features)
    w = policy(feats)
    loss = (w.a.sum() + w.b.sum() + w.c.sum() + w.d.sum())
    loss.backward()
    assert policy.head.weight.grad is not None
    assert torch.any(policy.head.weight.grad != 0)


def test_entropy_bounds():
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    policy = WeightPolicy(cfg)
    feats = _hidden(3, 8).as_features(cfg.use_contrast_features)
    w = policy(feats)
    ent = policy.entropy(w)
    # entropy of 3-way categorical in [0, ln 3]
    assert torch.all(ent >= 0)
    assert torch.all(ent <= torch.log(torch.tensor(3.0)) + 1e-5)
