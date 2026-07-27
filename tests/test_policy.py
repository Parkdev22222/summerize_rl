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
    # Weighted over b,c,d: a plain b+c+d sum is identically 1 (softmax) and,
    # with a fixed, would give an analytically-zero gradient.
    loss = 1.0 * w.b.sum() + 2.0 * w.c.sum() + 3.0 * w.d.sum()
    loss.backward()
    assert policy.head.weight.grad is not None
    assert torch.any(policy.head.weight.grad != 0)


def test_learn_a_is_default():
    # Default: `a` is learned per token (sigmoid), with a gradient.
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    assert cfg.learn_a is True and cfg.fixed_a == 0.5
    policy = WeightPolicy(cfg)
    w = policy(_hidden(4, 8).as_features(cfg.use_contrast_features))
    assert torch.all(w.a > 0) and torch.all(w.a < 1)
    assert w.a.requires_grad


def test_fixed_a_is_constant_when_disabled():
    # learn_a=False -> a is fixed at fixed_a, input-independent, no gradient.
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16, learn_a=False)
    policy = WeightPolicy(cfg)
    w1 = policy(_hidden(4, 8).as_features(cfg.use_contrast_features))
    w2 = policy(_hidden(4, 8).as_features(cfg.use_contrast_features))
    assert torch.allclose(w1.a, torch.full_like(w1.a, 0.5))
    assert torch.allclose(w2.a, torch.full_like(w2.a, 0.5))
    assert w1.a.requires_grad is False
    # b + c + d still a valid learned simplex.
    assert torch.allclose(w1.b + w1.c + w1.d, torch.ones_like(w1.b), atol=1e-5)


def test_fixed_a_custom_value():
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16, learn_a=False, fixed_a=0.3)
    policy = WeightPolicy(cfg)
    w = policy(_hidden(3, 8).as_features(cfg.use_contrast_features))
    assert torch.allclose(w.a, torch.full_like(w.a, 0.3))


def test_fixed_a_no_grad_to_a_column_but_bcd_learns():
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16, learn_a=False)
    policy = WeightPolicy(cfg)
    w = policy(_hidden(4, 8).as_features(cfg.use_contrast_features))
    # Weighted loss (not b+c+d, which is always 1 under softmax -> zero grad).
    (1.0 * w.b + 2.0 * w.c + 3.0 * w.d).sum().backward()
    g = policy.head.weight.grad
    assert g is not None
    assert torch.all(g[0] == 0)      # a-column receives no gradient
    assert torch.any(g[1:] != 0)     # b/c/d columns are still learning


def test_learn_a_true_varies_and_has_gradient():
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16, learn_a=True, init_std=0.1)
    policy = WeightPolicy(cfg)
    feats = _hidden(6, 8).as_features(cfg.use_contrast_features)
    w = policy(feats)
    assert torch.all(w.a > 0) and torch.all(w.a < 1)
    assert w.a.requires_grad
    w.a.sum().backward()
    assert policy.head.weight.grad[0].abs().sum() > 0  # a-column gets gradient


def test_a_max_caps_learned_a():
    # a_max < 1 bounds the learned `a` to (0, a_max); neutral init -> a_max/2.
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16, a_max=0.5, init_std=0.1)
    policy = WeightPolicy(cfg)
    feats = _hidden(16, 8).as_features(cfg.use_contrast_features)
    w = policy(feats)
    assert torch.all(w.a > 0) and torch.all(w.a < 0.5)
    assert w.a.requires_grad
    # Zero-bias, zero-weight head -> a = a_max * sigmoid(0) = a_max/2.
    cfg0 = PolicyConfig(llm_hidden_size=8, hidden_dim=16, a_max=0.5, init_std=0.0)
    p0 = WeightPolicy(cfg0)
    p0.eval()
    w0 = p0(torch.zeros(2, cfg0.input_dim))
    assert torch.allclose(w0.a, torch.full_like(w0.a, 0.25), atol=1e-4)


def test_a_max_default_is_uncapped():
    # Default a_max=1.0 == plain sigmoid: neutral init gives a=0.5 (unchanged).
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16, init_std=0.0)
    assert cfg.a_max == 1.0
    policy = WeightPolicy(cfg)
    policy.eval()
    w = policy(torch.zeros(2, cfg.input_dim))
    assert torch.allclose(w.a, torch.full_like(w.a, 0.5), atol=1e-4)


def test_entropy_bounds():
    cfg = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    policy = WeightPolicy(cfg)
    feats = _hidden(3, 8).as_features(cfg.use_contrast_features)
    w = policy(feats)
    ent = policy.entropy(w)
    # [b,c,d] categorical entropy (<= ln 3) + Bernoulli(a) entropy (<= ln 2)
    max_ent = torch.log(torch.tensor(3.0)) + torch.log(torch.tensor(2.0))
    assert torch.all(ent >= 0)
    assert torch.all(ent <= max_ent + 1e-5)
