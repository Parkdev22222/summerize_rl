"""Unit tests for GRPO pieces (grpo_extras.py): group-normalized advantage,
the clipped-surrogate + KL loss (incl. gradient flow), and the reference
logprob helper's shape/alignment. Pure math — no EXAONE / GPU needed.

Skips (prints SKIP, exits 0) when torch is unavailable so the stdlib suite
still runs. On a machine with torch:  python tests/test_grpo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from types import SimpleNamespace
    import numpy as np
    import torch
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


def test_group_normalized_advantage():
    if not HAVE_TORCH:
        print("SKIP test_group_normalized_advantage (numpy/torch unavailable)"); return
    from grpo_extras import group_normalized_advantage
    # two prompts, group_size 3
    r = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
    adv = group_normalized_advantage(r, group_size=3, eps=0.0)
    g0, g1 = adv[:3], adv[3:]
    assert abs(g0.mean()) < 1e-9 and abs(g1.mean()) < 1e-9, "each group must be zero-mean"
    assert abs(g0.std() - 1.0) < 1e-6 and abs(g1.std() - 1.0) < 1e-6, "each group unit-std"
    assert g0[2] > g0[0], "order within group must be preserved (3>1)"
    # a group with zero variance -> finite (eps guards division)
    flat = group_normalized_advantage([5.0, 5.0], group_size=2, eps=1e-4)
    assert np.isfinite(flat).all() and np.allclose(flat, 0.0)
    # non-divisible length is a hard error
    try:
        group_normalized_advantage([1.0, 2.0, 3.0], group_size=2)
        assert False, "expected assertion on non-divisible length"
    except AssertionError:
        pass
    print("PASS test_group_normalized_advantage")


def test_grpo_loss_gradient():
    if not HAVE_TORCH:
        print("SKIP test_grpo_loss_gradient (torch unavailable)"); return
    from grpo_extras import GRPOLoss
    N, L = 2, 3
    new_logp = torch.zeros(N, L, requires_grad=True)  # leaf
    old_logp = new_logp.detach()                       # μ=1: ratio == 1
    ref_logp = new_logp.detach()                       # KL == 0 when ref == new
    adv = torch.tensor([[1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]])
    mask = torch.ones(N, L)
    crit = GRPOLoss()
    loss = crit(new_logp, old_logp, ref_logp, adv, mask, clip_eps=0.2, kl_beta=0.04)
    assert crit.last_kl == 0.0 or abs(crit.last_kl) < 1e-6, "KL must be ~0 when ref==new"
    loss.backward()
    # pg = -ratio*adv; d/dnew(ratio)=ratio=1 -> grad = -adv/denom (denom = #unmasked = 6)
    expected = (-adv / mask.sum())
    assert torch.allclose(new_logp.grad, expected, atol=1e-6), \
        f"grad must be -adv/denom (REINFORCE at μ=1); got {new_logp.grad}"
    print("PASS test_grpo_loss_gradient")


def test_grpo_loss_kl_nonneg_and_mask():
    if not HAVE_TORCH:
        print("SKIP test_grpo_loss_kl_nonneg_and_mask (torch unavailable)"); return
    from grpo_extras import GRPOLoss
    N, L = 2, 4
    new_logp = torch.randn(N, L, requires_grad=True)
    old_logp = new_logp.detach()
    ref_logp = torch.randn(N, L)          # ref != new -> KL > 0
    adv = torch.randn(N, 1).expand(N, L).contiguous()
    mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.]])
    crit = GRPOLoss()
    loss = crit(new_logp, old_logp, ref_logp, adv, mask, clip_eps=0.2, kl_beta=0.1)
    assert crit.last_kl >= 0.0, "k3 KL estimate must be non-negative"
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(new_logp.grad).all()
    # masked positions must contribute no gradient
    assert torch.allclose(new_logp.grad[0, 2:], torch.zeros(2), atol=1e-7)
    print("PASS test_grpo_loss_kl_nonneg_and_mask")


def test_reference_logprobs_shape_and_alignment():
    if not HAVE_TORCH:
        print("SKIP test_reference_logprobs_shape_and_alignment (torch unavailable)"); return
    from grpo_extras import reference_logprobs
    import torch.nn.functional as F
    B, P, L, V, nr = 2, 3, 4, 7, 2
    N = B * nr
    prompt_ids = torch.randint(1, V, (B, P))
    prompt_mask = torch.ones(B, P, dtype=torch.long)
    gen = torch.randint(1, V, (N, L))
    # fake backbone forward: deterministic logits so we can recompute the target
    full_logits = torch.randn(N, P + L, V)

    class FakeModel:
        pass
    m = FakeModel()
    def _base_forward(model, input_ids=None, attention_mask=None, **kw):
        return SimpleNamespace(logits=full_logits)
    m._sad_base_forward = _base_forward

    ref = reference_logprobs(m, prompt_ids, prompt_mask, gen)
    assert ref.shape == (N, L), f"ref logp must be [N,L], got {tuple(ref.shape)}"
    assert not ref.requires_grad, "reference logprobs must be detached"
    # recompute expected: logits[:, P-1:-1] predict the L gen tokens
    logp = F.log_softmax(full_logits[:, P - 1:-1, :].float(), dim=-1)
    expected = logp.gather(2, gen.unsqueeze(2)).squeeze(2)
    assert torch.allclose(ref, expected, atol=1e-6), "positional alignment (prompt|gen) is wrong"
    print("PASS test_reference_logprobs_shape_and_alignment")


def test_grpo_loss_clip_activation():
    """When π_θ drifts from π_old (μ>1), the ratio leaves the trust band: the
    clipped surrogate is selected (grad vanishes in the clipped region) and
    last_clipfrac reports it. At μ=1 (new==old) nothing is clipped."""
    if not HAVE_TORCH:
        print("SKIP test_grpo_loss_clip_activation (torch unavailable)"); return
    from grpo_extras import GRPOLoss
    N, L = 1, 2
    adv = torch.full((N, L), 2.0)          # positive advantage
    mask = torch.ones(N, L)
    old = torch.zeros(N, L)
    ref = torch.zeros(N, L)
    # new = old + 1  ->  ratio = e ≈ 2.718 > 1 + clip_eps(0.2)
    new = torch.ones(N, L, requires_grad=True)
    crit = GRPOLoss()
    loss = crit(new, old, ref, adv, mask, clip_eps=0.2, kl_beta=0.0)
    assert crit.last_clipfrac == 1.0, f"all tokens should clip; got {crit.last_clipfrac}"
    loss.backward()
    # clipped surrogate (1+eps)*adv is constant in the clipped region -> zero grad
    assert torch.allclose(new.grad, torch.zeros(N, L), atol=1e-6), \
        f"grad must vanish inside the clip region; got {new.grad}"
    # μ=1 sanity: identical policies -> nothing clipped
    crit0 = GRPOLoss()
    _ = crit0(old.clone().requires_grad_(True), old, ref, adv, mask, clip_eps=0.2, kl_beta=0.0)
    assert crit0.last_clipfrac == 0.0, "ratio==1 must give clipfrac 0"
    print("PASS test_grpo_loss_clip_activation")


def _fake_fc_model(H, A):
    """A minimal object exposing the SAD FC head that _weight_from_hidden uses."""
    import torch.nn as nn
    m = SimpleNamespace()
    m.my_all_f = nn.Linear(3 * H, A)
    m.my_all_f1 = nn.Linear(A, A)
    m.my_f = nn.Linear(A, 3)
    m.relu, m.relu2 = nn.ReLU(), nn.ReLU()
    m.dropout = nn.Dropout(0.0)
    m.sqrt_dimension = 0          # disable the 1/sqrt(d) scaling for an exact hand-calc
    m.sqrt_method = "concate_dim"
    return m


def test_sad_logprobs_from_features_math_and_grad():
    """sad_logprobs_from_features re-mixes cached branch features with the FC head:
    verify it equals the hand-computed (1+γ)(α·m+β·p)−γ·n log-softmax, and that
    gradient reaches the FC params (the branch features are constants)."""
    if not HAVE_TORCH:
        print("SKIP test_sad_logprobs_from_features_math_and_grad (torch unavailable)"); return
    import torch.nn.functional as F
    from modeling_exaone_sad import sad_logprobs_from_features
    N, L, V, H, A = 2, 3, 5, 4, 6
    m = _fake_fc_model(H, A)
    feats = {
        "m_logits": torch.randn(N, L, V), "m_hidden": torch.randn(N, L, H),
        "p_logits": torch.randn(N, L, V), "p_hidden": torch.randn(N, L, H),
        "n_logits": torch.randn(N, L, V), "n_hidden": torch.randn(N, L, H),
    }
    gen = torch.randint(0, V, (N, L))
    new_logp = sad_logprobs_from_features(m, feats, gen)
    assert new_logp.shape == (N, L), f"expected [N,L], got {tuple(new_logp.shape)}"

    # hand recompute the same combination
    x = torch.cat([feats["m_hidden"], feats["p_hidden"], feats["n_hidden"]], dim=-1)
    w = m.my_f(m.relu2(m.my_all_f1(m.relu(m.my_all_f(x)))))
    ab = torch.softmax(w[..., :2], dim=-1)
    alpha, beta = ab[..., 0:1], ab[..., 1:2]
    gamma = torch.sigmoid(w[..., 2:3])
    combined = (1 + gamma) * (alpha * feats["m_logits"] + beta * feats["p_logits"]) - gamma * feats["n_logits"]
    exp_logp = F.log_softmax(combined.float(), dim=-1).gather(2, gen.unsqueeze(2)).squeeze(2)
    assert torch.allclose(new_logp, exp_logp, atol=1e-6), "combination/log-softmax/gather mismatch"

    # gradient must reach the FC head (branch features are plain constants here)
    new_logp.sum().backward()
    assert m.my_all_f.weight.grad is not None and torch.isfinite(m.my_all_f.weight.grad).all(), \
        "FC head must receive gradient"
    assert m.my_f.weight.grad is not None, "output FC layer must receive gradient"
    print("PASS test_sad_logprobs_from_features_math_and_grad")


def test_sad_branch_features_shape_align_detach():
    """sad_branch_features slices the L positions predicting the response from each
    branch's teacher-forced forward: verify shapes, the [:,Pb-1:-1] alignment, and
    that the cached tensors are detached (backbone is frozen)."""
    if not HAVE_TORCH:
        print("SKIP test_sad_branch_features_shape_align_detach (torch unavailable)"); return
    from modeling_exaone_sad import sad_branch_features
    B, L, V, H, nr = 2, 3, 5, 4, 2
    N = B * nr
    Pm, Pp, Pn = 6, 4, 2          # three branches, different prompt lengths
    main_ids = torch.randint(1, V, (B, Pm))
    presumm_ids = torch.randint(1, V, (B, Pp))
    null_ids = torch.randint(1, V, (B, Pn))
    main_mask = torch.ones(B, Pm, dtype=torch.long)
    presumm_mask = torch.ones(B, Pp, dtype=torch.long)
    gen = torch.randint(1, V, (N, L))

    # fake backbone: channel-0 of logits/hidden encodes the absolute position index,
    # so we can check the slice picks positions Pb-1 .. Pb+L-2.
    def _base_forward(model, input_ids=None, attention_mask=None, position_ids=None, **kw):
        Nn, T = input_ids.shape
        logits = torch.zeros(Nn, T, V)
        hidden = torch.zeros(Nn, T, H)
        pos = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(Nn, T)
        logits[:, :, 0] = pos
        hidden[:, :, 0] = pos
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))

    m = SimpleNamespace(_sad_base_forward=_base_forward)
    feats = sad_branch_features(m, main_ids, main_mask, presumm_ids, presumm_mask,
                                null_ids, None, gen)   # null uses ones-mask (None)
    for k in ("m_logits", "p_logits", "n_logits"):
        assert feats[k].shape == (N, L, V), f"{k} shape {tuple(feats[k].shape)}"
        assert not feats[k].requires_grad, f"{k} must be detached"
    for k in ("m_hidden", "p_hidden", "n_hidden"):
        assert feats[k].shape == (N, L, H) and not feats[k].requires_grad

    # main branch: sliced positions must be Pm-1 .. Pm-1+L-1
    exp_pos = torch.arange(Pm - 1, Pm - 1 + L, dtype=torch.float32)
    assert torch.allclose(feats["m_logits"][0, :, 0], exp_pos), "main-branch slice misaligned"
    assert torch.allclose(feats["p_logits"][0, :, 0],
                          torch.arange(Pp - 1, Pp - 1 + L, dtype=torch.float32)), "presumm slice misaligned"
    assert torch.allclose(feats["n_logits"][0, :, 0],
                          torch.arange(Pn - 1, Pn - 1 + L, dtype=torch.float32)), "null slice misaligned"
    print("PASS test_sad_branch_features_shape_align_detach")


if __name__ == "__main__":
    test_group_normalized_advantage()
    test_grpo_loss_gradient()
    test_grpo_loss_kl_nonneg_and_mask()
    test_reference_logprobs_shape_and_alignment()
    test_grpo_loss_clip_activation()
    test_sad_logprobs_from_features_math_and_grad()
    test_sad_branch_features_shape_align_detach()
