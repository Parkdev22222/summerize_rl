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


if __name__ == "__main__":
    test_group_normalized_advantage()
    test_grpo_loss_gradient()
    test_grpo_loss_kl_nonneg_and_mask()
    test_reference_logprobs_shape_and_alignment()
