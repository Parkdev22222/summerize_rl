"""GRPO (Group Relative Policy Optimization) pieces for SARA's RL loop.

SARA's default RL is self-critical (SCST): advantage = sample_reward −
greedy_reward, optimized with a plain REINFORCE criterion. GRPO instead:

  * drops the greedy critic and uses the **group** (the ``num_return_sequences``
    rollouts of one prompt) as the baseline: advantage = (r − group_mean) /
    (group_std + eps);
  * optimizes a **PPO clipped surrogate** on the importance ratio π_θ/π_old;
  * adds a **KL penalty to a frozen reference policy** (here the base LM without
    the context-aware FC head), using the unbiased k3 estimator.

Kept in a separate module (like reward_extras.py) so the upstream SARA files
stay minimally changed and the math is unit-testable. Enabled by --rl_algo grpo
(default scst).

Multi-epoch (μ>1) is supported: the rollout is re-scored under the *updated*
context-aware policy each inner iteration (see ``sad_branch_features`` /
``sad_logprobs_from_features`` in modeling_exaone_sad.py), so π_θ drifts from
π_old across epochs and the PPO clip becomes active. At μ=1 (or the very first
epoch) π_θ == π_old, the ratio is 1, and the clip is a no-op — the effective
objective is then group-advantage policy gradient + KL (DeepSeekMath GRPO with
one inner iteration). ``GRPOLoss.last_clipfrac`` reports the fraction of tokens
the clip actually bound, so you can watch it rise once μ>1 kicks in.

⚠️ Validate reference_logprobs on real hardware: it runs the backbone's own
forward (``_sad_base_forward``) over prompt+rollout to score the reference.
"""

from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_normalized_advantage(rewards_1d, group_size, eps: float = 1e-4):
    """GRPO advantage: standardize each prompt's group of rollout rewards.

    ``rewards_1d`` are per-rollout scalar rewards ordered so that every
    ``group_size`` consecutive entries belong to one prompt (HF `generate` with
    num_return_sequences yields exactly this layout). Returns a same-length,
    same-order 1-D array of advantages with per-group mean≈0, std≈1.
    """
    r = np.asarray(rewards_1d, dtype=np.float64).reshape(-1)
    n = r.shape[0]
    assert n % group_size == 0, "rewards length {} not divisible by group_size {}".format(n, group_size)
    groups = r.reshape(n // group_size, group_size)
    mean = groups.mean(axis=1, keepdims=True)
    std = groups.std(axis=1, keepdims=True)
    adv = (groups - mean) / (std + eps)
    return adv.reshape(-1)


class GRPOLoss(nn.Module):
    """PPO clipped surrogate + KL-to-reference, averaged over unmasked tokens.

    All inputs are per-token ``[N, L]`` (N = batch*num_return, L = gen length):
      new_logp  — logprob of the generated tokens under the current policy (grad)
      old_logp  — same under the sampling policy (detached; == new_logp at μ=1)
      ref_logp  — same under the frozen reference policy (detached)
      adv       — per-token advantage (group-normalized, broadcast over L)
      mask      — 1 for real tokens, 0 after EOS/pad
    KL uses the k3 estimator exp(Δ) − Δ − 1 (Δ = ref − new), which is ≥0 and
    unbiased. Returns a scalar loss; `.last_kl` holds the mean KL and
    `.last_clipfrac` the fraction of unmasked tokens whose ratio left the
    [1-eps, 1+eps] band (both detached, for logging).
    """

    def forward(self, new_logp, old_logp, ref_logp, adv, mask,
                clip_eps: float = 0.2, kl_beta: float = 0.04):
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        pg = -torch.min(surr1, surr2)

        diff = ref_logp - new_logp
        kl = torch.exp(diff) - diff - 1.0  # k3: unbiased, >= 0

        denom = mask.sum().clamp(min=1.0)
        self.last_kl = float(((kl * mask).sum() / denom).detach())
        # clip fraction: tokens where the ratio was pushed outside the trust band
        # (CleanRL-style diagnostic). 0 at μ=1 (ratio==1); rises once π_θ drifts.
        clipped = (torch.abs(ratio - 1.0) > clip_eps).to(mask.dtype)
        self.last_clipfrac = float(((clipped * mask).sum() / denom).detach())
        per_tok = pg + kl_beta * kl
        return (per_tok * mask).sum() / denom


def reference_logprobs(model, prompt_ids, prompt_mask, gen_result):
    """Per-token logprob of the generated tokens under the reference policy.

    Reference = the frozen base LM (no context-aware FC mixing). Runs the
    backbone's native forward over ``[prompt ; rollout]`` (teacher forcing) once,
    then gathers the logprob at each generated token. Returns a detached
    ``[N, L]`` tensor aligned with ``gen_result``.
    """
    nr = gen_result.shape[0] // prompt_ids.shape[0]
    dev = prompt_ids.device
    prompt = prompt_ids.repeat_interleave(nr, dim=0)
    pmask = prompt_mask.repeat_interleave(nr, dim=0)
    gen = gen_result.to(dev)
    full = torch.cat([prompt, gen], dim=1)
    gmask = (gen > 0).to(pmask.dtype)
    full_mask = torch.cat([pmask, gmask], dim=1)

    base_forward = getattr(model, "_sad_base_forward", None)
    with torch.no_grad():
        if base_forward is not None:
            out = base_forward(model, input_ids=full, attention_mask=full_mask,
                               use_cache=False, output_hidden_states=False, return_dict=True)
        else:
            out = model(input_ids=full, attention_mask=full_mask, use_cache=False, return_dict=True)
        logits = out.logits
        plen = prompt.shape[1]
        gen_logits = logits[:, plen - 1:-1, :]           # predicts positions plen..plen+glen-1
        logp = F.log_softmax(gen_logits.float(), dim=-1)
        ref = logp.gather(2, gen.unsqueeze(2)).squeeze(2)  # [N, L]
    return ref.detach()
