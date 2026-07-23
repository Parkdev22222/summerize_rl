"""Multi-branch PMI decoding (Section 2.1, 2.5.6).

Combined logit per token (work-plan formula 2.1):

    logit_c = (1 + a) * (b*logit_XQ + c*logit_SQ + d*logit_GQ) - a*logit_Q

with weights produced per step by the policy network. The LLM logits are
constants (detached); gradients flow only  logp -> combined logit -> (a,b,c,d)
-> policy theta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import DecodeConfig
from .llm_backend import LLMBackend, StepOutput
from .policy import BranchHidden, WeightPolicy, Weights


@dataclass
class Rollout:
    """One generated summary and the tensors SCST needs."""

    token_ids: list[int] = field(default_factory=list)
    logps: list[torch.Tensor] = field(default_factory=list)  # scalar, grad-carrying
    entropies: list[torch.Tensor] = field(default_factory=list)
    weight_trace: list[tuple[float, float, float, float]] = field(default_factory=list)
    contrasts: list[float] = field(default_factory=list)  # per-token PMI, reward-only
    text: str = ""
    hit_eos: bool = False

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def mean_contrast(self) -> float:
        """Mean per-token PMI contrast (source-conditioned vs. prior).

        A reward-side scalar (no gradient): how much the chosen tokens are
        favored by the source/core/term branches over the query prior Q. This
        is exactly the effect the prior-removal weight `a` modulates, so it
        gives `a` a learning signal the text-only reward terms cannot.
        """
        if not self.contrasts:
            return 0.0
        return sum(self.contrasts) / len(self.contrasts)

    def sum_logp(self) -> torch.Tensor:
        if not self.logps:
            return torch.zeros((), dtype=torch.float32)
        return torch.stack(self.logps).sum()

    def mean_entropy(self) -> torch.Tensor:
        if not self.entropies:
            return torch.zeros((), dtype=torch.float32)
        return torch.stack(self.entropies).mean()


def combine_logits(step: StepOutput, weights: Weights) -> torch.Tensor:
    """Apply the PMI combination formula for a single example (batch=1).

    step.logits: [4, V] (branch order XQ, SQ, GQ, Q), treated as constants.
    weights: each field shape [1].
    Returns combined logits [V] carrying grad to the policy weights.
    """
    logits = step.logits.detach()
    xq, sq, gq, q = logits[0], logits[1], logits[2], logits[3]
    a, b, c, d = weights.a, weights.b, weights.c, weights.d  # each [1]
    positive = b * xq + c * sq + d * gq
    return (1 + a) * positive - a * q  # [V]


def pmi_contrast(step: StepOutput, token: int) -> float:
    """Pointwise mutual information of `token`: source-conditioned vs. prior.

    Compares a policy-free uniform mixture of the source/core/term branches
    (XQ, SQ, GQ) against the query-only prior branch Q:

        pmi = log softmax(mean(XQ, SQ, GQ))[token] - log softmax(Q)[token]

    Positive when the chosen token is more probable given the source context
    than under the generic prior. Policy-independent (does not use a/b/c/d) so
    it is a stable reward measure; returned as a plain float (no gradient).
    """
    logits = step.logits.detach()
    positive = (logits[0] + logits[1] + logits[2]) / 3.0
    q = logits[3]
    lp_pos = F.log_softmax(positive, dim=-1)
    lp_q = F.log_softmax(q, dim=-1)
    return float((lp_pos[token] - lp_q[token]).item())


def _top_p_mask(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Return a boolean keep-mask for nucleus (top-p) sampling."""
    if top_p >= 1.0:
        return torch.ones_like(logits, dtype=torch.bool)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # Keep tokens up to and including the one that crosses top_p.
    sorted_keep = cumulative - probs <= top_p
    sorted_keep[0] = True  # always keep the top token
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep.scatter_(0, sorted_idx, sorted_keep)
    return keep


def _branch_hidden(step: StepOutput) -> BranchHidden:
    h = step.hidden  # [4, D]
    return BranchHidden(
        h_xq=h[0:1], h_sq=h[1:2], h_gq=h[2:3], h_q=h[3:4]
    )


def generate(
    backend: LLMBackend,
    branch_texts: dict[str, str],
    policy: WeightPolicy,
    config: DecodeConfig,
    *,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> Rollout:
    """Generate one summary with PMI-combined, policy-weighted decoding."""
    use_contrast = policy.config.use_contrast_features
    eos = config.eos_token_id if config.eos_token_id is not None else backend.eos_token_id

    state, step = backend.start(branch_texts)
    rollout = Rollout()

    for t in range(config.max_new_tokens):
        weights = policy(_branch_hidden(step).as_features(use_contrast))
        combined = combine_logits(step, weights)  # [V]

        # Enforce min_new_tokens by masking EOS until the floor is reached.
        if eos is not None and t < config.min_new_tokens:
            combined = combined.clone()
            combined[eos] = float("-inf")

        scaled = combined / max(config.temperature, 1e-6)

        if greedy:
            token = int(torch.argmax(scaled).item())
            logp = F.log_softmax(scaled, dim=-1)[token]
        else:
            keep = _top_p_mask(scaled, config.top_p)
            filtered = scaled.masked_fill(~keep, float("-inf"))
            logp_dist = F.log_softmax(filtered, dim=-1)
            probs = torch.exp(logp_dist)
            token = int(torch.multinomial(probs, 1, generator=generator).item())
            logp = logp_dist[token]

        rollout.token_ids.append(token)
        rollout.logps.append(logp)
        rollout.entropies.append(policy.entropy(weights).squeeze(0))
        rollout.contrasts.append(pmi_contrast(step, token))
        rollout.weight_trace.append(
            (
                float(weights.a.item()),
                float(weights.b.item()),
                float(weights.c.item()),
                float(weights.d.item()),
            )
        )

        if eos is not None and token == eos:
            rollout.hit_eos = True
            break

        step = backend.step(state, token)

    rollout.text = backend.decode(rollout.token_ids)
    return rollout


@dataclass
class ScoredSequence:
    """Per-token log-probs and entropies of a fixed token sequence."""

    logps: list[torch.Tensor] = field(default_factory=list)  # scalar, grad-carrying
    entropies: list[torch.Tensor] = field(default_factory=list)


def score_tokens(
    backend: LLMBackend,
    branch_texts: dict[str, str],
    policy: WeightPolicy,
    token_ids: list[int],
    config: DecodeConfig,
) -> ScoredSequence:
    """Teacher-forced re-scoring of a FIXED token sequence under `policy`.

    Recomputes each token's log-probability from the current policy weights,
    using the full temperature-scaled softmax (no nucleus truncation) so that
    importance ratios pi_theta / pi_theta_old are well defined. The same
    min_new_tokens EOS masking as sampling is applied so the distribution
    matches the one the tokens were drawn from. Used by GRPO to recompute the
    ratio (current policy) and the KL reference (frozen policy).
    """
    use_contrast = policy.config.use_contrast_features
    eos = config.eos_token_id if config.eos_token_id is not None else backend.eos_token_id

    state, step = backend.start(branch_texts)
    scored = ScoredSequence()
    for t, token in enumerate(token_ids):
        weights = policy(_branch_hidden(step).as_features(use_contrast))
        combined = combine_logits(step, weights)
        if eos is not None and t < config.min_new_tokens:
            combined = combined.clone()
            combined[eos] = float("-inf")
        scaled = combined / max(config.temperature, 1e-6)
        scored.logps.append(F.log_softmax(scaled, dim=-1)[token])
        scored.entropies.append(policy.entropy(weights).squeeze(0))
        step = backend.step(state, token)
    return scored
