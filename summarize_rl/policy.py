"""Weight-generating policy network (Section 2.2 of the work plan).

The ONLY trainable component. Given per-branch hidden states at decoding step t,
it emits the four PMI combination weights [a, b, c, d] with the plan's
constraints:

    a in (0, a_max)      -- prior-removal strength     (scaled sigmoid; a_max<=1)
    b + c + d = 1        -- source/core/term balance    (softmax)

Kept in fp32 for numerical stability even when the LLM runs in bf16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import PolicyConfig


@dataclass
class BranchHidden:
    """Last-layer hidden states of the four branches at one decoding step.

    Each tensor has shape [batch, llm_hidden_size].
    """

    h_xq: torch.Tensor
    h_sq: torch.Tensor
    h_gq: torch.Tensor
    h_q: torch.Tensor

    def as_features(self, use_contrast: bool) -> torch.Tensor:
        """Build the policy input feature vector x_t.

        Base: [h_XQ ; h_SQ ; h_GQ ; h_Q]
        Contrast (optional): + [h_XQ - h_Q ; h_SQ - h_Q ; h_GQ - h_Q]
        """
        parts = [self.h_xq, self.h_sq, self.h_gq, self.h_q]
        if use_contrast:
            parts += [
                self.h_xq - self.h_q,
                self.h_sq - self.h_q,
                self.h_gq - self.h_q,
            ]
        return torch.cat(parts, dim=-1)


@dataclass
class Weights:
    """PMI combination weights. Each tensor has shape [batch]."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor


class WeightPolicy(nn.Module):
    """MLP mapping branch features -> [a, b, c, d] with plan constraints."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        f, h = config.input_dim, config.hidden_dim

        self.norm = nn.LayerNorm(f)
        self.net = nn.Sequential(
            nn.Linear(f, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Linear(h, 4)  # raw = [u_a, u_b, u_c, u_d]

        self._init_weights()
        self.float()  # policy stays in fp32

    def _init_weights(self) -> None:
        """Warm-start: small last-layer weights, zero bias.

        With zero bias the head outputs ~0, giving a = sigmoid(0) = 0.5 and
        b = c = d = softmax(0,0,0) = 1/3, i.e. a neutral SARA-like start.
        """
        nn.init.normal_(self.head.weight, mean=0.0, std=self.config.init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> Weights:
        """features: [batch, input_dim] -> Weights (each [batch]).

        `a` is learned per token (sigmoid) when ``config.learn_a`` is True,
        otherwise held constant at ``config.fixed_a`` (no gradient flows through
        it, so the head's a-column simply stays at init). `b, c, d` are always
        a learned softmax over the source/core/term branches.
        """
        features = features.float()
        raw = self.head(self.net(self.norm(features)))  # [batch, 4]
        if self.config.learn_a:
            # Scaled sigmoid: a in (0, a_max). a_max < 1 caps prior removal so
            # contrastive decoding cannot tilt the byte-BPE distribution far
            # enough to emit invalid UTF-8 continuations (`�`). a_max=1 is the
            # plain sigmoid (0, 1) -- unchanged behavior.
            a = self.config.a_max * torch.sigmoid(raw[..., 0])
        else:
            a = torch.full_like(raw[..., 0], self.config.fixed_a)  # constant, no grad
        bcd = torch.softmax(raw[..., 1:], dim=-1)  # sums to 1
        return Weights(a=a, b=bcd[..., 0], c=bcd[..., 1], d=bcd[..., 2])

    def entropy(self, weights: Weights) -> torch.Tensor:
        """Exploration/anti-collapse entropy of the weights, per batch item.

        Sum of the [b, c, d] categorical entropy AND the Bernoulli entropy of
        `a`. Covering `a` matters: the categorical entropy alone leaves `a`
        unregularized, so as an exploration bonus it can only stop b/c/d from
        collapsing, not `a` from saturating at its 0/1 boundary. Maximized at
        the neutral start (a=0.5, b=c=d=1/3). (Section 2.5.5 / 2.5.8.)
        """
        p = torch.stack([weights.b, weights.c, weights.d], dim=-1)
        cat_ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
        a = weights.a
        a_ent = -(
            a * torch.log(a.clamp_min(1e-12))
            + (1 - a) * torch.log((1 - a).clamp_min(1e-12))
        )
        return cat_ent + a_ent
