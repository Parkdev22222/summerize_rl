"""Reference-free reward (Section 2.4, 2.5.7).

    R = w1*Faithfulness + w2*TripletCoverage + w3*TermUsage
        + w5*PriorContrast - w4*LengthPenalty

No gold summary is required. Each component is in a bounded, interpretable
range. Faithfulness is pluggable: the default is a dependency-free lexical
proxy; a real NLI/FactKB model can be injected via the FaithfulnessModel
protocol.

PriorContrast is a PMI term supplied by the decoder (mean per-token log-prob
of the chosen tokens under the source-conditioned branches minus under the
query prior Q). It is the only reward component sensitive to the prior-removal
weight `a`, so it is what lets `a` learn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .branches import Triplet
from .config import RewardConfig

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class FaithfulnessModel(Protocol):
    """Score how well `summary` is supported by `source`, in [0, 1]."""

    def score(self, summary: str, source: str) -> float: ...


class LexicalFaithfulness:
    """Dependency-free fallback: fraction of summary tokens grounded in source.

    A precision-style proxy for faithfulness — every content token in the
    summary should be traceable to the source. Real deployments swap in an NLI
    entailment or FactKB model.
    """

    def score(self, summary: str, source: str) -> float:
        summ = tokenize(summary)
        if not summ:
            return 0.0
        src = source.lower()
        # Substring grounding tolerates Korean particle agglutination
        # ("부대" is grounded by "부대가" in the source).
        grounded = sum(1 for tok in summ if tok in src)
        return grounded / len(summ)


@dataclass
class RewardBreakdown:
    faithfulness: float
    coverage: float
    term_usage: float
    contrast: float
    length_penalty: float
    total: float


def _contains(summary_norm: str, phrase: str) -> bool:
    """True if every token of `phrase` appears (as substring) in the summary.

    Substring matching (rather than token-set equality) tolerates Korean
    particle agglutination and multi-token entities/terms.
    """
    toks = tokenize(phrase)
    if not toks:
        return False
    return all(t in summary_norm for t in toks)


def triplet_coverage(summary: str, triplets: list[Triplet]) -> float:
    """Fraction of head/tail entities that surface in the summary."""
    entities = []
    for t in triplets:
        entities.append(t.head)
        entities.append(t.tail)
    if not entities:
        return 1.0  # nothing to cover -> vacuously complete
    summary_norm = summary.lower()
    hit = sum(1 for e in entities if _contains(summary_norm, e))
    return hit / len(entities)


def term_usage(summary: str, active_terms: list[str]) -> float:
    """Fraction of active standard terms correctly used in the summary."""
    if not active_terms:
        return 1.0  # no term to use -> vacuously complete
    summary_norm = summary.lower()
    hit = sum(1 for term in active_terms if _contains(summary_norm, term))
    return hit / len(active_terms)


def length_penalty(summary_tokens: int, config: RewardConfig, summary: str) -> float:
    """Overlength + n-gram repetition penalty (>= 0)."""
    over = max(0, summary_tokens - config.target_length) / max(config.target_length, 1)

    toks = tokenize(summary)
    n = config.repeat_ngram
    if len(toks) >= n:
        ngrams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        repetition = 1.0 - len(set(ngrams)) / len(ngrams)
    else:
        repetition = 0.0
    return over + repetition


def compute_reward(
    summary: str,
    source: str,
    triplets: list[Triplet],
    active_terms: list[str],
    config: RewardConfig,
    faithfulness_model: FaithfulnessModel | None = None,
    summary_length: int | None = None,
    contrast: float = 0.0,
) -> RewardBreakdown:
    fm = faithfulness_model or LexicalFaithfulness()
    faith = float(fm.score(summary, source))
    cov = triplet_coverage(summary, triplets)
    term = term_usage(summary, active_terms)
    n_tokens = summary_length if summary_length is not None else len(tokenize(summary))
    lpen = length_penalty(n_tokens, config, summary)

    total = (
        config.w_faithfulness * faith
        + config.w_coverage * cov
        + config.w_term * term
        + config.w_contrast * contrast
        - config.w_length * lpen
    )
    return RewardBreakdown(
        faithfulness=faith,
        coverage=cov,
        term_usage=term,
        contrast=contrast,
        length_penalty=lpen,
        total=total,
    )


def normalize_rewards(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Standardize rewards within a group: (R - mu) / (sigma + eps)."""
    if not rewards:
        return []
    n = len(rewards)
    mu = sum(rewards) / n
    var = sum((r - mu) ** 2 for r in rewards) / n
    sigma = var ** 0.5
    return [(r - mu) / (sigma + eps) for r in rewards]
