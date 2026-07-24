"""Reference-free reward (Section 2.4, 2.5.7).

    R = w1*Faithfulness + w2*TripletCoverage + w3*TermUsage
        + w7*KeySentenceCoverage + w5*PriorContrast
        - w4*LengthPenalty - w6*ExtractiveCopy

No gold summary is required. Each component is in a bounded, interpretable
range. Faithfulness is pluggable: the default is a dependency-free lexical
proxy; a real NLI/FactKB model can be injected via the FaithfulnessModel
protocol.

PriorContrast is a PMI term supplied by the decoder (mean per-token log-prob
of the chosen tokens under the source-conditioned branches minus under the
query prior Q). It is the only reward component sensitive to the prior-removal
weight `a`, so it is what lets `a` learn.

Anti-reward-hacking (why these terms exist): a precision-only faithfulness plus
a coverage/term reward that a source-*copy* also satisfies makes verbatim
copying the reward optimum. RL then routes all weight to the source branch
(b->1) and abandons the triplet/glossary branches. ExtractiveCopy penalizes
long verbatim spans, and TermUsage only credits standard terms that are NOT
already in the source (genuine standardization the glossary branch must add),
so the multi-branch mixture is worth using.
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
    """Dependency-free fallback: fraction of summary *content* tokens grounded.

    A precision-style proxy for faithfulness — every content token in the
    summary should be traceable to the source. Single-character tokens (Korean
    particles/fragments, punctuation) are dropped: on a long report almost any
    1-char token is a trivial substring, which let generic military text score
    high without being about the source. Real deployments swap in an NLI
    entailment or FactKB model via the FaithfulnessModel protocol.
    """

    def score(self, summary: str, source: str) -> float:
        summ = [t for t in tokenize(summary) if len(t) >= 2]
        if not summ:  # very short summary: fall back to all tokens
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
    key_sentence: float
    contrast: float
    length_penalty: float
    copy_penalty: float
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


def term_usage(summary: str, active_terms: list[str], source: str | None = None) -> float:
    """Fraction of *standardization-worthy* active terms used in the summary.

    When `source` is given, only credit standard terms that are NOT already
    verbatim in the source: those are the ones the glossary (GQ) branch has to
    inject, so copying the source cannot earn this reward. Without `source`,
    falls back to crediting every active term (legacy behavior).
    """
    if not active_terms:
        return 1.0  # no term to use -> vacuously complete
    if source is not None:
        src_norm = source.lower()
        target = [t for t in active_terms if not _contains(src_norm, t)]
        if not target:
            return 1.0  # every active term already in source -> nothing to add
    else:
        target = active_terms
    summary_norm = summary.lower()
    hit = sum(1 for term in target if _contains(summary_norm, term))
    return hit / len(target)


def key_sentence_coverage(summary: str, key_sentences: list[str]) -> float:
    """How much of the LLM-picked key sentences the summary reflects, in [0, 1].

    For each key sentence, the fraction of its content tokens (>=2 chars) that
    appear in the summary; averaged over the key sentences. A recall-style
    anchor to the source's *salient* content (complements triplet coverage,
    which only checks structured entities). Vacuously 1.0 if no key sentences.
    """
    if not key_sentences:
        return 1.0
    summary_norm = summary.lower()
    scores = []
    for sent in key_sentences:
        toks = [t for t in tokenize(sent) if len(t) >= 2]
        if not toks:
            continue
        grounded = sum(1 for t in toks if t in summary_norm)
        scores.append(grounded / len(toks))
    return sum(scores) / len(scores) if scores else 1.0


def extractive_copy(summary: str, source: str, n: int = 4) -> float:
    """Fraction of summary n-grams copied verbatim from the source, in [0, 1].

    Anti-copy signal: near 1.0 when the summary is a verbatim source span
    (the degenerate solution a source-only b->1 policy produces), near 0 when
    the summary recombines/abstracts. Token-level n-grams (same tokenizer for
    both sides) so it is language-agnostic and robust to spacing.
    """
    s = tokenize(summary)
    src = tokenize(source)
    if len(s) < n or len(src) < n:
        return 0.0
    src_ngrams = {tuple(src[i : i + n]) for i in range(len(src) - n + 1)}
    s_ngrams = [tuple(s[i : i + n]) for i in range(len(s) - n + 1)]
    copied = sum(1 for g in s_ngrams if g in src_ngrams)
    return copied / len(s_ngrams)


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
    key_sentences: list[str] | None = None,
) -> RewardBreakdown:
    fm = faithfulness_model or LexicalFaithfulness()
    faith = float(fm.score(summary, source))
    cov = triplet_coverage(summary, triplets)
    term = term_usage(summary, active_terms, source=source)
    n_tokens = summary_length if summary_length is not None else len(tokenize(summary))
    lpen = length_penalty(n_tokens, config, summary)
    copy = extractive_copy(summary, source, n=config.copy_ngram)
    # key_sentences is None when disabled/unavailable -> no contribution (0).
    keysent = key_sentence_coverage(summary, key_sentences) if key_sentences else 0.0

    total = (
        config.w_faithfulness * faith
        + config.w_coverage * cov
        + config.w_term * term
        + config.w_keysent * keysent
        + config.w_contrast * contrast
        - config.w_length * lpen
        - config.w_copy * copy
    )
    return RewardBreakdown(
        faithfulness=faith,
        coverage=cov,
        term_usage=term,
        key_sentence=keysent,
        contrast=contrast,
        length_penalty=lpen,
        copy_penalty=copy,
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
