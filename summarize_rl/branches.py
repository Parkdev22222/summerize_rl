"""Four-branch prompt construction and triplet serialization (Section 2.1).

The multi-branch PMI decoder needs four conditioning contexts:

    XQ : source text X + query Q
    SQ : serialized core info S (triplets) + query Q
    GQ : gated military glossary G + query Q
    Q  : query only (the model's prior)

This module builds the text for each branch. Actual tokenization is left to the
LLM backend so this stays model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .glossary import ActiveTerm


@dataclass
class Triplet:
    head: str
    relation: str
    tail: str


@dataclass
class Example:
    """One summarization input.

    `reference` and `keyfacts` are optional metadata (a gold summary and its
    atomic key facts). Training is reference-free and never reads them; they are
    carried only so evaluation/analysis can compare against a reference when the
    corpus provides one.
    """

    source: str  # X: concatenated child-node source text
    triplets: list[Triplet] = field(default_factory=list)
    query: str = "다음 내용을 군사 표준용어를 사용하여 요약하시오."
    reference: str | None = None  # gold summary (eval only, not used in reward)
    keyfacts: list[str] = field(default_factory=list)  # atomic facts (eval only)
    id: int | None = None  # source record id, if any


@dataclass
class BranchTexts:
    xq: str
    sq: str
    gq: str
    q: str

    def as_dict(self) -> dict[str, str]:
        return {"XQ": self.xq, "SQ": self.sq, "GQ": self.gq, "Q": self.q}


def serialize_triplets(triplets: list[Triplet], group_by_head: bool = True) -> str:
    """Serialize triplets into the S branch text.

    Child-grouped format (recommended in the plan): group tails under each head.
        head: relation tail; relation tail
    """
    if not triplets:
        return ""
    if not group_by_head:
        return "\n".join(f"({t.head}, {t.relation}, {t.tail})" for t in triplets)

    grouped: dict[str, list[str]] = {}
    for t in triplets:
        grouped.setdefault(t.head, []).append(f"{t.relation} {t.tail}")
    lines = [f"{head}: {'; '.join(vals)}" for head, vals in grouped.items()]
    return "\n".join(lines)


def format_glossary(active: list[ActiveTerm]) -> str:
    """Format gated glossary terms into the G branch text."""
    if not active:
        return ""
    return "표준용어: " + ", ".join(t.term for t in active)


def build_branches(
    example: Example,
    active_terms: list[ActiveTerm],
    *,
    group_by_head: bool = True,
) -> BranchTexts:
    """Assemble the four branch texts for one example."""
    q = example.query
    s = serialize_triplets(example.triplets, group_by_head=group_by_head)
    g = format_glossary(active_terms)

    xq = f"[원문]\n{example.source}\n\n[지시]\n{q}"
    sq = f"[핵심정보]\n{s}\n\n[지시]\n{q}"
    gq = f"[표준용어]\n{g}\n\n[지시]\n{q}"
    q_only = f"[지시]\n{q}"
    return BranchTexts(xq=xq, sq=sq, gq=gq, q=q_only)
