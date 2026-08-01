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
    """One summarization input."""

    source: str  # X: concatenated child-node source text
    triplets: list[Triplet] = field(default_factory=list)
    query: str = (
        "다음 보고서를 군사 표준용어로 핵심만 간결하게 한 번만 요약하라. "
        "원문에 있는 부대·수치·지명만 사용하고 없는 내용은 지어내지 마라. "
        "보고번호·DTG 같은 서식 머리말은 생략하고 상황·조치·건의 위주로 작성하라."
    )
    # Optional provenance carried from the corpus JSONL. These flow through the
    # trainers untouched but let the weakness-tracking loop (weakness.py) key
    # per-rollout failures by scenario id and mine failure metadata. Defaulted so
    # existing `Example(source, triplets)` call sites stay valid.
    id: str | None = None
    keyfacts: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


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
