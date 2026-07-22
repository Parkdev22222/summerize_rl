"""Military glossary gating (Section 2.3 of the work plan).

The glossary maps a standard term to the triggers (synonyms, colloquialisms,
concepts) that should activate it:

    { "표준용어": ["트리거1", "트리거2", ...] }

Only terms whose trigger appears in the source text are marked *active*. This
prevents ungrounded standard-term injection (hallucination): the GQ branch and
logit boosts only ever consider grounded terms.

String matching is the default. An optional embedding matcher can be injected to
catch paraphrases; it is defined by an interface here and left for later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ActiveTerm:
    """A standard term that was activated, with the trigger(s) that fired it."""

    term: str
    matched_triggers: list[str] = field(default_factory=list)


class SimilarityMatcher(Protocol):
    """Optional embedding-based matcher for paraphrase gating.

    Returns True if `trigger` is semantically present in `text`.
    """

    def matches(self, trigger: str, text: str) -> bool: ...


class Glossary:
    def __init__(
        self,
        entries: dict[str, list[str]],
        *,
        case_insensitive: bool = True,
        similarity_matcher: SimilarityMatcher | None = None,
    ):
        """entries maps standard term -> list of triggers.

        The standard term itself is always treated as one of its own triggers.
        """
        self.case_insensitive = case_insensitive
        self.similarity_matcher = similarity_matcher
        self.entries: dict[str, list[str]] = {}
        for term, triggers in entries.items():
            trig = list(dict.fromkeys([term, *triggers]))  # dedupe, keep order
            self.entries[term] = trig

    def _norm(self, s: str) -> str:
        return s.lower() if self.case_insensitive else s

    def gate(self, text: str) -> list[ActiveTerm]:
        """Return the standard terms whose triggers appear in `text`."""
        norm_text = self._norm(text)
        active: list[ActiveTerm] = []
        for term, triggers in self.entries.items():
            matched = []
            for trig in triggers:
                if self._norm(trig) in norm_text:
                    matched.append(trig)
                elif self.similarity_matcher is not None and (
                    self.similarity_matcher.matches(trig, text)
                ):
                    matched.append(trig)
            if matched:
                active.append(ActiveTerm(term=term, matched_triggers=matched))
        return active

    def active_terms(self, text: str) -> list[str]:
        """Convenience: just the activated standard-term strings."""
        return [t.term for t in self.gate(text)]
