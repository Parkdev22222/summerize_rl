"""Heuristic triplet extraction from structured source reports (Section 2.1).

The corpus has no gold triplets, so we derive the structured `S` branch from the
source text itself. The military scenario reports follow a very regular
markdown-ish layout:

    ### SECTION HEADER:
    #### SUBSECTION HEADER:
    - **Field name:** field value ...
    1. Numbered decision / course-of-action ...

`extract_triplets` walks that structure and emits one `Triplet` per
`- **Field:** value` line:

    head      = current subsection (e.g. "BLUE FORCE") or section if none
    relation  = the bolded field name (e.g. "Weapons Systems")
    tail      = a concise version of the value (first sentence, length-capped)

This is deterministic and dependency-free (no LLM, no parser library), so the
same source always yields the same triplets. It is intentionally simple: the
goal is a useful structured conditioning signal for the SQ branch, not a
perfect semantic KG. Swap this for a real KG/triplet loader when one exists.
"""

from __future__ import annotations

import re

from .branches import Triplet

# "### SCENARIO OVERVIEW:" -> "SCENARIO OVERVIEW"
_SECTION_RE = re.compile(r"^#{1,3}\s+(.*?):?\s*$")
# "#### BLUE FORCE (Galdovia):" -> "BLUE FORCE (Galdovia)"
_SUBSECTION_RE = re.compile(r"^#{4,}\s+(.*?):?\s*$")
# "- **Weapons Systems:** Equipped with ..." -> ("Weapons Systems", "Equipped ...")
_FIELD_RE = re.compile(r"^\s*[-*]\s+\*\*(.+?):?\*\*:?\s*(.*)$")
# "1. Conduct urban clearing ..." (numbered COA / decision items)
_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+(.*)$")
# Plain bullet with no bold field: "* In 2025, tensions escalate ..."
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")

_MAX_TAIL_CHARS = 160


def _clean_head(header: str) -> str:
    """Trim a section/subsection header to a compact entity string.

    "BLUE FORCE (Galdovia)" -> "BLUE FORCE"; keeps the parenthetical off the
    head so serialized triplets stay short, but leaves other headers intact.
    """
    header = header.strip()
    header = re.sub(r"\s*\([^)]*\)\s*$", "", header)  # drop trailing "(...)"
    return header.strip() or header


def _concise_tail(value: str) -> str:
    """Reduce a field value to its first sentence, length-capped."""
    value = value.strip().lstrip("-*").strip()
    # First sentence: split on ". " but keep abbreviations reasonable by only
    # cutting on a period followed by whitespace + capital / end of string.
    m = re.search(r"\.\s", value)
    if m:
        value = value[: m.start()].strip()
    if len(value) > _MAX_TAIL_CHARS:
        value = value[:_MAX_TAIL_CHARS].rsplit(" ", 1)[0].strip() + "…"
    return value


def extract_triplets(source_text: str) -> list[Triplet]:
    """Extract (head, relation, tail) triplets from one structured report.

    Returns an empty list for text with no recognizable fields (the SQ branch
    then serializes to empty, which the decoder handles).
    """
    triplets: list[Triplet] = []
    section = ""
    subsection = ""

    for raw in source_text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        sub = _SUBSECTION_RE.match(line)
        if sub:
            subsection = _clean_head(sub.group(1))
            continue

        sec = _SECTION_RE.match(line)
        if sec:
            section = _clean_head(sec.group(1))
            subsection = ""
            continue

        field = _FIELD_RE.match(line)
        if field:
            relation = field.group(1).strip()
            value = field.group(2).strip()
            if not value:
                continue
            head = subsection or section or "SCENARIO"
            tail = _concise_tail(value)
            if tail:
                triplets.append(Triplet(head=head, relation=relation, tail=tail))
            continue

        num = _NUMBERED_RE.match(line)
        if num:
            head = subsection or section or "SCENARIO"
            tail = _concise_tail(num.group(1))
            if tail:
                triplets.append(Triplet(head=head, relation="option", tail=tail))
            continue

        # Fallback: a plain bullet ("* free prose") with no bold field name.
        # Some reports use free-form bullets instead of "**Field:** value".
        bullet = _BULLET_RE.match(line)
        if bullet:
            head = subsection or section or "SCENARIO"
            tail = _concise_tail(bullet.group(1))
            if tail:
                triplets.append(Triplet(head=head, relation="detail", tail=tail))

    return triplets
