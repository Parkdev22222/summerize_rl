from summarize_rl.branches import Triplet, serialize_triplets
from summarize_rl.triplets import extract_triplets

SAMPLE = """### SCENARIO OVERVIEW:
- **Narrative Context:** In 2025, conflict ignites between Galdovia and Metania. It spreads.
- **Time Parameters:** The scenario occurs in late September, around 0600 hours.

### FORCE COMPOSITION:
#### BLUE FORCE (Galdovia):
- **Unit Type/Size/Organization:** 1st Mechanized Infantry Battalion (1,000 personnel).
- **Weapons Systems:** Small arms, 5 BMP-2 armored fighting vehicles.

### TACTICAL PARAMETERS:
- **Potential COA:**
  1. Conduct urban clearing operations.
  2. Establish a cordon around the sector.
"""


def test_extracts_field_triplets():
    triplets = extract_triplets(SAMPLE)
    heads_relations = {(t.head, t.relation) for t in triplets}
    assert ("SCENARIO OVERVIEW", "Narrative Context") in heads_relations
    assert ("SCENARIO OVERVIEW", "Time Parameters") in heads_relations


def test_subsection_becomes_head():
    triplets = extract_triplets(SAMPLE)
    # Parenthetical is trimmed off the head.
    blue = [t for t in triplets if t.head == "BLUE FORCE"]
    relations = {t.relation for t in blue}
    assert "Unit Type/Size/Organization" in relations
    assert "Weapons Systems" in relations


def test_tail_is_first_sentence_only():
    triplets = extract_triplets(SAMPLE)
    narrative = next(t for t in triplets if t.relation == "Narrative Context")
    assert narrative.tail == "In 2025, conflict ignites between Galdovia and Metania"
    assert "It spreads" not in narrative.tail


def test_numbered_options_captured():
    triplets = extract_triplets(SAMPLE)
    options = [t for t in triplets if t.relation == "option"]
    tails = {t.tail for t in options}
    assert "Conduct urban clearing operations." in tails
    assert "Establish a cordon around the sector." in tails


def test_empty_source_yields_no_triplets():
    assert extract_triplets("") == []
    assert extract_triplets("just some prose with no structure") == []


def test_triplets_serialize():
    triplets = extract_triplets(SAMPLE)
    s = serialize_triplets(triplets)
    assert "BLUE FORCE:" in s
    assert isinstance(triplets[0], Triplet)


def test_long_tail_is_truncated():
    long_val = "word " * 100
    src = f"### S:\n- **Field:** {long_val}"
    triplet = extract_triplets(src)[0]
    assert triplet.tail.endswith("…")
    assert len(triplet.tail) <= 161
