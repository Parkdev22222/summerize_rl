import json

from summarize_rl.branches import Example
from summarize_rl.data import (
    Corpus,
    default_data_path,
    load_examples,
    load_records,
    record_to_example,
)

RECORD = {
    "id": 7,
    "source_text": "### FORCE COMPOSITION:\n#### BLUE FORCE (Galdovia):\n"
    "- **Weapons Systems:** 5 BMP-2 armored vehicles.",
    "summary_text": "블루 포스는 BMP-2 장갑차를 운용한다.",
    "keyfacts": ["블루 포스는 BMP-2를 보유한다."],
    "split": "train",
}


def test_record_to_example_extracts_triplets_and_keeps_reference():
    ex = record_to_example(RECORD)
    assert isinstance(ex, Example)
    assert ex.source == RECORD["source_text"]
    assert ex.reference == RECORD["summary_text"]  # carried for eval only
    assert ex.keyfacts == RECORD["keyfacts"]
    assert ex.id == 7
    assert any(t.relation == "Weapons Systems" for t in ex.triplets)


def test_load_examples_from_file(tmp_path):
    path = tmp_path / "corpus.jsonl"
    rows = [
        dict(RECORD, id=1, split="train"),
        dict(RECORD, id=2, split="eval"),
        dict(RECORD, id=3, split="train"),
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    corpus = load_examples(str(path))
    assert isinstance(corpus, Corpus)
    assert len(corpus.train) == 2
    assert len(corpus.eval) == 1
    assert corpus.eval[0].id == 2


def test_keyfacts_string_is_wrapped():
    ex = record_to_example(dict(RECORD, keyfacts="single fact"))
    assert ex.keyfacts == ["single fact"]


def test_missing_optional_fields_are_tolerated():
    ex = record_to_example({"source_text": "### S:\n- **F:** v."})
    assert ex.reference is None
    assert ex.keyfacts == []
    assert ex.id is None


def test_bundled_corpus_loads_and_splits():
    # The real bundled corpus must be present and split into train/eval.
    records = load_records(default_data_path())
    assert len(records) == 150
    corpus = load_examples()
    assert len(corpus.train) == 140
    assert len(corpus.eval) == 10
    # Every example gets a non-empty triplet set from its structured source.
    assert all(ex.triplets for ex in corpus.train)
    # References (Korean gold summaries) are carried through for eval.
    assert all(ex.reference for ex in corpus.eval)
