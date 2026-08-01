"""load_corpus must carry scenario id/keyfacts/split metadata onto Example.

The weakness-tracking loop keys per-rollout failures by scenario id and mines
failure metadata (keyfacts, split), so the corpus loaders must stop discarding
those JSONL fields.
"""

import json

from examples.train_real import load_corpus


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_load_corpus_carries_id_keyfacts_split(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [
        {
            "id": 10000,
            "source_text": "원문 본문",
            "keyfacts": ["핵심1", "핵심2"],
            "triplets": [["1중대", "규모", "120명"]],
            "split": "train",
        }
    ])
    examples = load_corpus(str(p), query=None, limit=None)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.id == "10000"
    assert ex.keyfacts == ["핵심1", "핵심2"]
    assert ex.meta.get("split") == "train"
    assert ex.source == "원문 본문"
    assert ex.triplets[0].tail == "120명"


def test_load_corpus_tolerates_missing_optional_fields(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{"source_text": "본문만", "triplets": []}])
    ex = load_corpus(str(p), query=None, limit=None)[0]
    assert ex.id is None
    assert ex.keyfacts == []
    assert ex.meta.get("split") is None
