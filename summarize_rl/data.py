"""Corpus loader for the military scenario JSONL (Section 6, data wiring).

Each JSONL record has the shape:

    {
      "id": 0,
      "source_text": "### SCENARIO OVERVIEW: ...",   # X: the source report
      "summary_text": "2025년 9월 말, ...",            # gold summary (Korean)
      "keyfacts": ["...", "..."],                       # atomic facts (Korean)
      "split": "train" | "eval"
    }

`load_examples` turns each record into a training `Example`:

  * source     <- source_text
  * triplets   <- extract_triplets(source_text)   (no gold triplets in corpus)
  * reference  <- summary_text                     (eval only; reward is
                                                    reference-free and ignores it)
  * keyfacts   <- keyfacts                          (eval only)

The `split` field partitions the corpus into train/eval so the training driver
does not have to guess a hold-out.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .branches import Example
from .triplets import extract_triplets

# Default corpus location: <repo>/data/military_scenarios.jsonl
_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "military_scenarios.jsonl",
)


def default_data_path() -> str:
    """Absolute path to the bundled military scenario corpus."""
    return _DEFAULT_PATH


@dataclass
class Corpus:
    """Train/eval split of `Example`s built from the JSONL corpus."""

    train: list[Example]
    eval: list[Example]

    def __iter__(self):
        return iter(self.train + self.eval)


def record_to_example(record: dict) -> Example:
    """Build one `Example` from a raw JSONL record (triplets extracted here)."""
    source = record["source_text"]
    keyfacts = record.get("keyfacts") or []
    if isinstance(keyfacts, str):
        keyfacts = [keyfacts]
    return Example(
        source=source,
        triplets=extract_triplets(source),
        reference=record.get("summary_text"),
        keyfacts=list(keyfacts),
        id=record.get("id"),
    )


def load_records(path: str | None = None) -> list[dict]:
    """Read the raw JSONL records (one JSON object per line)."""
    path = path or _DEFAULT_PATH
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_examples(
    path: str | None = None,
    *,
    train_split: str = "train",
    eval_split: str = "eval",
) -> Corpus:
    """Load the corpus and split into train/eval `Example`s by the `split` field.

    Records whose split matches neither name are dropped. Triplets are extracted
    from each `source_text` at load time.
    """
    train: list[Example] = []
    eval_: list[Example] = []
    for record in load_records(path):
        example = record_to_example(record)
        split = record.get("split", train_split)
        if split == eval_split:
            eval_.append(example)
        elif split == train_split:
            train.append(example)
    return Corpus(train=train, eval=eval_)
