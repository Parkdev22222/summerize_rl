"""Merge per-batch agent outputs (data/parts/*.jsonl) into the final dataset.

Each part line must be an object with keys:
    id (int), source_text (str, Korean), source_text_en (str, English original),
    summary_text (str), keyfacts (list[str]), triplets (list[[h,r,t]]), split (str)

Usage:
    python data/build_dataset.py            # build + validate
    python data/build_dataset.py --check     # validate only, no write
"""

from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS_GLOB = os.path.join(HERE, "parts", "*.jsonl")
OUT_PATH = os.path.join(HERE, "scenarios_ko.jsonl")

REQUIRED_KEYS = {
    "id": int,
    "source_text": str,
    "source_text_en": str,
    "summary_text": str,
    "keyfacts": list,
    "triplets": list,
    "split": str,
}


def load_parts() -> list[dict]:
    records: list[dict] = []
    for path in sorted(glob.glob(PARTS_GLOB)):
        with open(path, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"[{path}:{ln}] bad JSON: {e}")
    return records


def validate(records: list[dict]) -> None:
    errors: list[str] = []
    seen: dict[int, int] = {}
    for r in records:
        rid = r.get("id")
        for key, typ in REQUIRED_KEYS.items():
            if key not in r:
                errors.append(f"id={rid}: missing key '{key}'")
            elif not isinstance(r[key], typ):
                errors.append(f"id={rid}: key '{key}' expected {typ.__name__}, got {type(r[key]).__name__}")
        if isinstance(rid, int):
            seen[rid] = seen.get(rid, 0) + 1
        # triplet shape
        for i, t in enumerate(r.get("triplets", [])):
            if not (isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) and x.strip() for x in t)):
                errors.append(f"id={rid}: triplet #{i} not a [head,relation,tail] of 3 non-empty strings: {t!r}")
        n = len(r.get("triplets", []))
        if isinstance(r.get("triplets"), list) and not (5 <= n <= 25):
            errors.append(f"id={rid}: triplet count {n} outside expected 10-20-ish range")
        if not r.get("keyfacts"):
            errors.append(f"id={rid}: empty keyfacts")
        if not (r.get("source_text") or "").strip():
            errors.append(f"id={rid}: empty source_text (Korean)")
    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        errors.append(f"duplicate ids: {dups}")
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)


def main() -> None:
    check_only = "--check" in sys.argv
    records = load_parts()
    records.sort(key=lambda r: r.get("id", 1 << 30))
    validate(records)
    ids = [r["id"] for r in records]
    missing = sorted(set(range(min(ids), max(ids) + 1)) - set(ids)) if ids else []
    tcounts = [len(r["triplets"]) for r in records]
    print(f"records: {len(records)}  ids: {min(ids)}..{max(ids)}")
    if missing:
        print(f"MISSING ids: {missing}")
    print(f"triplets/record: min={min(tcounts)} max={max(tcounts)} avg={sum(tcounts)/len(tcounts):.1f}")
    if check_only:
        return
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
