"""Extract {"원문", "mine", "base"} for every record from a compare_gemini result.

Reads a compare_gemini output file (either the JSONL you pasted -- a summary line
followed by one record per line -- or the single-JSON {"summary":..,"results":[..]}
format) and writes a compact JSON array with just the source and the two summaries.

Usage:
    python extract_pairs.py compare_result.jsonl pairs.json
    python extract_pairs.py compare_result.json  pairs.json   # also works
"""

import json
import sys


def _records(path):
    """Yield per-example dicts from either a JSON object/array or JSONL file."""
    text = open(path, encoding="utf-8").read().strip()
    # try whole-file JSON first ({"summary":..,"results":[..]} or a bare array)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "results" in obj:
            yield from obj["results"]
            return
        if isinstance(obj, list):
            yield from obj
            return
    except json.JSONDecodeError:
        pass
    # fall back to JSONL: one JSON object per line
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _pick(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python extract_pairs.py <input.json[l]> <output.json>")
    src, dst = sys.argv[1], sys.argv[2]

    pairs = []
    for d in _records(src):
        source = _pick(d, "source", "원문")
        mine = _pick(d, "mine", "llm_mlp_summary")
        base = _pick(d, "base", "llm_summary")
        # skip non-record lines (e.g. the leading {"summary": ...} line)
        if source is None or mine is None or base is None:
            continue
        pairs.append({"원문": source, "mine": mine, "base": base})

    json.dump(pairs, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {len(pairs)} records -> {dst}")


if __name__ == "__main__":
    main()
