"""Rank RAW / NEUTRAL / TRAINED summaries with the Gemini API over the corpus.

For every source record this builds the SAME three summaries that
``examples.diagnose`` prints — RAW (frozen LLM alone), NEUTRAL (untrained
policy), TRAINED (loaded checkpoint) — then asks the Gemini API which one best
summarizes the source, and tallies how often each wins. Reports how many of the
records the TRAINED model won (its "1st-place" count).

Candidate order is shuffled per record so Gemini can't win by position; the
winning number is mapped back to which model produced it.

Setup (Gemini):
    pip install google-genai        # or: pip install google-generativeai
    export GEMINI_API_KEY=<your key>

Example:
    GEMINI_API_KEY=... CUDA_VISIBLE_DEVICES=0 uv run -m examples.eval_gemini \
        --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
        --ckpt checkpoints/best.pt --trust-remote-code

This file only READS the library; it does not modify any existing module.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import torch

from summarize_rl.branches import Example, Triplet, build_branches
from summarize_rl.glossary import Glossary
from summarize_rl.infer import Summarizer, build_hf_summarizer
from summarize_rl.policy import WeightPolicy

LABELS = ("RAW", "NEUTRAL", "TRAINED")


# -- Gemini adapter (works with either the new or the old SDK) ----------------
def make_gemini(api_key: str, model_name: str):
    """Return call(prompt)->str using whichever google SDK is installed."""
    try:
        from google import genai  # new SDK: pip install google-genai

        client = genai.Client(api_key=api_key)

        def call(prompt: str) -> str:
            resp = client.models.generate_content(model=model_name, contents=prompt)
            return resp.text or ""

        return call
    except ImportError:
        pass
    import google.generativeai as genai  # old SDK: pip install google-generativeai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    def call(prompt: str) -> str:
        return model.generate_content(prompt).text or ""

    return call


def gemini_prompt(source: str, ordered: list[tuple[str, str]]) -> str:
    body = "\n".join(f"[요약 {i + 1}]\n{text}\n" for i, (_lab, text) in enumerate(ordered))
    return (
        "다음 [원문]에 대한 세 개의 요약 후보 중, 원문을 가장 정확하고 충실하게 "
        "요약한 것 하나만 골라라.\n"
        "후보 번호(1, 2, 3) 하나만 출력하라. 다른 말은 하지 마라.\n\n"
        f"[원문]\n{source}\n\n{body}\n[가장 좋은 요약 번호]\n"
    )


def gemini_pick(call, source: str, ordered: list[tuple[str, str]], retries: int = 3):
    """Return 1..3 (best candidate) or None on failure."""
    prompt = gemini_prompt(source, ordered)
    for attempt in range(retries):
        try:
            text = call(prompt) or ""
            m = re.search(r"[123]", text)
            return int(m.group()) if m else None
        except Exception as e:  # transient API/rate errors -> backoff and retry
            if attempt == retries - 1:
                print(f"   [gemini 오류] {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


# -- three summaries per record (same as examples.diagnose) -------------------
def three_summaries(example, backend, cfg, glossary, trained, neutral) -> dict:
    active = glossary.gate(example.source) if glossary else []
    branch_texts = build_branches(example, active).as_dict()
    raw = backend.generate_text(branch_texts["XQ"], cfg.decode.max_new_tokens).strip()
    neu = neutral.summarize(example.source, triplets=example.triplets).text.strip()
    tr = trained.summarize(example.source, triplets=example.triplets).text.strip()
    return {"RAW": raw, "NEUTRAL": neu, "TRAINED": tr}


def load_corpus(path: str, limit: int | None) -> list[Example]:
    examples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            examples.append(Example(
                source=rec["source_text"],
                triplets=[Triplet(*t) for t in rec.get("triplets", []) if len(t) == 3],
            ))
            if limit and len(examples) >= limit:
                break
    return examples


def load_glossary(path: str | None):
    if path:
        with open(path, encoding="utf-8") as fh:
            return Glossary(json.load(fh))
    from examples.sample_data import MILITARY_GLOSSARY

    return MILITARY_GLOSSARY


def main() -> None:
    p = argparse.ArgumentParser(description="Gemini-judged RAW/NEUTRAL/TRAINED win-rate over the corpus.")
    p.add_argument("--model", required=True, help="HF model for the frozen backbone")
    p.add_argument("--ckpt", required=True, help="trained checkpoint (checkpoints/best.pt)")
    p.add_argument("--data", default="data/scenarios_ko.jsonl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=False)
    p.add_argument("--no-chat-template", dest="use_chat_template", action="store_false", default=True)
    p.add_argument("--glossary", default=None)
    p.add_argument("--limit", type=int, default=None, help="use only the first N records (smoke)")
    p.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini model id")
    p.add_argument("--sleep", type=float, default=0.5, help="seconds between Gemini calls (rate limit)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional path to write per-record results as JSONL")
    args = p.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY)를 설정하세요.")
    call = make_gemini(api_key, args.gemini_model)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    glossary = load_glossary(args.glossary)
    trained, cfg, step = build_hf_summarizer(
        args.model, args.ckpt, glossary=glossary,
        device=args.device, dtype=args.dtype,
        attn_implementation=args.attn, trust_remote_code=args.trust_remote_code,
        use_chat_template=args.use_chat_template,
    )
    backend = trained.backend
    dev = getattr(backend, "device", "cpu")
    neutral = Summarizer(backend, WeightPolicy(cfg.policy).to(dev), cfg, glossary=glossary)

    examples = load_corpus(args.data, args.limit)
    print(f"model={args.model} ckpt={args.ckpt} (step={step}) "
          f"gemini={args.gemini_model} records={len(examples)}\n")

    tally = {lab: 0 for lab in LABELS}
    undecided = 0
    out_fh = open(args.out, "w", encoding="utf-8") if args.out else None

    for i, example in enumerate(examples):
        summ = three_summaries(example, backend, cfg, glossary, trained, neutral)
        ordered = list(summ.items())  # [(label, text), ...]
        random.shuffle(ordered)  # de-bias position
        pick = gemini_pick(call, example.source, ordered)
        if pick is None:
            undecided += 1
            winner = None
        else:
            winner = ordered[pick - 1][0]
            tally[winner] += 1
        if out_fh:
            out_fh.write(json.dumps({
                "index": i, "winner": winner,
                "order": [lab for lab, _ in ordered],
            }, ensure_ascii=False) + "\n")
        done = i + 1
        print(f"[{done}/{len(examples)}] winner={winner or '판정불가'}  "
              f"누적 RAW={tally['RAW']} NEUTRAL={tally['NEUTRAL']} TRAINED={tally['TRAINED']} "
              f"(undecided={undecided})")
        time.sleep(args.sleep)

    if out_fh:
        out_fh.close()

    n = len(examples)
    judged = n - undecided
    print("\n================ 결과 ================")
    for lab in LABELS:
        pct = 100 * tally[lab] / judged if judged else 0.0
        print(f"  {lab:8s} 1등 {tally[lab]:4d}회  ({pct:5.1f}%)")
    print(f"  판정불가            {undecided:4d}회")
    print(f"\n★ TRAINED 모델 1등: {tally['TRAINED']}/{n} "
          f"(판정된 {judged}건 중 {100 * tally['TRAINED'] / judged if judged else 0:.1f}%)")


if __name__ == "__main__":
    main()
