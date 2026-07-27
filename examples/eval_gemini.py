"""Rank RAW / NEUTRAL / TRAINED summaries with the Gemini API over the corpus.

For every source record this builds the SAME three summaries that
``examples.diagnose`` prints — RAW (frozen LLM alone), NEUTRAL (untrained
policy), TRAINED (loaded checkpoint) — and judges them two ways (hybrid):

  1. Comparative — asks Gemini which single summary is best and tallies each
     model's 1st-place count (robust headline metric, little scale bias).
  2. Per-dimension — asks Gemini to score every candidate 1-5 on accuracy
     (unit status: weapons / personnel / casualties), no-fabrication, salience,
     and brevity, then reports each model's mean per dimension (diagnostic:
     shows WHERE trained wins/loses). Disable with --no-scores.

Candidate order is shuffled per record so Gemini can't win by position; the
winning number / scores are mapped back to which model produced each.

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
def make_gemini(api_key: str, model_name: str, temperature: float = 0.0):
    """Return call(prompt)->str using whichever google SDK is installed.

    ``temperature`` is fixed (default 0) so the judge is reproducible across runs
    and across the pick/score calls.
    """
    try:
        from google import genai  # new SDK: pip install google-genai

        client = genai.Client(api_key=api_key)

        def call(prompt: str) -> str:
            resp = client.models.generate_content(
                model=model_name, contents=prompt,
                config={"temperature": temperature},
            )
            return resp.text or ""

        return call
    except ImportError:
        pass
    import google.generativeai as genai  # old SDK: pip install google-generativeai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    def call(prompt: str) -> str:
        return model.generate_content(
            prompt, generation_config={"temperature": temperature}
        ).text or ""

    return call


def gemini_prompt(source: str, ordered: list[tuple[str, str]]) -> str:
    body = "\n".join(f"[요약 {i + 1}]\n{text}\n" for i, (_lab, text) in enumerate(ordered))
    return (
        "다음 [원문]에 대한 세 개의 요약 후보 중, 아래 채점 기준으로 가장 우수한 것 "
        "하나만 골라라.\n"
        "채점 기준:\n"
        "1. 부대별 현황의 정확성 — 각 부대의 무장(박격포·기관총·무반동총 등 화기), 총 "
        "병력(인원 규모), 부상자·사망자 등 사상자 수치를 원문과 정확히 일치시켰는가.\n"
        "2. 창작(환각) 여부 — 원문에 없는 부대·수치·사건을 지어내지 않았는가.\n"
        "3. 중요 내용 반영 — 핵심 상황과 조치·건의 등 중요한 내용을 빠짐없이 담았는가.\n"
        "4. 간결성 — 군더더기 없이 컴팩트하게 핵심만 표현했는가.\n"
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


# -- per-dimension absolute scoring (hybrid: complements the comparative pick) -
# English keys are asked of the judge (robust JSON); Korean labels are for the
# report. brevity vs salience deliberately stay separate (they trade off).
DIMS = [
    ("accuracy", "정확성"),
    ("no_fabrication", "창작없음"),
    ("salience", "중요내용"),
    ("brevity", "간결성"),
]


def gemini_score_prompt(source: str, ordered: list[tuple[str, str]]) -> str:
    body = "\n".join(f"[요약 {i + 1}]\n{text}\n" for i, (_lab, text) in enumerate(ordered))
    return (
        "다음 [원문]에 대한 세 요약 후보를 각 항목마다 1~5점으로 채점하라 "
        "(5=매우 우수, 1=매우 미흡).\n"
        "채점 항목:\n"
        "- accuracy: 각 부대의 무장(박격포·기관총·무반동총 등)·총 병력·부상자/사망자 "
        "수치를 원문과 정확히 일치시켰는가.\n"
        "- no_fabrication: 원문에 없는 부대·수치·사건을 지어내지 않았는가(창작이 없을수록 높음).\n"
        "- salience: 핵심 상황과 조치·건의 등 중요한 내용을 빠짐없이 담았는가.\n"
        "- brevity: 군더더기 없이 컴팩트하게 핵심만 표현했는가.\n"
        "먼저 각 후보를 한두 문장으로 간단히 평가한 뒤, 맨 마지막 줄에 아래 형식의 JSON "
        "하나만 출력하라:\n"
        '{"1":{"accuracy":n,"no_fabrication":n,"salience":n,"brevity":n},'
        '"2":{...},"3":{...}}\n\n'
        f"[원문]\n{source}\n\n{body}\n[채점 결과]\n"
    )


def _parse_scores(text: str | None) -> dict | None:
    """Pull the {"1":{...},"2":{...},"3":{...}} JSON out of the judge's text.

    Tolerates a chain-of-thought preamble before the JSON (takes the span from
    the first '{' to the last '}'). Returns {cand_idx(1-3): {dim: score}} with
    each score clamped to [1,5], or None if it can't be parsed/validated.
    """
    if not text or "{" not in text or "}" not in text:
        return None
    blob = text[text.find("{"): text.rfind("}") + 1]
    try:
        obj = json.loads(blob)
    except Exception:
        return None
    out: dict[int, dict[str, float]] = {}
    for k in ("1", "2", "3"):
        d = obj.get(k)
        if not isinstance(d, dict):
            return None
        dd: dict[str, float] = {}
        for dim, _lab in DIMS:
            v = d.get(dim)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return None
            dd[dim] = max(1.0, min(5.0, float(v)))
        out[int(k)] = dd
    return out


def gemini_scores(call, source: str, ordered: list[tuple[str, str]], retries: int = 3):
    """Return {cand_idx: {dim: score}} for the 3 candidates, or None on failure."""
    prompt = gemini_score_prompt(source, ordered)
    for attempt in range(retries):
        try:
            parsed = _parse_scores(call(prompt) or "")
            if parsed is not None:
                return parsed
        except Exception as e:  # transient API/rate errors
            if attempt == retries - 1:
                print(f"   [gemini 점수 오류] {e}")
                return None
        if attempt < retries - 1:
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
    p = argparse.ArgumentParser(description="Gemini-judged RAW/NEUTRAL/TRAINED: comparative win-rate + per-dimension scores.")
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
    p.add_argument("--no-scores", dest="scores", action="store_false", default=True,
                   help="disable the per-dimension 1-5 scoring pass (keep only the comparative "
                        "win-count). Scoring adds one extra Gemini call per record.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Gemini decoding temperature (default 0 for reproducible judging).")
    args = p.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY)를 설정하세요.")
    call = make_gemini(api_key, args.gemini_model, temperature=args.temperature)

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
    # per-model, per-dimension score accumulators (hybrid: absolute rubric)
    dim_sums = {lab: {dim: 0.0 for dim, _ in DIMS} for lab in LABELS}
    dim_counts = {lab: 0 for lab in LABELS}
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

        # per-dimension absolute scores (second, independent Gemini call)
        rec_scores = None
        if args.scores:
            time.sleep(args.sleep)  # space out the two calls per record
            sc = gemini_scores(call, example.source, ordered)
            if sc is not None:
                rec_scores = {}
                for idx in (1, 2, 3):
                    lab = ordered[idx - 1][0]
                    rec_scores[lab] = sc[idx]
                    for dim, _l in DIMS:
                        dim_sums[lab][dim] += sc[idx][dim]
                    dim_counts[lab] += 1

        if out_fh:
            out_fh.write(json.dumps({
                "index": i, "winner": winner,
                "order": [lab for lab, _ in ordered],
                "scores": rec_scores,
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
    print("\n=========== 비교형 승률 (누가 1등) ===========")
    for lab in LABELS:
        pct = 100 * tally[lab] / judged if judged else 0.0
        print(f"  {lab:8s} 1등 {tally[lab]:4d}회  ({pct:5.1f}%)")
    print(f"  판정불가            {undecided:4d}회")
    print(f"\n★ TRAINED 모델 1등: {tally['TRAINED']}/{n} "
          f"(판정된 {judged}건 중 {100 * tally['TRAINED'] / judged if judged else 0:.1f}%)")

    if args.scores and any(dim_counts.values()):
        labels_ko = [lab for _d, lab in DIMS]
        header = "  " + f"{'모델':8s}" + "".join(f"{h:>9s}" for h in labels_ko) + f"{'(n)':>7s}"
        print("\n=========== 항목별 평균 점수 (1~5) ===========")
        print(header)
        for lab in LABELS:
            c = dim_counts[lab]
            cells = "".join(
                f"{(dim_sums[lab][dim] / c):>9.2f}" if c else f"{'-':>9s}"
                for dim, _l in DIMS
            )
            print(f"  {lab:8s}{cells}{c:>7d}")
        print("  (정확성·창작없음·중요내용은 높을수록, 간결성은 높을수록 컴팩트)")


if __name__ == "__main__":
    main()
