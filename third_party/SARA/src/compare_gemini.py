"""Win-rate: my method (LLM+MLP / SARA SAD head) vs a pure LLM (Gemini API).

For every test example this produces two summaries of the same 원문:
  * MINE   -- the trained SARA model (single_infer's decode path, FC weights only)
  * GEMINI -- a pure LLM baseline: the Gemini API summarizing the raw source with
              the SAME instruction MINE gets (no MLP, no context-aware decoding)
then asks Gemini to judge which one summarizes the 원문 better, and tallies a
win rate. Each pair is judged in BOTH orders (candidate positions swapped); a win
counts only when the two orders agree, otherwise it is a tie -- this cancels
position bias. It reuses single_infer.py's model loading + decode helpers.

Setup:
    pip install google-genai        # or: pip install google-generativeai
    export GEMINI_API_KEY=<your key>

Example (from third_party/SARA/src):
    GEMINI_API_KEY=... python compare_gemini.py \
        --model_name_or_path LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --loading_mode bf16 \
        --dataset summarize_rl_ko \
        --save_checkpoint_path ../runs/exaone_ko_ckpts --load_best 1 \
        --plausibility_alpha 0.1 \
        --out ../runs/compare_gemini.jsonl

Caveats:
  * Gemini judges a candidate it also generated -> possible self-preference bias.
    Point --judge_model at a different model to reduce it.
  * MINE sees the source truncated to --max_input_length; GEMINI sees the full
    source. Raise --max_input_length to level the field.
"""

import argparse
import json
import os
import re
import time

from utils import load_dataset, template_input_decoder

from single_infer import (
    add_model_decode_args,
    build_gen_config,
    generate_summary,
    load_model,
    load_tokenizer,
    prepare_example,
)


# -- Gemini adapter (works with either the new or the old google SDK) ---------
def make_gemini(api_key, model_name, temperature=0.0):
    try:
        from google import genai  # new SDK: pip install google-genai

        client = genai.Client(api_key=api_key)

        def call(prompt):
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

    def call(prompt):
        return model.generate_content(
            prompt, generation_config={"temperature": temperature}
        ).text or ""

    return call


def _retry(fn, retries=3, label="gemini"):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # transient API / rate errors -> backoff
            if attempt == retries - 1:
                print(f"   [{label} 오류] {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


# -- pairwise judge -----------------------------------------------------------
_CRITERIA = (
    "채점 기준:\n"
    "1. 부대별 현황의 정확성 — 각 부대의 무장(박격포·기관총·무반동총 등 화기), 총 병력, "
    "부상자·사망자 수치를 원문과 정확히 일치시켰는가.\n"
    "2. 창작(환각) 여부 — 원문에 없는 부대·수치·사건을 지어내지 않았는가.\n"
    "3. 중요 내용 반영 — 핵심 상황과 조치·건의 등 중요한 내용을 빠짐없이 담았는가.\n"
    "4. 간결성 — 군더더기 없이 컴팩트하게 핵심만 표현했는가.\n"
)


def judge_prompt(source, cand1, cand2):
    return (
        "다음 [원문]에 대한 두 요약 후보 중, 아래 채점 기준으로 더 우수한 것 하나를 골라라.\n"
        f"{_CRITERIA}"
        "후보 번호(1 또는 2) 하나만 출력하라. 다른 말은 하지 마라.\n\n"
        f"[원문]\n{source}\n\n[요약 1]\n{cand1}\n\n[요약 2]\n{cand2}\n\n[더 좋은 요약 번호]\n"
    )


def judge_once(call, source, cand1, cand2, retries=3):
    """Return 1 or 2 (better candidate) or None."""
    def _do():
        text = call(judge_prompt(source, cand1, cand2)) or ""
        m = re.search(r"[12]", text)
        return int(m.group()) if m else None
    return _retry(_do, retries=retries, label="judge")


def judge_pair(call, source, mine, gemini, retries=3):
    """Judge both orders; decisive only when they agree. Returns
    'mine' | 'gemini' | 'tie'."""
    a = judge_once(call, source, mine, gemini, retries)    # order A: 1=mine, 2=gemini
    b = judge_once(call, source, gemini, mine, retries)    # order B: 1=gemini, 2=mine
    win_a = {1: "mine", 2: "gemini"}.get(a)
    win_b = {1: "gemini", 2: "mine"}.get(b)
    if win_a and win_a == win_b:
        return win_a
    return "tie"  # disagreement (incl. a genuine draw) or an API failure


def build_args():
    p = argparse.ArgumentParser(
        description="Win-rate of the trained SARA model vs a pure Gemini LLM."
    )
    add_model_decode_args(p)
    p.add_argument("--limit", type=int, default=0, help="cap on #test examples (0 = all)")
    p.add_argument("--gemini_model", default="gemini-2.5-flash")
    p.add_argument("--judge_model", default=None, help="defaults to --gemini_model")
    p.add_argument("--gemini_temperature", type=float, default=0.0)
    p.add_argument("--out", default=None, help="optional JSONL of per-example results")
    return p.parse_args()


def main():
    args = build_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY)를 설정하세요.")
    summarizer_call = make_gemini(api_key, args.gemini_model, args.gemini_temperature)
    judge_call = make_gemini(api_key, args.judge_model or args.gemini_model, 0.0)

    _, _, test_set = load_dataset(args.dataset, args.data_type)
    if not test_set:
        raise SystemExit("test split is empty")
    n = len(test_set) if args.limit <= 0 else min(args.limit, len(test_set))

    tokenizer = load_tokenizer(args)
    model = load_model(args)
    gen_cfg = build_gen_config(args, tokenizer)
    device = "cuda" if _cuda_available() else "cpu"

    tally = {"mine": 0, "gemini": 0, "tie": 0}
    rows_out = []
    for i in range(n):
        split_row = test_set[i]
        source = split_row[0]  # full raw 원문 (judge + Gemini see this)

        # MINE: trained SARA decode (may truncate source to --max_input_length)
        row, meta = prepare_example(split_row, tokenizer, args)
        mine, _ = generate_summary(model, tokenizer, gen_cfg, row, args, device)

        # GEMINI: pure LLM summarizing the raw 원문 with the SAME instruction
        gemini = _retry(
            lambda: summarizer_call(template_input_decoder(split_row, args.dataset)),
            label="gemini-summary",
        ) or ""

        winner = judge_pair(judge_call, source, mine, gemini) if gemini.strip() else "tie"
        tally[winner] += 1
        decided = tally["mine"] + tally["gemini"]
        wr = 100.0 * tally["mine"] / decided if decided else 0.0
        print(f"[{i + 1}/{n}] winner={winner:6s} | "
              f"mine {tally['mine']} / gemini {tally['gemini']} / tie {tally['tie']} "
              f"| running win-rate {wr:.1f}%")

        rows_out.append({
            "index": i, "winner": winner,
            "mine": mine, "gemini": gemini,
            "gold": meta["reference"], "source": source,
        })

    decided = tally["mine"] + tally["gemini"]
    win_rate = 100.0 * tally["mine"] / decided if decided else 0.0
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"판정 완료: {n} examples  (model={args.gemini_model}, judge={args.judge_model or args.gemini_model})")
    print(f"MINE 승 {tally['mine']}  |  GEMINI 승 {tally['gemini']}  |  무승부 {tally['tie']}")
    print(f"내 방법 승률 (무승부 제외): {win_rate:.1f}%   ({tally['mine']}/{decided})")
    print(bar)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "summary": {"n": n, **tally, "win_rate_excl_tie": win_rate},
                "gemini_model": args.gemini_model,
                "judge_model": args.judge_model or args.gemini_model,
            }, ensure_ascii=False) + "\n")
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[out] per-example results -> {args.out}")


def _cuda_available():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
