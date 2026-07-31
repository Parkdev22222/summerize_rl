"""Win-rate: my method (EXAONE + MLP / SARA) vs the raw EXAONE (pure LLM).

Both summaries come from the SAME local backbone you load here -- the only
difference is my method's trained MLP + context-aware decoding:
  * MINE -- EXAONE + trained SAD/FC head, 3-branch context-aware decode (single_infer)
  * BASE -- the raw EXAONE alone: same prompt, plain single-branch generation,
            no MLP, no context-aware combination (the pure-LLM baseline)
Gemini is only the JUDGE: for each test example it decides which summary better
summarizes the 원문, and we tally a win rate. Each pair is judged in BOTH orders
(candidate positions swapped); a win counts only when the two orders agree,
otherwise tie -- this cancels position bias. Since neither summary is Gemini's
own, the judge has no self-preference bias.

Setup (judge only):
    pip install google-genai        # or: pip install google-generativeai
    export GEMINI_API_KEY=<your key>

Example (from third_party/SARA/src):
    GEMINI_API_KEY=... python compare_gemini.py \
        --model_name_or_path LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --loading_mode bf16 \
        --dataset summarize_rl_ko \
        --save_checkpoint_path ../runs/exaone_ko_ckpts --load_best 1 \
        --plausibility_alpha 0.1 \
        --out ../runs/compare_gemini.jsonl

Note: MINE additionally gets keyfacts via the presumm branch; BASE sees only the
report prompt. So this compares the full method against the plain backbone, not
the MLP in isolation.
"""

import argparse
import copy
import json
import os
import re
import time

from utils import load_dataset

from single_infer import (
    add_model_decode_args,
    build_gen_config,
    generate_summary,
    load_model,
    load_tokenizer,
    prepare_example,
)


# -- Gemini adapter (judge only; works with either google SDK) ----------------
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
    "채점 기준 (반드시 우선순위 순서대로 적용한다. 상위 기준에서 분명한 차이가 나면 "
    "그것만으로 승부를 정하고, 두 요약이 사실상 대등할 때에만 다음 기준으로 내려간다):\n"
    "1순위 — 사실 정확성 (가장 중요, 가장 크게 감점). 다음 두 가지를 최우선으로 본다:\n"
    "   (a) 환각: 원문에 없는 부대·장비·사건·수치를 지어냈는가.\n"
    "   (b) 수치 오류: 장비 수, 총 병력, 부상자·사망자 등 수치를 원문과 다르게 적었는가.\n"
    "   이런 오류(환각·수치 오류)가 더 적은 요약을 크게 우대하고, 하나라도 있으면 강하게 감점한다.\n"
    "2순위 — 핵심 내용 포함. 원문의 중요한 상황·조치·건의 등 핵심 문장을 빠뜨렸는가. "
    "누락이 적은 요약이 우수하다.\n"
    "3순위 — 간결성·가독성 (위 두 기준이 대등할 때만 고려). 군더더기 없이 핵심만 명료하게 "
    "표현했는가.\n"
)


def judge_prompt(source, cand1, cand2):
    return (
        "다음 [원문]에 대한 두 요약 후보 중, 아래 채점 기준을 우선순위대로 적용해 더 우수한 것 "
        "하나를 골라라.\n"
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


def judge_pair(call, source, mine, base, retries=3):
    """Judge both orders; decisive only when they agree. Returns
    'mine' | 'base' | 'tie'."""
    a = judge_once(call, source, mine, base, retries)    # order A: 1=mine, 2=base
    b = judge_once(call, source, base, mine, retries)    # order B: 1=base, 2=mine
    win_a = {1: "mine", 2: "base"}.get(a)
    win_b = {1: "base", 2: "mine"}.get(b)
    if win_a and win_a == win_b:
        return win_a
    return "tie"  # disagreement (incl. a genuine draw) or an API failure


def rouge_scores(evaluator, pred, gold):
    """Per-example ROUGE-1/2/Lsum F + their sum (added_results, the training's
    model-selection metric). Empty prediction -> all zeros."""
    if not pred or not pred.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0, "added": 0.0}
    d = evaluator.calculate_rouge([pred], [gold])
    r1 = float(d.get("rouge1_fmeasure", 0.0))
    r2 = float(d.get("rouge2_fmeasure", 0.0))
    rl = float(d.get("rougeLsum_fmeasure", 0.0))
    return {"rouge1": r1, "rouge2": r2, "rougeLsum": rl, "added": r1 + r2 + rl}


def build_args():
    p = argparse.ArgumentParser(
        description="Compare the trained SARA model vs the raw EXAONE backbone: "
                    "gold-ROUGE (local) and/or a Gemini pairwise judge."
    )
    add_model_decode_args(p)
    p.add_argument("--limit", type=int, default=0, help="cap on #test examples (0 = all)")
    p.add_argument("--gemini_model", default="gemini-3.5-flash", help="Gemini judge model id")
    p.add_argument("--gemini_temperature", type=float, default=0.0)
    p.add_argument("--skip_judge", action="store_true",
                   help="skip the Gemini judge -> local gold-ROUGE only (no API key needed)")
    p.add_argument("--no_rouge", action="store_true",
                   help="skip the local gold-ROUGE comparison")
    p.add_argument("--out", default=None, help="optional JSONL of per-example results")
    return p.parse_args()


def main():
    args = build_args()
    do_judge = not args.skip_judge
    do_rouge = not args.no_rouge
    if not do_judge and not do_rouge:
        raise SystemExit("--skip_judge and --no_rouge together leave nothing to compare")

    judge_call = None
    if do_judge:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY)를 설정하세요 "
                             "(또는 --skip_judge 로 ROUGE만).")
        judge_call = make_gemini(api_key, args.gemini_model, args.gemini_temperature)

    evaluator = None
    if do_rouge:
        from eval import Evaluator  # imports torchmetrics lazily
        evaluator = Evaluator()

    _, _, test_set = load_dataset(args.dataset, args.data_type)
    if not test_set:
        raise SystemExit("test split is empty")
    n = len(test_set) if args.limit <= 0 else min(args.limit, len(test_set))

    tokenizer = load_tokenizer(args)
    model = load_model(args)
    gen_cfg = build_gen_config(args, tokenizer)
    # BASE is a plain single-branch decode with no contrastive tilt, so it needs
    # no byte-level plausibility floor -- keep it truly raw.
    base_cfg = copy.deepcopy(gen_cfg)
    base_cfg.plausibility_alpha = 0.0
    device = "cuda" if _cuda_available() else "cpu"

    tally = {"mine": 0, "base": 0, "tie": 0}                 # Gemini judge
    rtally = {"mine": 0, "base": 0, "tie": 0}                # gold-ROUGE (added)
    rsum = {"mine": {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0},
            "base": {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}}
    rows_out = []
    for i in range(n):
        split_row = test_set[i]
        source = split_row[0]  # full raw 원문
        gold = split_row[1]

        row, meta = prepare_example(split_row, tokenizer, args)
        mine, _ = generate_summary(model, tokenizer, gen_cfg, row, args, device)
        base, _ = generate_summary(model, tokenizer, base_cfg, row, args, device, pure=True)

        # clear, self-describing keys: 원문 / 순수 LLM 요약 / LLM+MLP 요약
        rec = {
            "index": i,
            "source": source,            # 원문
            "gold": gold,                # 정답 요약 (참고용)
            "llm_summary": base,         # 순수 LLM (vanilla EXAONE)
            "llm_mlp_summary": mine,     # LLM + MLP (SARA)
        }
        _label = {"mine": "llm_mlp", "base": "llm", "tie": "tie"}
        line = f"[{i + 1}/{n}]"

        if do_rouge:
            rm = rouge_scores(evaluator, mine, gold)
            rb = rouge_scores(evaluator, base, gold)
            for k in rsum["mine"]:
                rsum["mine"][k] += rm[k]
                rsum["base"][k] += rb[k]
            rwin = "mine" if rm["added"] > rb["added"] else ("base" if rb["added"] > rm["added"] else "tie")
            rtally[rwin] += 1
            rdec = rtally["mine"] + rtally["base"]
            rwr = 100.0 * rtally["mine"] / rdec if rdec else 0.0
            rec["rouge_llm_mlp"] = rm
            rec["rouge_llm"] = rb
            rec["rouge_winner"] = _label[rwin]
            line += (f" ROUGE-added mlp {rm['added']:.3f} / llm {rb['added']:.3f} "
                     f"-> {_label[rwin]:7s} (win {rwr:.0f}%)")

        if do_judge:
            winner = judge_pair(judge_call, source, mine, base)
            tally[winner] += 1
            dec = tally["mine"] + tally["base"]
            wr = 100.0 * tally["mine"] / dec if dec else 0.0
            rec["judge_winner"] = _label[winner]
            line += f" | judge {_label[winner]:7s} (win {wr:.0f}%)"

        print(line)
        rows_out.append(rec)

    bar = "=" * 64
    print(f"\n{bar}")
    print(f"완료: {n} examples")
    if do_rouge:
        mavg = {k: rsum["mine"][k] / n for k in rsum["mine"]}
        bavg = {k: rsum["base"][k] / n for k in rsum["base"]}
        rdec = rtally["mine"] + rtally["base"]
        rwr = 100.0 * rtally["mine"] / rdec if rdec else 0.0
        print("[gold-ROUGE] 평균 F-measure (학습이 최적화한 지표):")
        print(f"  MINE : R1 {mavg['rouge1']:.4f}  R2 {mavg['rouge2']:.4f}  RLsum {mavg['rougeLsum']:.4f}")
        print(f"  BASE : R1 {bavg['rouge1']:.4f}  R2 {bavg['rouge2']:.4f}  RLsum {bavg['rougeLsum']:.4f}")
        print(f"  ROUGE 승률(added, 무승부 제외): MINE {rwr:.1f}%  "
              f"(mine {rtally['mine']} / base {rtally['base']} / tie {rtally['tie']})")
    if do_judge:
        dec = tally["mine"] + tally["base"]
        wr = 100.0 * tally["mine"] / dec if dec else 0.0
        print(f"[Gemini judge={args.gemini_model}] 원문 기준 요약 품질:")
        print(f"  MINE 승 {tally['mine']}  |  BASE 승 {tally['base']}  |  무승부 {tally['tie']}")
        print(f"  내 방법 승률(무승부 제외): {wr:.1f}%   ({tally['mine']}/{dec})")
    print(bar)

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        summary = {
            "n": n,
            "dataset": args.dataset,
            "input_mode": args.input_mode,
            "legend": {"llm_summary": "순수 LLM (vanilla EXAONE)",
                       "llm_mlp_summary": "LLM + MLP (SARA)"},
        }
        if do_rouge:
            summary["rouge_win"] = {"llm_mlp": rtally["mine"], "llm": rtally["base"], "tie": rtally["tie"]}
            summary["rouge_llm_mlp_avg"] = {k: rsum["mine"][k] / n for k in rsum["mine"]}
            summary["rouge_llm_avg"] = {k: rsum["base"][k] / n for k in rsum["base"]}
        if do_judge:
            summary["judge_win"] = {"llm_mlp": tally["mine"], "llm": tally["base"], "tie": tally["tie"]}
            summary["judge_model"] = args.gemini_model
        # single valid JSON: {summary, results:[...]}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": rows_out}, f,
                      ensure_ascii=False, indent=2)
        print(f"[out] full results ({n} examples) -> {args.out}")


def _cuda_available():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
