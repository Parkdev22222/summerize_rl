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


# -- per-dimension scoring judge ----------------------------------------------
# Each candidate is scored 1-5 on three dimensions, independently (one prompt per
# summary, no pairwise position bias). English keys for robust JSON; Korean labels
# for the report. Priority (accuracy >> coverage > brevity) is baked into the
# scoring guide, so a hallucination or wrong number pushes accuracy toward 1.
DIMS = [
    ("accuracy", "정확성"),   # 환각 없음 + 장비수·병력·수치 정확
    ("coverage", "누락"),     # 원문 핵심 문장 포함 (5=누락 없음)
    ("brevity", "간결성"),    # 군더더기 없이 간결
]

_SCORE_GUIDE = (
    "아래 [요약]이 [원문]을 얼마나 잘 요약했는지 세 항목을 각각 1~5점으로 채점하라 "
    "(5=매우 우수, 1=매우 미흡).\n"
    "채점 항목:\n"
    "- accuracy (정확성, 가장 중요): 원문에 없는 부대·장비·사건·수치를 지어냈거나(환각), "
    "장비 수·총 병력·부상자/사망자 등 수치를 원문과 다르게 적었으면 강하게 감점한다. "
    "그런 오류가 하나라도 있으면 1~2점, 전혀 없고 정확하면 5점.\n"
    "- coverage (누락): 원문의 중요한 상황·조치·건의 등 핵심 문장을 빠짐없이 담았는가. "
    "핵심 누락이 많을수록 낮고, 누락이 없으면 5점.\n"
    "- brevity (간결성): 군더더기 없이 핵심만 명료하게 표현했는가.\n"
    "먼저 한두 문장으로 간단히 평가한 뒤, 맨 마지막 줄에 아래 형식의 JSON 하나만 출력하라:\n"
    '{"accuracy": n, "coverage": n, "brevity": n}\n'
)


def score_prompt(source, summary):
    return (
        f"{_SCORE_GUIDE}\n[원문]\n{source}\n\n[요약]\n{summary}\n\n[채점 결과]\n"
    )


def _parse_dim_scores(text):
    """Pull {"accuracy":n,"coverage":n,"brevity":n} (each clamped 1-5) from the
    judge's text, tolerating a preamble before the JSON. None if unparseable."""
    if not text or "{" not in text or "}" not in text:
        return None
    blob = text[text.find("{"): text.rfind("}") + 1]
    try:
        obj = json.loads(blob)
    except Exception:
        return None
    out = {}
    for key, _lab in DIMS:
        v = obj.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        out[key] = max(1.0, min(5.0, float(v)))
    out["avg"] = sum(out[k] for k, _ in DIMS) / len(DIMS)
    return out


def score_summary(call, source, summary, retries=3):
    """Score one summary on the three dimensions (1-5) + their average, or None."""
    if not summary or not summary.strip():
        return {"accuracy": 1.0, "coverage": 1.0, "brevity": 1.0, "avg": 1.0}
    def _do():
        parsed = _parse_dim_scores(call(score_prompt(source, summary)) or "")
        if parsed is None:
            raise ValueError("unparseable score JSON")
        return parsed
    return _retry(_do, retries=retries, label="judge-score")


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
                    "gold-ROUGE (local) and/or a Gemini per-dimension score judge "
                    "(accuracy/coverage/brevity, 1-5)."
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

    tally = {"mine": 0, "base": 0, "tie": 0}                 # Gemini judge (by avg score)
    ssum = {"mine": {k: 0.0 for k, _ in DIMS}, "base": {k: 0.0 for k, _ in DIMS}}
    scount = 0                                               # #examples actually scored
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
            sm = score_summary(judge_call, source, mine)   # LLM+MLP
            sb = score_summary(judge_call, source, base)   # 순수 LLM
            if sm is None or sb is None:
                # scoring failed for this example -> record but don't tally
                rec["score_llm_mlp"], rec["score_llm"] = sm, sb
                line += " | judge SCORE-FAIL"
            else:
                scount += 1
                for k, _ in DIMS:
                    ssum["mine"][k] += sm[k]
                    ssum["base"][k] += sb[k]
                jwin = "mine" if sm["avg"] > sb["avg"] else ("base" if sb["avg"] > sm["avg"] else "tie")
                tally[jwin] += 1
                dec = tally["mine"] + tally["base"]
                wr = 100.0 * tally["mine"] / dec if dec else 0.0
                rec["score_llm_mlp"], rec["score_llm"] = sm, sb
                rec["judge_winner"] = _label[jwin]
                # print each dimension's score every time
                dims = "  ".join(
                    f"{lab} mlp {sm[k]:.0f}/llm {sb[k]:.0f}" for k, lab in DIMS
                )
                line += (f" | {dims}  avg mlp {sm['avg']:.2f}/llm {sb['avg']:.2f} "
                         f"-> {_label[jwin]:7s} (win {wr:.0f}%)")

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
        c = scount or 1
        m_avg = {k: ssum["mine"][k] / c for k, _ in DIMS}
        b_avg = {k: ssum["base"][k] / c for k, _ in DIMS}
        m_all = sum(m_avg.values()) / len(DIMS)
        b_all = sum(b_avg.values()) / len(DIMS)
        print(f"[Gemini judge={args.gemini_model}] 항목별 평균 점수 (1~5, {scount} examples 채점):")
        hdr = "  ".join(f"{lab}" for _, lab in DIMS)
        print(f"        {hdr}   전체평균")
        print(f"  MINE  " + "     ".join(f"{m_avg[k]:.2f}" for k, _ in DIMS) + f"    {m_all:.2f}")
        print(f"  BASE  " + "     ".join(f"{b_avg[k]:.2f}" for k, _ in DIMS) + f"    {b_all:.2f}")
        print(f"  평균점수 기준 승: MINE {tally['mine']} / BASE {tally['base']} / 무승부 {tally['tie']}"
              f"  (MINE 승률 {wr:.1f}%)")
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
            c = scount or 1
            summary["judge_model"] = args.gemini_model
            summary["judge_scored"] = scount
            summary["judge_win_by_avg"] = {"llm_mlp": tally["mine"], "llm": tally["base"], "tie": tally["tie"]}
            summary["score_llm_mlp_avg"] = {k: ssum["mine"][k] / c for k, _ in DIMS}
            summary["score_llm_avg"] = {k: ssum["base"][k] / c for k, _ in DIMS}
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
