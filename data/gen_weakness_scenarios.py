"""Weak-axis-targeted synthetic scenario generation (Phase 4).

Reuses the data.gen_test_scenarios template engine (country/terrain/force pools,
Korean-particle helpers, section structure) but modulates *generation knobs* by
the diagnosed weak axes. The steering is concept-level — it reads the failure
meta profile (numeric density, echelon count, ...) rather than copying failed
text — so the policy is stressed on the failure *kind*, not memorized examples.

Axis vocabulary is the reward-breakdown names produced by the diagnoser
(faithfulness / coverage / key_sentence / hallucination). backbone_limited axes
are already excluded from report.weak_axes, so they earn no knobs here.

Usage:
    python -m data.gen_weakness_scenarios --axes hallucination coverage \
        --n 16 --round 1 --out data/synth_r1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

from summarize_rl.rewards import _FACT_RE, _contains, tokenize

from data.gen_test_scenarios import UNIT_TYPES, _neun, make_scenario

# Offset so synthetic ids never collide with the train set (0-9999) or the
# generated test set (10000+).
ID_OFFSET = 20000


def default_params() -> dict:
    return {
        "numeric_density": 1.0,     # >1 injects near-value numeric traps
        "entity_crossover": False,  # subject-swap (who took damage vs withdrew)
        "near_miss_names": False,   # unit named one step off a frequent form
        "late_key_events": False,   # push a key event / recommendations to the tail
        "extra_recommendations": 0, # number of trailing 건의 lines
        "long_key_sentences": False,# long subordinate-clause key judgement
    }


# Reward-axis -> which knobs it turns on (and their baseline magnitude).
AXIS_KNOBS = {
    "faithfulness": {"numeric_density": 1.6, "entity_crossover": True},
    "coverage": {"late_key_events": True, "extra_recommendations": 2},
    "key_sentence": {"long_key_sentences": True},
    "hallucination": {"near_miss_names": True},
    # copy_penalty has no synthesis analog (it's an anti-copy behavior, not a
    # scenario property) -> intentionally absent.
}


def apply_knobs(params: dict, knobs: dict, intensity: float = 1.0) -> dict:
    p = dict(params)
    for k, v in knobs.items():
        if k == "numeric_density":
            p[k] = max(p[k], float(v) * intensity)
        elif k == "extra_recommendations":
            p[k] = p[k] + int(round(float(v) * intensity))
        else:
            p[k] = bool(v) or p[k]
    return p


def synthesis_params_from(report) -> dict:
    """Turn a DiagnosisReport into generation knobs, scaled by per-axis severity.

    Failure meta statistics set intensity (e.g. numeric-dense failures push the
    numeric knob further); the failed text itself is never used.
    """
    p = default_params()
    ms = report.meta_stats
    for axis in report.weak_axes:  # backbone_limited already excluded upstream
        knobs = AXIS_KNOBS.get(axis)
        if not knobs:
            continue
        p = apply_knobs(p, knobs, intensity=ms.severity(axis))
    return p


def synthesis_budget(report, train_size: int, ratio: float = 0.25,
                     min_per_axis: int = 8) -> int:
    """M = max(ratio*train_size, min_per_axis * #weak_axes). Scales with breadth."""
    n_axes = max(1, len(report.weak_axes))
    return max(round(ratio * train_size), min_per_axis * n_axes)


def _grounded(phrase: str, source_norm: str) -> bool:
    return _contains(source_norm, phrase)


def triplets_grounded(rec: dict) -> bool:
    """Every triplet's tail (the stated fact) must appear in the source.

    Heads are often structural labels the template invents ("교전지역",
    "작전단계") rather than verbatim source spans, so grounding is checked on the
    content tail — the claim that must not be a fabrication.
    """
    source_norm = rec["source_text"].lower()
    for t in rec["triplets"]:
        if len(t) != 3:
            return False
        if not _grounded(t[2], source_norm):
            return False
    return True


def _dedupe_triplets(triplets: list) -> list:
    seen, out = set(), []
    for t in triplets:
        k = tuple(t)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def make_weakness_scenario(rng: random.Random, seq: int, params: dict,
                           round_k: int) -> dict:
    """A base scenario augmented with weak-axis traps, kept fully self-grounded."""
    rec = make_scenario(rng, seq)
    rec["id"] = ID_OFFSET + seq
    rec["split"] = f"synthetic_r{round_k}"
    extra: list[str] = []

    if params.get("numeric_density", 1.0) > 1.0:
        # Near-value numeric traps: companies with close counts + a digit-
        # confusable reserve figure (e.g. 120 vs 1,200), stresses numeric fidelity.
        n_items = max(2, min(6, int(round(2 * params["numeric_density"]))))
        base = rng.choice([100, 110, 120])
        lines = [f"{i + 1}중대 {base + i * 10}명" for i in range(n_items)]
        reserve = f"예비대 {base * 10:,}명"
        extra.append("### 병력 세부:\n- " + ", ".join(lines) + f", {reserve} 규모이다.\n")
        for ln in lines:
            unit, cnt = ln.split(" ", 1)
            rec["keyfacts"].append(f"{ln} 규모이다.")
            rec["triplets"].append([unit, "규모", cnt])
        r_unit, r_cnt = reserve.split(" ", 1)
        rec["keyfacts"].append(f"{reserve} 규모이다.")
        rec["triplets"].append([r_unit, "규모", r_cnt])

    if params.get("near_miss_names"):
        # A unit named one step off a frequent training form, to catch a policy
        # that leaks the high-frequency name it saw in training.
        base_unit = rng.choice(UNIT_TYPES)
        near = base_unit.replace("보병", "수색") if "보병" in base_unit else base_unit + "대"
        name = f"제1{near}"
        extra.append(f"### 추가 제대:\n- {_neun(name)} 측방에서 관측되었다.\n")
        rec["keyfacts"].append(f"{_neun(name)} 측방에서 관측되었다.")
        rec["triplets"].append([name, "위치", "측방"])

    if params.get("entity_crossover"):
        # Subject-swap trap across echelons.
        extra.append("### 교차 보고:\n- 블루 2중대가 피해를 입었고 레드 1대대가 철수했다.\n")
        rec["keyfacts"].append("블루 2중대가 피해를 입었고 레드 1대대가 철수했다.")
        rec["triplets"].append(["블루 2중대", "상태", "피해"])
        rec["triplets"].append(["레드 1대대", "행동", "철수"])

    n_rec = params.get("extra_recommendations", 0)
    if params.get("late_key_events") or n_rec:
        pool = ["즉각 예비대 투입을 건의한다.", "항공 화력 지원을 요청한다.",
                "의료 후송 우선순위 조정을 건의한다."]
        recs = pool[:max(2, n_rec)]
        extra.append("### 지휘관 건의 (후반부):\n" + "".join(f"- {r}\n" for r in recs))
        rec["keyfacts"].extend(recs)

    if params.get("long_key_sentences"):
        long_sent = (
            "적 주력이 야음을 틈타 우회 기동을 시도하는 정황이 포착되었으며, 이에 따라 "
            "예비대의 조기 전개와 측방 경계 강화가 요구되는 상황으로 판단된다."
        )
        extra.append("### 핵심 판단:\n- " + long_sent + "\n")
        rec["keyfacts"].append(long_sent)

    if extra:
        rec["source_text"] = rec["source_text"] + "\n" + "\n".join(extra)
    rec["triplets"] = _dedupe_triplets(rec["triplets"])
    return rec


def entity_overlap(rec: dict, train_names: set[str]) -> float:
    """Fraction of the record's triplet-head entities also seen in the train set."""
    heads = {t[0] for t in rec["triplets"]}
    if not heads:
        return 0.0
    return sum(1 for h in heads if any(_contains(h.lower(), n.lower()) for n in train_names)) / len(heads)


def generate(n: int, axes: list[str], round_k: int, seed: int = 4242,
             params: dict | None = None) -> list[dict]:
    """Generate `n` unique weak-axis scenarios for the given axes."""
    if params is None:
        params = default_params()
        for axis in axes:
            knobs = AXIS_KNOBS.get(axis)
            if knobs:
                params = apply_knobs(params, knobs, intensity=1.0)
    rng = random.Random(seed)
    seen, out, seq, attempts = set(), [], 0, 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        rec = make_weakness_scenario(rng, seq, params, round_k)
        sig = rec["source_text"][:500]
        if sig in seen or not triplets_grounded(rec):
            continue
        seen.add(sig)
        out.append(rec)
        seq += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate weak-axis-targeted scenarios.")
    p.add_argument("--axes", nargs="+", default=["hallucination"],
                   help="reward-breakdown axes to target (faithfulness/coverage/"
                        "key_sentence/hallucination)")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--out", default="data/synth_r1.jsonl")
    args = p.parse_args()

    records = generate(args.n, args.axes, args.round, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    numeric = sum(
        sum(1 for m in _FACT_RE.findall(r["source_text"]) if any(c.isdigit() for c in m))
        for r in records
    ) / max(1, len(records))
    print(f"wrote {len(records)} weak-axis scenarios (axes={args.axes}, "
          f"round={args.round}) -> {args.out} (avg numeric tokens={numeric:.1f})")


if __name__ == "__main__":
    main()
