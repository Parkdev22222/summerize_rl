"""Extra reference-free reward components ported into SARA's reward pipeline.

These three reward terms are lifted from the `summarize_rl` project
(github.com/parkdev22222/summerize_rl) and added on top of SARA's existing
ROUGE + FactKB reward:

  1. triplet_coverage   — per-entity content-token recall over KB triplets
  2. ungrounded_fact_penalty — fraction of the summary's checkable facts
                               (unit designations / quantities) NOT in the source
  3. GeminiJudge        — LLM-as-judge accuracy score (0-100) via the Gemini API

All logic is Korean-domain tuned (regexes / prompt), matching the Korean data
this SARA fork is trained on. `Evaluator` in utils.py imports from here so the
upstream SARA files stay minimally changed.

Origin (unmodified reference): summarize_rl/rewards.py, summarize_rl/judge.py,
summarize_rl/branches.py, examples/eval_gemini.py.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Logging helpers                                                              #
# --------------------------------------------------------------------------- #
def ema_update(prev, new, beta: float = 0.98):
    """Exponential moving average for smoothing noisy per-step TensorBoard curves.

    ``prev`` is the previous EMA (``None`` on the first call -> seed with ``new``);
    higher ``beta`` = smoother/slower. Pure/stdlib so it is unit-testable without
    torch. Logging-only: never feeds back into the reward or the gradient.
    """
    new = float(new)
    if prev is None:
        return new
    return float(beta * float(prev) + (1.0 - beta) * new)

# --------------------------------------------------------------------------- #
# Shared text helpers (from summarize_rl/rewards.py)                            #
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _nospace(text: str) -> str:
    """Lowercased text with all whitespace removed.

    Korean surface forms space compound entities inconsistently ("블루포스" vs
    "블루 포스"); matching against the space-stripped summary makes grounding
    robust to that split.
    """
    return _WS_RE.sub("", text.lower())


@dataclass
class Triplet:
    """A KB triplet. triplet_coverage only reads head/tail; relation is kept
    for parity with the source data ([head, relation, tail])."""

    head: str
    relation: str
    tail: str


def to_triplets(raw: object) -> list[Triplet]:
    """Coerce our data's triplet rows into Triplet objects.

    Accepts a list of ``[head, relation, tail]`` (our JSONL format), a list of
    ``Triplet``, or ``None`` -> empty list. Rows that are not 3-length are
    skipped defensively.
    """
    if not raw:
        return []
    out: list[Triplet] = []
    for t in raw:
        if isinstance(t, Triplet):
            out.append(t)
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            out.append(Triplet(str(t[0]), str(t[1]), str(t[2])))
        elif isinstance(t, dict) and "head" in t and "tail" in t:
            out.append(Triplet(str(t["head"]), str(t.get("relation", "")), str(t["tail"])))
    return out


def render_triplets(raw: object) -> str:
    """Render KB triplets ([head, relation, tail]) as '- head relation tail' lines.

    Pure/stdlib so both training (test_performance_decoder_new_fc) and inference
    (single_infer) can put triplets into the main branch identically. Empty when
    there are no triplets.
    """
    lines = []
    for t in to_triplets(raw):
        parts = [p for p in (t.head, t.relation, t.tail) if p]
        if parts:
            lines.append("- " + " ".join(parts))
    return "\n".join(lines)


def main_input_slot(document: str, triplets_raw: object, input_mode: str) -> str:
    """What fills the main branch's document slot for a given --input_mode.

    Shared by training and inference so the two never drift:
      * document          -> the report only (in-distribution / default)
      * document+triplets -> report, then rendered triplets under [관계 정보]
      * triplets          -> triplets only (no report)
    """
    triplet_text = render_triplets(triplets_raw)
    if input_mode == "document":
        return document
    if input_mode == "triplets":
        if not triplet_text.strip():
            raise ValueError("input_mode 'triplets' but example has no triplets")
        return triplet_text
    if input_mode == "document+triplets":
        if triplet_text.strip():
            return f"{document}\n\n[관계 정보]\n{triplet_text}"
        return document
    raise ValueError(f"unknown input_mode: {input_mode!r}")


# --------------------------------------------------------------------------- #
# On-the-fly triplet extraction with the backbone LLM                          #
#   For raw documents that have NO KB triplets: when --input_mode needs        #
#   triplets, ask the (frozen) backbone to pull [head|relation|tail] facts     #
#   from the source so document+triplets / triplets modes still work.          #
# --------------------------------------------------------------------------- #

def triplet_extract_prompt(document: str) -> str:
    """Korean prompt telling the backbone to extract KB triplets from a report.

    Emphasizes the facts this domain cares about (units, weapon systems + their
    quantities, troop counts, casualties) and fixes a strict 'head | relation |
    tail' line format so the output is machine-parseable."""
    return (
        "당신은 군사 보고서에서 지식 트리플을 추출하는 도구다. 아래 [원문]에서 '검증 가능한 "
        "사실'을 '개체 | 관계 | 값' 형식의 트리플로 빠짐없이 뽑아라. 특히 각 부대가 보유한 "
        "무기체계와 그 수량, 총 병력, 부상자·사망자 등 수치가 딸린 사실을 반드시 포함하라.\n"
        "규칙: (1) 한 줄에 트리플 하나. (2) '개체 | 관계 | 값' 형식만 쓰고 다른 설명·머리말은 "
        "쓰지 마라. (3) 원문에 없는 내용을 지어내지 마라. (4) 수량은 원문 그대로(단위 포함) 적어라.\n"
        "예)\n"
        "제3기계화보병대대 | 보유 | 대전차미사일 5기\n"
        "제3기계화보병대대 | 병력 | 1,200명\n\n"
        f"[원문]\n{document}\n\n[트리플]\n"
    )


def _parse_extracted_triplets(text, max_triplets: int = 30) -> list:
    """Parse 'head | relation | tail' lines into ``[[head, relation, tail], ...]``.

    Tolerant of list bullets and of a tail that itself contains '|'. Dedupes and
    caps at ``max_triplets``. Returns [] when nothing parseable is found."""
    out, seen = [], set()
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-•*· ").strip()
        if line.count("|") < 2:
            continue
        parts = [p.strip() for p in line.split("|")]
        head, rel = parts[0], parts[1]
        tail = "|".join(parts[2:]).strip()  # keep tails that contain '|'
        if not head or not tail:
            continue
        key = (head, rel, tail)
        if key in seen:
            continue
        seen.add(key)
        out.append([head, rel, tail])
        if len(out) >= max_triplets:
            break
    return out


def extract_triplets_with_model(model, tokenizer, document: str,
                                max_new_tokens: int = 256) -> list:
    """Extract KB triplets from ``document`` using the loaded backbone (frozen).

    Runs a plain single-branch greedy decode (no presumm/null, so the SAD/FC head
    is not involved) under ``torch.no_grad`` -- the extraction reflects the base
    model, never touches the RL graph, and works for any backbone (EXAONE/Llama/
    Qwen). Returns ``[[head, relation, tail], ...]`` ([] on empty/failed extract)."""
    if not document or not document.strip():
        return []
    import torch
    from types import SimpleNamespace

    prompt = triplet_extract_prompt(document)
    # Prefer the model's chat template (instruction-tuned backbones).
    try:
        text_in = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
    except Exception:  # no chat template -> raw prompt
        text_in = prompt
    enc = tokenizer(text_in, return_tensors="pt", truncation=True, max_length=1800)
    dev = next(model.parameters()).device
    input_ids = enc.input_ids.to(dev)
    attn = enc.attention_mask.to(dev)
    gc = SimpleNamespace(
        do_sample=False, top_k=0, top_p=1.0, temperature=1.0,
        min_new_tokens=1, max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                generation_config=gc, return_dict_in_generate=True,
            )
        gen = out.sequences[:, input_ids.shape[1]:]
        text_out = tokenizer.decode(gen[0], skip_special_tokens=True)
    except Exception as e:  # noqa: BLE001 - extraction must not crash inference
        print("[triplet-extract] 실패(빈 트리플로 진행): {}: {}".format(type(e).__name__, e))
        return []
    return _parse_extracted_triplets(text_out)


# --------------------------------------------------------------------------- #
# 1. Triplet coverage (from summarize_rl/rewards.py: triplet_coverage)          #
# --------------------------------------------------------------------------- #

def triplet_coverage(summary: str, triplets: list[Triplet]) -> float:
    """Mean per-entity content-token recall over head/tail entities, in [0, 1].

    Partial credit — the fraction of an entity's content tokens present — rather
    than all-or-nothing, because KB tails are long descriptive phrases. Matching
    is whitespace-insensitive; content tokens are >=2 chars and not pure digits.
    Vacuously 1.0 when there is nothing to cover.
    """
    entities: list[str] = []
    for t in triplets:
        entities.append(t.head)
        entities.append(t.tail)
    if not entities:
        return 1.0  # nothing to cover -> vacuously complete
    entities = list(dict.fromkeys(entities))  # dedupe, preserve order
    summary_norm = summary.lower()
    summ_ns = _nospace(summary_norm)
    scores: list[float] = []
    for e in entities:
        toks = [t for t in tokenize(e) if len(t) >= 2 and not t.isdigit()]
        if not toks:
            continue  # unscorable (single-char / pure-number entity)
        hit = sum(1 for t in toks if t in summary_norm or t in summ_ns)
        scores.append(hit / len(toks))
    return sum(scores) / len(scores) if scores else 1.0


# --------------------------------------------------------------------------- #
# 2. Ungrounded-fact / hallucination penalty                                   #
#    (from summarize_rl/rewards.py: ungrounded_fact_penalty)                    #
# --------------------------------------------------------------------------- #

# Checkable "facts" a military summary must not invent: unit designations
# (제1기계화보병대대, 3중대, 기갑여단) and quantities (전차 4대, 40발, 800m).
_FACT_RE = re.compile(
    r"제?\s*\d*\s*[가-힣]{0,8}?(?:여단|대대|중대|소대|사단|연대|전투단|편대|전대)"
    r"|\d[\d,]*\s*(?:대|명|발|문|정|기|km|m|여단|대대|중대|소대)"
    r"|(?:19|20)\d{2}"  # 4-digit year (e.g. a summary changing 2026 -> 2023)
)


def ungrounded_fact_penalty(summary: str, source: str) -> float:
    """Fraction of the summary's checkable facts NOT grounded in the source, [0,1].

    Extracts unit designations and quantities from the summary and checks each
    against the source (whitespace- and comma-insensitive). 0.0 when the summary
    states no checkable facts.
    """
    def _norm(text: str) -> str:
        return re.sub(r"[\s,]", "", text.lower())

    src = _norm(source)
    facts = {_norm(m) for m in _FACT_RE.findall(summary)}
    facts = {f for f in facts if len(f) >= 2}
    if not facts:
        return 0.0
    ungrounded = sum(1 for f in facts if f not in src)
    return ungrounded / len(facts)


# --------------------------------------------------------------------------- #
# 3. LLM-as-judge via Gemini (from summarize_rl/judge.py + examples/eval_gemini)#
# --------------------------------------------------------------------------- #

_SCORE_RE = re.compile(r"\d{1,3}")


def _parse_score(text: str | None) -> float | None:
    """Pull a 0-100 score out of the judge's text -> [0,1], or None if absent.

    Takes the LAST 0-100 integer in the output (a short preamble before the
    number is tolerated).
    """
    nums = [int(x) for x in _SCORE_RE.findall(text or "")]
    nums = [n for n in nums if 0 <= n <= 100]
    if not nums:
        return None
    return max(0.0, min(1.0, nums[-1] / 100.0))


def judge_prompt(source: str, summary: str) -> str:
    """Korean absolute-accuracy judge prompt (from BackboneJudge._prompt)."""
    return (
        "당신은 군사 보고서 요약을 채점하는 심사관이다. 아래 [요약]이 [원문]을 "
        "얼마나 정확히 반영했는지 0~100 정수 하나로만 평가하라.\n"
        "채점 기준: (1) 부대·수치·지명·시간을 정확히 반영했는가, "
        "(2) 원문에 없는 부대·사건·숫자를 지어내지 않았는가, "
        "(3) 각 제대의 상황과 핵심 조치·건의를 빠짐없이 담았는가.\n"
        "정확할수록 100, 지어내거나 핵심을 빠뜨릴수록 0. 숫자만 출력하라.\n\n"
        f"[원문]\n{source}\n\n[요약]\n{summary}\n\n[점수(0-100)]\n"
    )


def _keyfact_lines(keyfacts, max_keyfacts: int = 8) -> list[str]:
    """Normalize keyfacts (list[str] or newline-joined str) to <= N clean lines."""
    if not keyfacts:
        return []
    if isinstance(keyfacts, str):
        items = keyfacts.split("\n")
    else:
        items = list(keyfacts)
    out = []
    for it in items:
        s = str(it).strip().lstrip("-•· ").strip()
        if s:
            out.append(s)
        if len(out) >= max_keyfacts:
            break
    return out


def judge_prompt_subscore(source: str, summary: str, keyfacts=None) -> str:
    """Per-criterion (0-5 each) military-summary judge prompt.

    Scoring three axes separately (accuracy / non-fabrication / coverage) and
    summing to 0-15 spreads the reward better than a single 0-100 guess (which
    tends to anchor near one value). Criteria are tuned to this data (military
    situation reports: unit names, troop counts, equipment quantities, dates/
    places, operational phase).

    When ``keyfacts`` are supplied (this repo's gold key facts), they are shown
    as a checklist and criterion 3 becomes "how many of these are correctly &
    faithfully reflected" — a semantic check (paraphrase counts) that the lexical
    coverage reward misses. Capped to the top few so the prompt stays short.
    """
    kf = _keyfact_lines(keyfacts)
    if kf:
        checklist = "\n".join("  {}. {}".format(i + 1, s) for i, s in enumerate(kf))
        kf_block = "[핵심 사실]\n{}\n\n".format(checklist)
        crit3 = (
            "- 핵심포함(0-5): 위 [핵심 사실]들을 (표현이 달라도) 의미상 빠짐없이 정확히 "
            "반영했는가(많이 반영·정확할수록 5, 빠뜨리거나 왜곡할수록 0).\n"
        )
    else:
        kf_block = ""
        crit3 = (
            "- 핵심포함(0-5): 양측의 목표·병력 편성·주요 장비·현재 작전 국면·핵심 지형/교전규칙 "
            "등 중요한 사실을 빠짐없이 담았는가.\n"
        )
    return (
        "당신은 군사 상황보고 요약을 채점하는 엄정한 심사관이다. [원문]을 근거로 [요약]을 채점하라.\n"
        "[1단계 — 정확성 사실 대조] 먼저 [원문]의 '검증 가능한 사실 항목'을 빠짐없이 한 줄씩 "
        "나열하라: 각 부대의 보유 무기체계와 그 수량, 총 병력, 부상자·사망자 등 수치가 딸린 "
        "항목을 각각 하나의 항목으로 센다(예: '대전차미사일 5기', '박격포 10문'은 각각 별개 항목). "
        "각 항목이 [요약]에 정확히(무기체계 이름 + 원문 수량까지 일치) 반영됐으면 O, 누락·수량 "
        "생략·수치 오류면 X로 표시하라.\n"
        "[2단계 — 나머지 항목 0~5점(정수)]\n"
        "- 비조작(0-5): 원문에 없는 부대·수치·장비·사건을 지어내지 않았는가(지어내면 감점).\n"
        + crit3 +
        "[출력] 대조표를 먼저 보인 뒤, 마지막에 아래 형식으로 출력하라(정확성_적중 = O 개수/전체 "
        "항목 수):\n"
        "정확성_적중: <O수>/<전체수>\n비조작: <0-5>\n핵심포함: <0-5>\n\n"
        f"[원문]\n{source}\n\n{kf_block}[요약]\n{summary}\n\n채점:\n"
    )


_SUB_LABELS = ("정확성", "비조작", "핵심포함")


def _parse_subscore(text: str | None) -> float | None:
    """Parse the sub-score judge output -> reward in [0,1], or None if unreadable.

    Primary path (mirrors compare_gemini): the accuracy sub-score is COMPUTED IN
    CODE from the O/X checklist ratio ``정확성_적중: h/t`` -> 5*(h/t), so accuracy no
    longer depends on the judge's own arithmetic/scale; it is summed with the
    비조작/핵심포함 0-5 sub-scores to 0-15 and normalized. Robust fallbacks keep a
    weaker local judge usable: (2) three labeled 0-5 sub-scores, (3) ``[총점]``
    0-15, (4) the last 0-15 integer.
    """
    t = text or ""
    # (1) accuracy from the O/X checklist counts (compare_gemini-style)
    acc = None
    m = re.search(r"정확성[_ ]?적중\s*[:：]?\s*(\d+)\s*/\s*(\d+)", t)
    if m:
        h, tot = int(m.group(1)), int(m.group(2))
        acc = 5.0 if tot <= 0 else 5.0 * (max(0, min(h, tot)) / tot)
    others = []
    for label in ("비조작", "핵심포함"):
        mm = re.search(label + r"\s*[:：]?\s*([0-5])", t)
        if mm:
            others.append(int(mm.group(1)))
    if acc is not None and len(others) == 2:
        return (acc + others[0] + others[1]) / 15.0
    # (2) fallback: three labeled 0-5 sub-scores
    subs = []
    for label in _SUB_LABELS:
        m = re.search(label + r"\s*[:：]?\s*([0-5])", t)
        if m:
            subs.append(int(m.group(1)))
    if len(subs) == 3:
        return sum(subs) / 15.0
    # (3) fallback: [총점] 0-15
    m = re.search(r"총점\s*\]?\s*[:：]?\s*(\d{1,2})", t)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 15:
            return v / 15.0
    # (4) fallback: last 0-15 integer
    nums = [int(x) for x in re.findall(r"\d{1,2}", t)]
    nums = [n for n in nums if 0 <= n <= 15]
    if nums:
        return nums[-1] / 15.0
    return None


def make_gemini(api_key: str, model_name: str, temperature: float = 0.0):
    """Return call(prompt)->str using whichever google SDK is installed.

    ``temperature`` is fixed (default 0) so the judge is reproducible.
    (Ported from examples/eval_gemini.py.)
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


def _default_data_dir() -> str:
    """Resolve this repo's data/ dir (../../../data from src/) or $SUMMARIZE_RL_DATA."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get(
        "SUMMARIZE_RL_DATA",
        os.path.normpath(os.path.join(here, "..", "..", "..", "data")),
    )


def _read_ours(path: str) -> list:
    """Read one summarize_rl JSONL file into SARA rows with triplets kept.

    Each row -> ``[document, summary, presumm(keyfacts joined), triplets, split]``.
    """
    rows = []
    with open(path, "r", newline="\n") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            presumm = "\n".join(r.get("keyfacts", []) or [])
            rows.append([
                r["source_text"], r["summary_text"], presumm,
                r.get("triplets", []), r.get("split", "train"),
            ])
    return rows


def load_ours_dataset(data_dir: str | None = None):
    """Load the summarize_rl Korean corpus as SARA (train, validation, test).

    Pure-stdlib (no torch) so it is unit-testable on its own. Rows are
    ``[document, summary, presumm, triplets]``; train/eval split by the `split`
    field; the held-out test_scenarios file is the test set when present.
    """
    data_dir = data_dir or _default_data_dir()
    main = _read_ours(os.path.join(data_dir, "scenarios_ko.jsonl"))
    train = [row[:4] for row in main if row[4] == "train"]
    validation = [row[:4] for row in main if row[4] != "train"]
    test_path = os.path.join(data_dir, "test_scenarios_ko.jsonl")
    if os.path.exists(test_path):
        test = [row[:4] for row in _read_ours(test_path)]
    else:
        test = validation
    return train, validation, test


class BackboneJudge:
    """LLM-as-judge that scores with the LOCAL backbone (e.g. EXAONE 3.5).

    Reuses the already-loaded model — no API key, no network. It calls
    ``model.generate`` WITHOUT presumm/null inputs, which the SAD generate
    treats as plain single-branch decoding (the FC policy is not involved), so
    the score reflects the frozen base model's judgment. Generation runs under
    ``torch.no_grad`` so it never touches the RL gradient graph.

    ``score(source, summary) -> float`` in [0,1]; results are cached per
    (source, summary).
    """

    def __init__(self, model, tokenizer, device=None, max_new_tokens=16):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._cache: dict[tuple[str, str], float] = {}
        self._warned = False
        self.calls = 0
        self.failures = 0

    def _warn(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            print("[judge] " + msg)

    def _prompt_text(self, source: str, summary: str, keyfacts=None) -> str:
        prompt = judge_prompt_subscore(source, summary, keyfacts)
        # Prefer the model's chat template (EXAONE is instruction-tuned).
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
        except Exception:  # no chat template -> raw prompt
            return prompt

    def score(self, source: str, summary: str, keyfacts=None) -> float:
        key = (source, summary)
        if key in self._cache:
            return self._cache[key]
        self.calls += 1
        try:
            import torch
            from types import SimpleNamespace

            tok = self.tokenizer
            enc = tok(self._prompt_text(source, summary, keyfacts), return_tensors="pt",
                      truncation=True, max_length=1800)
            dev = self.device or next(self.model.parameters()).device
            input_ids = enc.input_ids.to(dev)
            attn = enc.attention_mask.to(dev)
            gc = SimpleNamespace(
                do_sample=False, top_k=0, top_p=1.0, temperature=1.0,
                min_new_tokens=1, max_new_tokens=self.max_new_tokens,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            )
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    generation_config=gc, return_dict_in_generate=True,
                )
            gen = out.sequences[:, input_ids.shape[1]:]
            text = tok.decode(gen[0], skip_special_tokens=True)
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            self._warn("backbone judge failed (returning 0.0): {}: {}".format(type(e).__name__, e))
            return 0.0
        val = _parse_subscore(text)
        if val is None:
            self.failures += 1
            self._warn("no 0-100 score parsed (returning 0.0). raw={!r}".format((text or "")[:160]))
            val = 0.0
        val = float(val)
        self._cache[key] = val
        return val


class GeminiJudge:
    """LLM-as-judge backed by a ``call(prompt)->str`` function (Gemini).

    ``judge_call`` is any callable returning the model's text for a prompt
    (build one with :func:`make_gemini`). ``score`` returns a faithfulness
    accuracy in [0,1], or 0.0 when the call/parse fails. Results are cached per
    (source, summary) so repeated rollouts of the same pair cost one API call.
    """

    def __init__(self, judge_call):
        self.judge_call = judge_call
        self._cache: dict[tuple[str, str], float] = {}
        self._warned = False  # only surface the first failure, to avoid log spam
        self.calls = 0
        self.failures = 0

    def _warn(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            print("[judge] " + msg)

    def score(self, source: str, summary: str, keyfacts=None) -> float:
        # keyfacts accepted for a uniform judge interface; legacy Gemini path
        # keeps the absolute-score prompt and ignores them.
        if self.judge_call is None:
            return 0.0
        key = (source, summary)
        if key in self._cache:
            return self._cache[key]
        self.calls += 1
        try:
            text = self.judge_call(judge_prompt(source, summary))
        except Exception as e:  # network / rate-limit / SDK / auth error
            self.failures += 1
            self._warn("Gemini call failed (returning 0.0): {}: {}".format(type(e).__name__, e))
            return 0.0
        val = _parse_score(text)
        if val is None:  # response had no 0-100 integer to read
            self.failures += 1
            self._warn("no 0-100 score parsed (returning 0.0). raw={!r}".format((text or "")[:160]))
            val = 0.0
        val = float(val)
        self._cache[key] = val
        return val
