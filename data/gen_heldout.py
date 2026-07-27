"""Generate held-out (unseen) SITREP scenarios for offline Gemini evaluation.

The training corpus (``data/scenarios_ko.jsonl``) is analytical scenario briefs
(Galdovia/Metania). These held-out records are a DIFFERENT genre -- Korean
field-report SITREPs (보고번호 / 예하 중대별 상황 / 조치·건의) -- so evaluating
on them tests generalization to both unseen content AND an unseen format.

Each record mirrors the JSONL schema the evaluator reads: ``source_text`` (the
SITREP) plus ``triplets`` ([head, rel, tail]) derived from the SAME facts, so
the SQ (core-info) branch is filled exactly as in training (no fabrication from
an empty SQ). Deterministic given ``--seed`` for reproducible evals.

Usage:
    python -m data.gen_heldout --n 50 --out data/heldout_scenarios_ko.jsonl
    # then evaluate on it (no eval-code change needed):
    GEMINI_API_KEY=... uv run -m examples.eval_gemini \
        --model <M> --ckpt checkpoints/best.pt --trust-remote-code \
        --data data/heldout_scenarios_ko.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

MOUNTAINS = [
    "매봉산", "백석산", "가리산", "오봉산", "월악산", "천마산", "용문산", "치악산",
    "소백산", "운장산", "덕유산", "금학산", "비슬산", "팔공산", "무등산", "조령산",
    "황악산", "화악산", "명지산", "계방산", "응봉산", "국망봉", "선자령", "대암산",
]
BN_CALLSIGNS = [
    "화랑", "백호", "청룡", "맹호", "백마", "비룡", "독수리", "불사조", "번개", "질풍",
    "태풍", "충무", "을지", "광개토", "강감찬", "이순신", "권율", "김유신", "계백", "온달",
]
CO_CALLSIGNS = [
    "호랑이", "독사", "매", "벼락", "표범", "살모사", "흑곰", "천둥", "화룡", "백랑",
    "돌풍", "해일", "화산", "서리", "혹한", "벼랑", "야수", "붉은매", "백호랑", "질주",
]
KOR_ORD = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차"]
ENEMY_SIZE = ["1개 소대", "약 2개 소대", "1개 중대", "증강된 1개 소대", "2개 분대 규모"]
ENEMY_KIND = ["경보병", "기계화보병", "특수전 병력", "정찰대", "보병"]
APPROACH = ["북측 능선", "좌측 계곡", "우측 계곡", "정면", "남측 접근로", "동측 사면", "서측 능선"]
POSITIONS = ["좌측 능선", "중앙 정상부", "우측 능선", "전방 초소", "측방 진지", "후사면"]
DEF_ACTIONS = [
    ("K3 기관총과 클레이모어", "적의 접근을 저지"),
    ("K6 중기관총", "적 화력을 제압"),
    ("60mm 박격포와 소총 사격", "적의 전진을 차단"),
    ("대전차 로켓(팬저파우스트)", "적 장갑차 1대를 파괴"),
    ("매복조와 수류탄", "적 침투조를 격멸"),
    ("조명지뢰와 기관총", "야간 침투를 저지"),
    ("81mm 박격포 근접지원", "적 밀집대형을 타격"),
]
CASUALTIES = ["인원 손실 없음", "부상 2명", "부상 3명", "전사 1명·부상 2명", "경상 1명", "부상 4명"]
REQ_CO = [
    "위생병 및 후송을 요청함", "탄약 재보급을 요청함", "대포병 사격을 요청함",
    "조명지뢰 추가 설치를 건의함", "증원 병력을 요청함", "야시장비 보급을 건의함",
]
MORTAR = ["81mm 박격포", "60mm 박격포", "4.2인치 박격포"]
HEAVY = ["106mm 무반동총", "90mm 무반동총", "TOW 대전차미사일", "K4 고속유탄기관총"]
BN_ACT = [
    "대대 예비인 {r}중대를 {hill}고지 후사면에 대기시킴",
    "연대에 대포병 사격 및 탄약 재보급을 요청함",
    "생포한 포로의 진술을 정보과에서 확인 중임",
    "의무후송 헬기(MEDEVAC)를 연대에 요청함",
    "예비대를 좌측 능선으로 전환 배치함",
]
RECS = [
    "{hill}고지 북측 접근로에 대한 연대 포병의 화력지원을 요청함",
    "야간 방어를 대비하여 조명탄 및 야시장비 추가 보급을 건의함",
    "의무후송 및 응급의료 지원을 요청함",
    "공병 지원을 통한 장애물 보강을 건의함",
    "드론 정찰 자산의 지원을 요청함",
]
KO_NUM = ["1", "2", "3", "4", "5", "6"]


def _ro(word: str) -> str:
    """Korean instrumental particle: 로 after a vowel or ㄹ, else 으로."""
    ch = word[-1]
    if not ("가" <= ch <= "힣"):
        return "로"
    jong = (ord(ch) - 0xAC00) % 28
    return "로" if jong in (0, 8) else "으로"  # 8 = ㄹ


def _dtg(rng: random.Random) -> str:
    day = rng.randint(1, 28)
    hh = rng.randint(0, 23)
    mm = rng.choice(["00", "10", "15", "30", "45"])
    mon = rng.choice(["JAN", "MAR", "MAY", "JUL", "SEP", "NOV"])
    return f"{day:02d}{hh:02d}{mm}K {mon} 26"


def _company_block(rng, idx, callsign, hill):
    unit = f"{KO_NUM[idx]}중대"
    pos = POSITIONS[idx % len(POSITIONS)]
    esize = rng.choice(ENEMY_SIZE)
    ekind = rng.choice(ENEMY_KIND)
    appr = rng.choice(APPROACH)
    dist = rng.choice([200, 300, 400, 500, 600, 800])
    weapon, effect = rng.choice(DEF_ACTIONS)
    cas = rng.choice(CASUALTIES)
    req = rng.choice(REQ_CO)
    tmin = rng.randint(30, 58)
    text = (
        f"   {KOR_ORD[idx]}. {unit} (\"{callsign}\") — {pos} ({hill}고지)\n"
        f"       (1) 2406{tmin}시 적 {ekind} {esize}가 {unit} {appr} {dist}m로 접근함.\n"
        f"       (2) {unit}는 {weapon}{_ro(weapon)} {effect}하였음.\n"
        f"       (3) 아군 피해는 {cas}이며, {unit}는 {req}\n"
    )
    triplets = [
        [unit, "위치", f"{hill}고지 {pos}"],
        [unit, "교전", f"적 {ekind} {esize}"],
        [unit, "조치", weapon],
        [unit, "피해", cas],
        [unit, "건의", req.replace("함", "").strip()],
    ]
    return text, triplets


def make_scenario(rng: random.Random, seq: int) -> dict:
    mountain = rng.choice(MOUNTAINS)
    hill = rng.choice([208, 305, 412, 449, 507, 560, 621, 688, 733, 811, 356, 274])
    bn = rng.choice(BN_CALLSIGNS)
    regt = rng.choice([f"제0{rng.randint(1,9)}보병연대", "제00보병연대"])
    n_rifle = rng.choice([2, 3, 3, 3])  # mostly 3 rifle companies
    cos = rng.sample(CO_CALLSIGNS, n_rifle + 1)
    enemy_size_bn = rng.choice(["대대 규모", "중대 규모", "증강된 중대 규모"])
    enemy_kind_bn = rng.choice(ENEMY_KIND)
    n_axis = rng.choice(["2개", "3개", "복수의"])
    dtg = _dtg(rng)

    triplets: list[list[str]] = [
        ["대대", "방어지역", f"{mountain} {hill}고지"],
        ["적", "규모", f"{enemy_size_bn} {enemy_kind_bn}"],
        ["적", "공격방향", f"{n_axis} 방향"],
    ]

    header = (
        "■ 상황보고 (SITREP)\n\n"
        f"보고번호 : SITREP-2026-{700+seq:04d}-{rng.randint(1,99):03d}\n"
        f"발신 : 제{rng.randint(1,9)}보병대대 (호출부호 \"{bn}\")\n"
        f"수신 : {regt} 작전과\n"
        f"DTG : {dtg}\n"
        "분류 : 대외비 (가상 훈련자료)\n\n"
    )
    sec1 = (
        "1. 개요\n"
        f"   가. 대대는 {mountain}({hill}고지) 일대에서 방어진지를 점령하고, "
        f"{rng.choice(APPROACH)}으로 접근하는 적 보병부대와 교전 중임.\n"
        f"   나. 적은 {enemy_size_bn} {enemy_kind_bn}{_ro(enemy_kind_bn)}, 박격포 지원 하에 "
        f"{hill}고지를 탈취하기 위해 {n_axis} 방향에서 공격 중인 것으로 판단됨.\n"
        f"   다. 대대는 예하 {n_rifle}개 소총중대와 1개 화기중대로 {hill}고지 및 "
        "인접 능선을 방어 중임.\n\n"
    )

    sec2 = "2. 예하 중대별 상황\n\n"
    for i in range(n_rifle):
        block, tr = _company_block(rng, i, cos[i], hill)
        sec2 += block + "\n"
        triplets += tr

    # weapons company
    wcs = cos[n_rifle]
    mortar = rng.choice(MORTAR)
    rounds = rng.choice([33, 40, 48, 55, 62, 70, 88])
    heavy = rng.choice(HEAVY)
    wc_letter = KOR_ORD[n_rifle]
    sec2 += (
        f"   {wc_letter}. 화기중대 (\"{wcs}\") — 대대 화력지원\n"
        f"       (1) 화기중대는 {mortar}로 예하 중대 정면에 화력지원을 실시 중임.\n"
        f"       (2) 화기중대는 240700시까지 박격포탄 {rounds}발을 사격함.\n"
        f"       (3) 화기중대는 {heavy}{_ro(heavy)} 적 화기진지 1개소를 제압함.\n"
        "       (4) 화기중대는 박격포탄 잔량 부족으로 긴급 재보급을 요청함.\n\n"
    )
    triplets += [
        ["화기중대", "화력지원", mortar],
        ["화기중대", "사격량", f"박격포탄 {rounds}발"],
        ["화기중대", "제압", f"{heavy} 적 화기진지 1개소"],
        ["화기중대", "건의", "박격포탄 긴급 재보급"],
    ]

    # battalion actions + recommendations
    r_res = KO_NUM[n_rifle]  # reserve = next company after the rifle companies
    acts = rng.sample(BN_ACT, 3)
    sec3 = "3. 대대 조치사항\n"
    for j, a in enumerate(acts):
        sec3 += f"   {KOR_ORD[j]}. {a.format(r=r_res, hill=hill)}.\n"
    sec3 += "\n"
    recs = rng.sample(RECS, 2)
    sec4 = "4. 건의사항\n"
    for j, rc in enumerate(recs):
        sec4 += f"   {KOR_ORD[j]}. {rc.format(hill=hill)}.\n"
    sec4 += "\n끝."

    triplets.append(["대대", "건의", recs[0].format(hill=hill)])

    source_text = header + sec1 + sec2 + sec3 + sec4
    # de-dup triplets, keep order
    seen = set()
    uniq = []
    for t in triplets:
        k = tuple(t)
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return {"id": seq, "source_text": source_text, "triplets": uniq, "split": "heldout"}


def main() -> None:
    p = argparse.ArgumentParser(description="Generate held-out SITREP scenarios (JSONL).")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", default="data/heldout_scenarios_ko.jsonl")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    rng = random.Random(args.seed)
    seen_sig = set()
    records = []
    seq = 0
    attempts = 0
    while len(records) < args.n and attempts < args.n * 50:
        attempts += 1
        rec = make_scenario(rng, seq)
        sig = rec["source_text"][:400]
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        records.append(rec)
        seq += 1

    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} held-out scenarios -> {args.out} "
          f"(mean triplets={sum(len(r['triplets']) for r in records)/len(records):.1f})")


if __name__ == "__main__":
    main()
