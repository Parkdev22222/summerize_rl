"""Generate a held-out TEST set matching data/scenarios_ko.jsonl in genre + format.

The model was trained on ALL of scenarios_ko.jsonl (no train/test split was held
out), so there is no unseen in-distribution test set. This generates fresh
records in the SAME genre (fictional modern blue-vs-red conflict analytical
briefs) and the SAME section structure (시나리오 개요 / 지형 분석 / 부대 구성 /
전술 요소 / 결정 지점), but with different countries/terrain/forces so the
content is unseen. Deterministic given --seed.

Schema mirrors the fields the pipeline uses plus reference fields:
    id, source_text, summary_text, keyfacts, triplets ([h,r,t]), split="test".
(source_text_en is omitted -- not read by diagnose/eval_gemini/train.)

Usage:
    python -m data.gen_test_scenarios --n 30 --out data/test_scenarios_ko.jsonl
    # evaluate the trained model on it (reference-free):
    GEMINI_API_KEY=... uv run -m examples.eval_gemini \
        --model <M> --ckpt checkpoints/best.pt --trust-remote-code \
        --data data/test_scenarios_ko.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

# Fictional states (distinct from the training set's 갈도비아/메타니아).
COUNTRIES = [
    "아르멘시아", "벨루타니아", "노르덴 공화국", "제리스탄", "오발리아", "테라노바",
    "실바니아", "카르파티아", "베가란트", "이스카니아", "모르비아", "산크투스 연방",
    "레반티아", "코르다바", "아젠타", "보레아", "돌로미아", "카레시아",
]
REGIONS = [
    "카레나", "벨포트", "노바그라드", "산텔로", "리엔츠", "포르비크", "델가도",
    "하벤슈타트", "오르시아", "칼레아", "브란트", "세리온", "미르가",
]

# Terrain archetype -> tailored descriptions.
TERRAIN = {
    "도시": {
        "유형": "밀집된 건물과 좁은 거리로 특징지어지는 도시 지형",
        "특징": "주요 교차로의 애로점과 고층 건물의 감제 관측, 잔해에 의한 엄폐가 풍부하다",
        "기동": "좁은 도로와 보행로가 주 이동로이며, 잔해 장벽이 차량 기동을 제한한다",
    },
    "산악": {
        "유형": "가파른 능선과 계곡으로 이뤄진 산악 지형",
        "특징": "고지가 감제 관측과 방어 우위를 제공하며 접근로가 능선으로 제한된다",
        "기동": "능선과 계곡길이 주 기동로이고 차량 접근이 크게 제약된다",
    },
    "삼림": {
        "유형": "밀림과 관목으로 시야가 제한되는 삼림 지형",
        "특징": "은폐·엄폐가 풍부하나 관측과 통신이 제한된다",
        "기동": "임도와 소로가 주 기동로이며 매복에 취약하다",
    },
    "하천삼각주": {
        "유형": "다수의 수로와 습지로 분할된 하천 삼각주 지형",
        "특징": "교량과 도하점이 핵심 애로점이며 습지가 기동을 제한한다",
        "기동": "제방과 교량이 주 기동로이고 도하 장비가 필요하다",
    },
    "해안": {
        "유형": "항만과 저지대가 혼재한 해안 지형",
        "특징": "항만 시설과 방파제가 핵심 지형이며 상륙 접근이 가능하다",
        "기동": "해안도로가 주 기동로이며 조수 간만이 상륙에 영향을 준다",
    },
    "사막": {
        "유형": "개활지와 모래언덕이 펼쳐진 사막 지형",
        "특징": "은폐물이 적어 장거리 관측·교전이 유리하다",
        "기동": "개활지라 기계화 기동이 용이하나 보급선이 길어진다",
    },
}
WEATHER = [
    "흐린 하늘과 간헐적인 비로 가시거리가 약 200m로 감소한다",
    "짙은 안개로 가시거리가 100m 이하이며 항공 지원이 제한된다",
    "맑고 건조하여 가시거리가 양호하나 한낮 열기가 장비에 영향을 준다",
    "강설과 결빙으로 기동성이 저하되고 저체온 위험이 있다",
    "야간·저조도 조건으로 야시장비 의존도가 높다",
]
CIV = [
    "민간인이 잔류해 있어 부수적 피해 최소화 규칙이 엄격히 적용된다",
    "대피가 대부분 완료되었으나 산발적 민간 통행이 관측된다",
    "핵심 기반시설(상수도·전력)이 전투로 손상 위험에 노출되어 있다",
]

UNIT_TYPES = [
    "기계화보병대대", "보병여단", "기갑연대", "공수대대", "해병연대",
    "산악보병대대", "차량화보병대대", "특수전여단", "경보병연대",
]
WEAPONS_HEAVY = [
    "주력전차 {n}대", "보병전투차 {n}대", "자주포 {n}문", "공격헬기 {n}대",
    "다연장로켓 {n}문", "장갑차 {n}대", "대전차미사일 {n}기",
]
WEAPONS_LIGHT = [
    "소화기와 분대지원화기", "박격포와 대전차로켓", "저격소총과 유탄발사기",
    "휴대용 대공미사일", "기관총과 클레이모어",
]
LOGI = [
    "탄약·의료 물자는 충분하나 연료가 정원의 {p}% 수준이다",
    "재보급이 지연되어 {d}일치 탄약만 보유한 것으로 판단된다",
    "보급선이 안정적이며 {d}일 이상 지속 작전이 가능하다",
    "의료 후송 여건이 제한되어 중상자 처리가 어렵다",
]
COMMS = ["암호화 무전기와 정찰 드론을 완비함", "통신이 간헐적으로 두절됨",
         "유선·무선 이중 통신망을 운용함", "전자전으로 통신 교란을 받고 있음"]
TRAIN = ["전투 경험이 풍부한 정예", "훈련 수준이 높으나 실전 경험은 제한적",
         "예비 병력 위주로 실전 경험이 적음", "혼합된 훈련 수준"]
MORALE = ["사기가 높음", "사기는 높으나 보급 우려가 있음",
          "장기 교전으로 사기가 저하됨", "방어 의지는 강하나 불안감이 있음"]

PHASES = ["접근 및 정찰 단계", "초기 접촉 및 화력 준비 단계",
          "본격 교전 단계", "돌파 시도 단계", "방어 및 지연 단계"]
ROE = ["인구 밀집 지역 중화기 사용 제한", "국경선 월경 금지",
       "항공 화력은 표적 확인 후에만 허용", "포로 처우 규정 엄격 준수"]

COA_ATT = [
    "제병협동으로 도심/고지 소탕을 실시한다",
    "측방으로 포위망을 형성해 적을 고립시킨다",
    "화력 우위를 활용해 정면 돌파를 시도한다",
    "야간 침투로 후방 지휘시설을 타격한다",
    "기만 작전으로 예비대를 유인한다",
]
COA_DEF = [
    "거점을 요새화하고 치고 빠지기로 소모를 강요한다",
    "종심 방어로 돌파를 지연시키며 예비대를 보존한다",
    "취약 시간대에 국지 반격을 실시한다",
    "장애물과 매복으로 접근로를 통제한다",
    "기동 방어로 포위를 회피한다",
]


def _has_jong(word: str) -> bool:
    """Does the last char have a final consonant (받침)?"""
    ch = word[-1]
    return ("가" <= ch <= "힣") and (ord(ch) - 0xAC00) % 28 != 0


def _eul(w): return w + ("을" if _has_jong(w) else "를")
def _neun(w): return w + ("은" if _has_jong(w) else "는")
def _ga(w): return w + ("이" if _has_jong(w) else "가")
def _wa(w): return w + ("과" if _has_jong(w) else "와")


def _ro(w):  # 로 after a vowel or ㄹ, else 으로
    ch = w[-1]
    if not ("가" <= ch <= "힣"):
        return w + "로"
    jong = (ord(ch) - 0xAC00) % 28
    return w + ("로" if jong in (0, 8) else "으로")


def _pct(rng): return rng.choice([40, 50, 60, 70])
def _days(rng): return rng.choice([2, 3, 4, 5])


def _force_block(rng, name, role):
    utype = rng.choice(UNIT_TYPES)
    size = rng.choice([600, 800, 1000, 1200, 1500])
    ncoy = rng.choice([2, 3, 4])
    heavy = rng.choice(WEAPONS_HEAVY).format(n=rng.choice([3, 4, 5, 6, 8, 10]))
    light = rng.choice(WEAPONS_LIGHT)
    logi = rng.choice(LOGI).format(p=_pct(rng), d=_days(rng))
    comms = rng.choice(COMMS)
    train = rng.choice(TRAIN)
    morale = rng.choice(MORALE)
    text = (
        f"#### {role} ({name}):\n"
        f"- **부대 유형/규모/편성:** {utype} {size:,}명, {ncoy}개 중대로 편성됨.\n"
        f"- **무기 체계:** {light}, {_ro(heavy)} 무장.\n"
        f"- **군수 현황:** {logi}\n"
        f"- **통신 능력:** {comms}.\n"
        f"- **훈련 수준 및 경험:** {train}.\n"
        f"- **사기 및 심리 상태:** {morale}.\n"
    )
    facts = [
        f"{_neun(name)} {utype} {size:,}명으로 편성된다.",
        f"{_neun(name)} {_eul(heavy)} 보유한다.",
    ]
    triplets = [
        [name, "규모", f"{utype} {size:,}명"],
        [name, "편성", f"{ncoy}개 중대"],
        [name, "보유", heavy],
        [name, "무장", light],
        [name, "군수", logi],
    ]
    meta = {"name": name, "utype": utype, "size": size, "heavy": heavy}
    return text, facts, triplets, meta


def make_scenario(rng: random.Random, seq: int) -> dict:
    blue, red = rng.sample(COUNTRIES, 2)
    region = rng.choice(REGIONS)
    terr_key = rng.choice(list(TERRAIN))
    terr = TERRAIN[terr_key]
    year = rng.choice([2024, 2025, 2026, 2027])
    hour = rng.choice(["0430", "0600", "1400", "1900", "2300"])
    dur = _days(rng)
    weather = rng.choice(WEATHER)
    civ = rng.choice(CIV)
    phase = rng.choice(PHASES)
    roe = rng.choice(ROE)
    blue_goal = rng.choice(["국경 지역 통제권 확보", "핵심 항만 장악", "적 지휘부 무력화",
                            "보급 요충지 점령", "포위망 형성"])
    red_goal = rng.choice(["영토 방어 및 지역 안정", "핵심 도시 사수", "반격을 통한 실지 회복",
                           "지연전으로 증원 확보", "주민 보호"])
    blue_coas = rng.sample(COA_ATT, 3)
    red_coas = rng.sample(COA_DEF, 3)

    b_text, b_facts, b_trip, bm = _force_block(rng, blue, "블루 포스")
    r_text, r_facts, r_trip, rm = _force_block(rng, red, "레드 포스")

    source_text = (
        "### 시나리오 개요:\n"
        f"- **서사적 배경:** {year}년, {_wa(blue)} {red} 간 분쟁이 {region} 일대에서 "
        f"발발한다. {_neun(blue)} {_eul(blue_goal)} 목표로 하고, {_neun(red)} {_eul(red_goal)} "
        "추구한다. 교전은 고강도 재래전 양상을 띤다.\n"
        f"- **시간 요소:** 시나리오는 약 {hour}시에 시작된다.\n"
        f"- **교전 지속 기간:** 예상 {dur}일이며 초기 접촉 후 지속 교전으로 전환된다.\n\n"
        "### 지형 분석:\n"
        f"- **주요 지형 유형:** {region}의 {terr['유형']}이다.\n"
        f"- **주요 지형 특징:** {terr['특징']}.\n"
        f"- **기동로 및 장애물:** {terr['기동']}.\n"
        f"- **기상 조건 및 가시성:** {weather}.\n"
        f"- **민간인 존재 및 기반 시설:** {civ}.\n\n"
        "### 부대 구성:\n"
        f"{b_text}\n{r_text}\n"
        "### 전술 요소:\n"
        f"- **현재 작전 단계:** {_neun(blue)} {phase}에 있다.\n"
        f"- **적 위치/배치에 대한 알려진 정보:** {_neun(red)} 주요 애로점에 방어 거점과 "
        "매복 진지를 구축한 것으로 판단된다.\n"
        f"- **교전 규칙 제약:** {roe}.\n"
        "- **양측의 잠재적 행동 방책(COA):**\n"
        f"  - **블루 포스:**\n    1. {blue_coas[0]}\n    2. {blue_coas[1]}\n    3. {blue_coas[2]}\n"
        f"  - **레드 포스:**\n    1. {red_coas[0]}\n    2. {red_coas[1]}\n    3. {red_coas[2]}\n\n"
        "### 결정 지점:\n"
        f"1. **블루 포스:** 즉시 공세로 전환할지, 방어선을 먼저 구축할지 결정한다.\n"
        "   - **주요 변수:** 군수 지원, 매복 가능성, 민간인 위험, 기상.\n"
        "   - **2차 효과:** 즉시 공세는 사상자를 늘리나 신속한 목표 달성이 가능하다.\n"
        f"2. **레드 포스:** 정적 방어에 집중할지, 기동 방어로 전환할지 결정한다.\n"
        "   - **주요 변수:** 병력 사기, 적 능력 정보, 방어 우위 상실 위험.\n"
        "   - **2차 효과:** 정적 방어는 결속을 다지나 포위될 수 있다.\n"
        f"3. **양측:** {terr_key} 환경에서 화력 지원의 효과와 부수 피해 위험을 판단한다.\n"
        "   - **주요 변수:** 가용 화력, 표적 정보, 민간인 규모.\n"
        "   - **2차 효과:** 효과적 화력은 방어를 무력화하나 정당성 훼손 위험이 있다.\n"
    )

    summary_text = (
        f"{year}년 {_wa(blue)} {red} 간 분쟁이 {region}의 {terr_key} 지형에서 발생했다. "
        f"{_neun(blue)} {bm['utype']} {bm['size']:,}명과 {_eul(bm['heavy'])} 운용하며 "
        f"{_eul(blue_goal)} 목표로 {phase}에 있다. {_neun(red)} {rm['utype']} {rm['size']:,}명으로 "
        f"{_eul(red_goal)} 위해 거점을 방어 중이다. {weather.split('로')[0]}로 가시성이 제약되며 "
        f"{_ga(roe)} 적용된다."
    )

    keyfacts = [
        f"{year}년 {_wa(blue)} {red} 간 분쟁이 {region}에서 발생했다.",
        f"교전 지형은 {terr_key}이며 {terr['유형']}이다.",
        f"{blue}의 목표는 {blue_goal}이다.",
        f"{red}의 목표는 {red_goal}이다.",
        *b_facts, *r_facts,
        f"현재 {_neun(blue)} {phase}에 있다.",
        f"교전 규칙으로 {_ga(roe)} 적용된다.",
        f"기상은 {weather}.",
        f"예상 교전 지속 기간은 {dur}일이다.",
    ]
    triplets = [
        [blue, "목표", blue_goal],
        [red, "목표", red_goal],
        ["교전지역", "위치", f"{region} {terr_key}"],
        ["작전단계", "현황", phase],
        ["교전규칙", "제약", roe],
        *b_trip, *r_trip,
    ]
    # dedupe triplets, preserve order
    seen, uniq = set(), []
    for t in triplets:
        k = tuple(t)
        if k not in seen:
            seen.add(k); uniq.append(t)

    return {
        "id": 10000 + seq,  # offset so ids never collide with the training set
        "source_text": source_text,
        "summary_text": summary_text,
        "keyfacts": keyfacts,
        "triplets": uniq,
        "split": "test",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a held-out test set like scenarios_ko.jsonl.")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out", default="data/test_scenarios_ko.jsonl")
    p.add_argument("--seed", type=int, default=2027)
    args = p.parse_args()

    rng = random.Random(args.seed)
    seen_sig, records, seq, attempts = set(), [], 0, 0
    while len(records) < args.n and attempts < args.n * 60:
        attempts += 1
        rec = make_scenario(rng, seq)
        sig = rec["source_text"][:500]
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        records.append(rec)
        seq += 1

    with open(args.out, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    avg = sum(len(r["triplets"]) for r in records) / len(records)
    print(f"wrote {len(records)} test scenarios -> {args.out} (avg triplets={avg:.1f})")


if __name__ == "__main__":
    main()
