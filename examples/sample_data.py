"""Small synthetic corpus + military glossary for demos and smoke runs.

This stands in for the real KG source/triplet corpus and the real standard-term
glossary, which are out of scope for the core-algorithm library. Swap these for
real data loaders when integrating.
"""

from summarize_rl.branches import Example, Triplet
from summarize_rl.glossary import Glossary

# 표준용어 <- 트리거(동의어/구어/개념)
MILITARY_GLOSSARY = Glossary(
    {
        "기동": ["이동", "움직임", "전진", "maneuver"],
        "정찰": ["수색", "정탐", "recon"],
        "화력지원": ["포격", "포병 지원", "fire support"],
        "방어": ["수비", "저지", "defense"],
        "점령": ["장악", "확보", "occupy"],
    }
)

EXAMPLES = [
    Example(
        source="적 1개 중대가 야간을 틈타 능선을 따라 이동하였고, 이후 145고지를 장악하였다. "
        "아군은 포병 지원을 요청하여 대응하였다.",
        triplets=[
            Triplet("적중대", "행동", "이동"),
            Triplet("적중대", "점령", "145고지"),
            Triplet("아군", "요청", "포병지원"),
        ],
    ),
    Example(
        source="정찰조가 수색 임무를 수행하던 중 적 기갑부대의 전진을 관측하였다. "
        "지휘관은 저지 진지를 편성하도록 지시하였다.",
        triplets=[
            Triplet("정찰조", "임무", "수색"),
            Triplet("적기갑부대", "행동", "전진"),
            Triplet("지휘관", "지시", "저지진지"),
        ],
    ),
    Example(
        source="아군 소대는 하천 도하 후 목표 지역으로 움직였으며, 적의 화력을 확보된 엄폐물로 회피하였다.",
        triplets=[
            Triplet("아군소대", "행동", "도하"),
            Triplet("아군소대", "행동", "움직임"),
        ],
    ),
]
