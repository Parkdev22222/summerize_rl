from summarize_rl.branches import (
    Example,
    Triplet,
    build_branches,
    format_glossary,
    serialize_triplets,
)
from summarize_rl.glossary import ActiveTerm


def test_serialize_grouped_by_head():
    triplets = [
        Triplet("1소대", "위치", "고지"),
        Triplet("1소대", "임무", "정찰"),
        Triplet("2소대", "임무", "방어"),
    ]
    s = serialize_triplets(triplets, group_by_head=True)
    assert "1소대: 위치 고지; 임무 정찰" in s
    assert "2소대: 임무 방어" in s


def test_serialize_flat():
    triplets = [Triplet("A", "r", "B")]
    s = serialize_triplets(triplets, group_by_head=False)
    assert s == "(A, r, B)"


def test_serialize_empty():
    assert serialize_triplets([]) == ""


def test_example_optional_metadata_fields_default_empty():
    # Backward-compatible positional construction (source, triplets) still works.
    ex = Example("원문", [Triplet("A", "r", "B")])
    assert ex.id is None
    assert ex.keyfacts == []
    assert ex.meta == {}


def test_example_carries_id_keyfacts_meta():
    ex = Example(
        "원문",
        [],
        id="10000",
        keyfacts=["핵심1", "핵심2"],
        meta={"split": "train"},
    )
    assert ex.id == "10000"
    assert ex.keyfacts == ["핵심1", "핵심2"]
    assert ex.meta == {"split": "train"}


def test_format_glossary():
    active = [ActiveTerm("기동"), ActiveTerm("정찰")]
    out = format_glossary(active)
    assert "기동" in out and "정찰" in out


def test_format_glossary_empty():
    assert format_glossary([]) == ""


def test_build_branches_contains_parts():
    ex = Example(
        source="적 부대가 이동 중이다.",
        triplets=[Triplet("적부대", "행동", "이동")],
        query="요약하시오.",
    )
    active = [ActiveTerm("기동")]
    b = ex_branches = build_branches(ex, active)
    assert "적 부대가 이동 중이다." in b.xq
    assert "적부대: 행동 이동" in b.sq
    assert "기동" in b.gq
    assert b.q.strip().endswith("요약하시오.")
    # Q branch must NOT leak source/triplet/glossary content.
    assert "적 부대가 이동" not in b.q
    assert "적부대" not in b.q
    assert "기동" not in b.q


def test_branch_dict_keys():
    ex = Example(source="x")
    b = build_branches(ex, [])
    assert set(b.as_dict().keys()) == {"XQ", "SQ", "GQ", "Q"}
