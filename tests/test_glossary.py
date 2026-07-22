from summarize_rl.glossary import ActiveTerm, Glossary


GLOSSARY = {
    "기동": ["움직임", "이동", "maneuver"],
    "화력지원": ["포격 지원", "fire support"],
    "정찰": ["수색"],
}


def test_trigger_activates_term():
    g = Glossary(GLOSSARY)
    active = g.active_terms("부대의 이동이 관측되었다.")
    assert "기동" in active
    assert "화력지원" not in active


def test_term_is_its_own_trigger():
    g = Glossary(GLOSSARY)
    active = g.active_terms("정찰 임무를 수행했다.")
    assert "정찰" in active


def test_no_trigger_no_activation():
    g = Glossary(GLOSSARY)
    active = g.gate("특이사항 없음.")
    assert active == []


def test_case_insensitive_english_trigger():
    g = Glossary(GLOSSARY, case_insensitive=True)
    active = g.active_terms("Fire Support was requested.")
    assert "화력지원" in active


def test_matched_triggers_recorded():
    g = Glossary(GLOSSARY)
    active = g.gate("적의 움직임과 이동 경로를 파악했다.")
    entry = next(t for t in active if t.term == "기동")
    assert isinstance(entry, ActiveTerm)
    assert "움직임" in entry.matched_triggers
    assert "이동" in entry.matched_triggers


def test_similarity_matcher_injection():
    class FakeMatcher:
        def matches(self, trigger, text):
            return trigger == "fire support" and "포탄" in text

    g = Glossary(GLOSSARY, similarity_matcher=FakeMatcher())
    # "포탄" doesn't literally contain any trigger, but the fake matcher fires.
    active = g.active_terms("포탄이 떨어졌다.")
    assert "화력지원" in active
