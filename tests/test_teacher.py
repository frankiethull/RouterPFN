"""Teacher-helper unit tests (MIT). All offline — no Ollama, no Laya."""

from router_pfn.teacher import (
    ALL_DOMAINS,
    SUB_DOMAINS,
    _clean_lines,
    _near_dup,
    _norm,
    consensus_counts,
    parent_of,
    rate_question,
)


def test_norm_basic():
    assert _norm("  What is X?! ") == "what is x"
    assert _norm("Solve x^2=4...") == "solve x24"


def test_clean_lines_drops_meta():
    text = (
        "What is the capital of Japan?\n"
        "I need to generate two questions at difficulty 4/5.\n"
        "1. How many sides does a hexagon have?\n"
        "Hi\n"
    )
    lines = _clean_lines(text)
    assert lines == [
        "What is the capital of Japan?",
        "How many sides does a hexagon have?",
    ]


def test_near_dup_detection():
    kept = [set(_norm("What is the capital city of Japan?").split())]
    assert _near_dup("What is the capital city of Japan!", kept)
    assert _near_dup("what is the capital city of japan", kept)
    assert not _near_dup("Prove the Riemann hypothesis with edge cases", kept)
    assert _near_dup("", kept)


def test_sub_domains_map_to_charter_parents():
    assert set(ALL_DOMAINS) >= set(SUB_DOMAINS)
    for sub, parent in SUB_DOMAINS.items():
        assert parent_of(sub) == parent
    assert parent_of("knowledge") == "knowledge"


def test_consensus_counts():
    rows = [{"consensus": 1}, {"consensus": 3}, {"consensus": 3}, {}]
    assert consensus_counts(rows) == {1: 1, 2: 0, 3: 3, 4: 0, 5: 0}


def test_rate_question_llm_primary_no_network(monkeypatch):
    import router_pfn.teacher as t

    monkeypatch.setattr(t, "llm_judge", lambda model, q: {"a": 2, "b": 4}[model])
    monkeypatch.setattr(
        t, "laya_rate", lambda q: {"expected": 2.9, "proba": {"3": 1.0}})
    row = rate_question("q?", "knowledge", 5, "gen", ["a", "b"], seed=0)
    assert row["consensus"] == 4  # median of LLM judges, not the mean with Laya
    assert row["parent"] == "knowledge"
    assert row["spread"] == 4.0 - 2.0

    row2 = rate_question("q?", "code_review", 1, "gen", [], seed=0,
                         use_laya=False)
    assert row2["consensus"] == 1  # falls back to intended
    assert row2["parent"] == "programming"
