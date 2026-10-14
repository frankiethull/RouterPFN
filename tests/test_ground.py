"""Grounding unit tests (MIT). All offline — no Ollama, no Laya."""

from router_pfn.ground import pick_subset, weakest_tier


def test_weakest_tier():
    assert weakest_tier(True, True) == "weak"
    assert weakest_tier(True, False) == "weak"
    assert weakest_tier(False, True) == "strong"
    assert weakest_tier(False, False) == "strong"  # safe default


def test_pick_subset_stratified():
    rows = [{"id": f"{d}-{i}", "consensus": d} for d in (1, 2, 3, 4, 5) for i in range(10)]
    sub = pick_subset(rows, 3, seed=0)
    assert len(sub) == 15
    from collections import Counter
    assert Counter(r["consensus"] for r in sub) == {1: 3, 2: 3, 3: 3, 4: 3, 5: 3}


def test_pick_subset_short_level():
    rows = [{"id": "a", "consensus": 1}]
    assert len(pick_subset(rows, 30, seed=0)) == 1


def test_pick_subset_deterministic():
    rows = [{"id": str(i), "consensus": (i % 5) + 1} for i in range(50)]
    assert pick_subset(rows, 4, seed=7) == pick_subset(rows, 4, seed=7)
