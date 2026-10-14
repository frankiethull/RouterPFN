"""Seed-corpus unit tests (MIT). Offline — no models beyond dummy engine."""

from router_pfn.router import RouterPFNClassifier
from router_pfn.seed import build_seed, seed_questions_labels


def _rows():
    rows = []
    for d in (1, 2, 3, 4, 5):
        for i in range(6):
            rows.append({"id": f"{d}-{i}", "question": f"q{d}-{i}",
                         "consensus": d, "spread": float(i),
                         "domain": "knowledge"})
    return rows


def test_build_seed_stratified_lowest_spread_first():
    seed = build_seed(_rows(), n_per_level=2, seed=0)
    assert len(seed) == 10
    from collections import Counter
    assert Counter(r["difficulty"] for r in seed) == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    # lowest spread wins within each level
    assert {r["question"] for r in seed if r["difficulty"] == 3} == {"q3-0", "q3-1"}
    # slim schema only
    assert set(seed[0]) == {"question", "difficulty", "domain"}


def test_build_seed_short_level_takes_all():
    assert len(build_seed([{"id": "a", "question": "q", "consensus": 1}], 40)) == 1


def test_bundled_seed_loads_and_stratifies():
    questions, labels = seed_questions_labels()
    assert len(questions) == 200
    from collections import Counter
    assert Counter(labels) == {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}
    assert all(q.strip() for q in questions)


def test_from_seed_zero_data_onboarding():
    clf = RouterPFNClassifier.from_seed(
        engine="dummy", roster=["weak-local", "strong-frontier"])
    r = clf.route("What is the capital of Japan?")
    assert r["model"] in ("weak-local", "strong-frontier")
    assert r["predicted_difficulty"] in (1, 2, 3, 4, 5)
