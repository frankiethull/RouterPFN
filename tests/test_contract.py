"""RouterPFN API contract tests (MIT): sklearn-compatible difficulty 1–5."""

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from router_pfn.engines import available_engines, get_engine
from router_pfn.router import RouterPFNClassifier
from router_pfn.synth import generate


def _toy_questions(n=60, seed=0):
    rows = generate(n=n, seed=seed)
    return [r["question"] for r in rows], [r["difficulty"] for r in rows]


def test_fit_predict_proba_predict():
    X, y = _toy_questions()
    clf = RouterPFNClassifier(engine="dummy").fit(X, y)
    assert list(clf.classes_) == [1, 2, 3, 4, 5]
    assert clf.n_features_in_ == 24
    P = clf.predict_proba(X)
    assert P.shape == (len(X), 5)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    pred = clf.predict(X)
    assert set(np.unique(pred)).issubset({1, 2, 3, 4, 5})


def test_route_returns_roster_id():
    X, y = _toy_questions()
    roster = ["opencode/muse-spark-1.3-free", "ollama/gemma4"]
    clf = RouterPFNClassifier(engine="dummy", roster=roster).fit(X, y)
    r = clf.route("Prove every even number > 2 is the sum of two primes, with edge cases.")
    assert r["model"] in roster
    assert r["predicted_difficulty"] in (1, 2, 3, 4, 5)
    assert abs(sum(r["difficulty_proba"].values()) - 1.0) < 1e-6


def test_three_model_roster():
    X, y = _toy_questions()
    roster = ["cheap", "standard", "frontier"]
    clf = RouterPFNClassifier(engine="dummy", roster=roster).fit(X, y)
    assert clf.route_many(["hi", "solve x^2 - 5x + 6 = 0 with proof"])[0]["model"] in roster


def test_five_model_roster():
    X, y = _toy_questions()
    roster = [f"m{i}" for i in range(1, 6)]
    clf = RouterPFNClassifier(engine="dummy", roster=roster).fit(X, y)
    assert clf.route("What is 2+2?")["model"] in roster
    with pytest.raises(ValueError):
        RouterPFNClassifier(roster=["m1"]).fit(X, y)


def test_pipeline_and_cv_compatible():
    X, y = _toy_questions(n=40)
    pipe = Pipeline([("clf", RouterPFNClassifier(engine="dummy"))])
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(X)
    scores = cross_val_score(RouterPFNClassifier(engine="dummy"), X, y, cv=2)
    assert scores.shape == (2,)


def test_get_set_params():
    clf = RouterPFNClassifier(policy="threshold", confidence_threshold=0.7)
    assert clf.get_params()["confidence_threshold"] == 0.7
    clf.set_params(policy="conservative")
    assert clf.policy == "conservative"


def test_rejects_bad_labels_and_roster():
    X, _ = _toy_questions(n=10)
    with pytest.raises(ValueError):
        RouterPFNClassifier().fit(X, [0] * 10)
    with pytest.raises(ValueError):
        RouterPFNClassifier(roster=["only"]).fit(X, [1] * 10)


def test_engine_registry_and_missing_dep_error():
    assert set(available_engines()) >= {"dummy", "tabpfn", "tabicl", "kumo", "fiorino"}
    assert get_engine("dummy").name == "dummy"
    with pytest.raises(ValueError):
        get_engine("nope")
