"""Laya-label unit tests (MIT). All offline — no models, no network."""

from router_pfn.labels import (
    agreement,
    apply_laya_labels,
    fit_laya_calibration,
    train_test_split,
)


def _rows():
    return [
        {"consensus": 1, "laya_expected": 2.0, "laya_proba": {"1": 0.6, "2": 0.4}},
        {"consensus": 2, "laya_expected": 2.4, "laya_proba": {"2": 0.7, "3": 0.3}},
        {"consensus": 3, "laya_expected": 2.7, "laya_proba": {"3": 0.8, "2": 0.2}},
        {"consensus": 4, "laya_expected": 3.0, "laya_proba": {"3": 0.5, "4": 0.5}},
        {"consensus": 5, "laya_expected": 3.3, "laya_proba": {"4": 0.6, "3": 0.4}},
        {"consensus": 3, "laya_expected": 2.8, "laya_proba": {"3": 0.9}},
        {"consensus": 4, "laya_expected": 3.1, "laya_proba": {"4": 0.9}},
        {"consensus": 2, "laya_expected": 2.5, "laya_proba": {"2": 0.9}},
        {"consensus": 1, "laya_expected": 2.1, "laya_proba": {"1": 0.9}},
        {"consensus": 5, "laya_expected": 3.4, "laya_proba": {"4": 0.9}},
    ]


def test_split_matches_bench_convention():
    rows = _rows()
    train, test = train_test_split(rows)
    assert len(train) == 7 and len(test) == 3
    assert train == rows[:7] and test == rows[7:]


def test_calibration_is_monotonic_and_bounded():
    train, _ = train_test_split(_rows())
    a, b = fit_laya_calibration(train)
    assert a > 0  # higher expected -> higher difficulty
    labeled = apply_laya_labels(_rows(), a, b)
    cals = [r["laya_calibrated"] for r in labeled]
    assert all(1 <= c <= 5 for c in cals)
    assert cals[0] <= cals[-1]  # ordering preserved


def test_argmax_maps_0index_to_1to5():
    labeled = apply_laya_labels(_rows(), 1.0, 0.0)
    assert labeled[0]["laya_argmax"] == 1
    assert labeled[2]["laya_argmax"] == 3


def test_agreement():
    rows = [{"a": 1, "b": 1}, {"a": 1, "b": 2}]
    assert agreement(rows, "a", "b") == 0.5
    assert agreement([], "a", "b") == 0.0
