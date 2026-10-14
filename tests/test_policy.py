"""RouterPFN policy tests (MIT): tier mapping for N=2 and N=3 rosters."""

import numpy as np
import pytest

from router_pfn.policy import apply_policy, difficulty_to_tier


def test_two_model_default_cut():
    assert difficulty_to_tier(1, 2) == 0
    assert difficulty_to_tier(3, 2) == 0
    assert difficulty_to_tier(4, 2) == 1
    assert difficulty_to_tier(5, 2) == 1


def test_three_model_default_cuts():
    assert difficulty_to_tier(2, 3) == 0
    assert difficulty_to_tier(3, 3) == 1
    assert difficulty_to_tier(4, 3) == 1
    assert difficulty_to_tier(5, 3) == 2


def test_custom_bounds():
    assert difficulty_to_tier(2, 2, [2]) == 0
    assert difficulty_to_tier(3, 2, [2]) == 1


def test_four_and_five_model_defaults():
    assert [difficulty_to_tier(d, 4) for d in (1, 2, 3, 4, 5)] == [0, 0, 1, 2, 3]
    assert [difficulty_to_tier(d, 5) for d in (1, 2, 3, 4, 5)] == [0, 1, 2, 3, 4]


def test_custom_bounds_n4():
    assert difficulty_to_tier(3, 4, [1, 2, 4]) == 2
    assert difficulty_to_tier(5, 4, [1, 2, 4]) == 3
    with pytest.raises(ValueError):
        difficulty_to_tier(3, 4, [2])  # need N-1 cuts


def test_rejects_bad_sizes():
    for bad in (1, 6):
        with pytest.raises(ValueError):
            difficulty_to_tier(3, bad)


def test_argmax_routes_strong_on_hard():
    p = np.array([0.05, 0.05, 0.1, 0.3, 0.5])
    model, conf, esc = apply_policy(p, ["weak", "strong"])
    assert model == "strong" and esc is False
    assert conf == pytest.approx(0.8)


def test_threshold_escalates_on_low_confidence():
    p = np.array([0.25, 0.25, 0.2, 0.15, 0.15])  # weak mass 0.7... make it unsure
    q = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # weak 0.6 vs strong 0.4
    model, conf, esc = apply_policy(
        np.array([0.3, 0.15, 0.1, 0.225, 0.225]),
        ["weak", "strong"],
        policy="threshold",
        confidence_threshold=0.6,
    )
    assert model == "strong" and esc is True
    assert q.sum() == pytest.approx(1.0) and p.sum() == pytest.approx(1.0)


def test_conservative_escalates_on_close_call():
    p = np.array([0.2, 0.2, 0.11, 0.24, 0.25])  # weak 0.51 vs strong 0.49
    model, _, esc = apply_policy(p, ["weak", "strong"], policy="conservative")
    assert model == "strong" and esc is True


def test_five_model_routing():
    roster = ["m1", "m2", "m3", "m4", "m5"]
    model, _, esc = apply_policy(np.array([0.02, 0.03, 0.05, 0.1, 0.8]), roster)
    assert model == "m5" and esc is False
    model, _, _ = apply_policy(np.array([0.8, 0.1, 0.05, 0.03, 0.02]), roster)
    assert model == "m1"


def test_rejects_bad_roster_and_proba():
    with pytest.raises(ValueError):
        apply_policy(np.array([0.2] * 5), ["only-one"])
    with pytest.raises(ValueError):
        apply_policy(np.array([0.2] * 5), [f"m{i}" for i in range(6)])
    with pytest.raises(ValueError):
        apply_policy(np.array([0.5, 0.5]), ["weak", "strong"])
    with pytest.raises(ValueError):
        apply_policy(np.array([0.2] * 5), ["weak", "strong"], policy="nope")
