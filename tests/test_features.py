"""RouterPFN feature tests (MIT): deterministic, cheap, no network."""

import time

from router_pfn.features import FEATURE_NAMES, featurize


def test_feature_names_stable():
    assert len(FEATURE_NAMES) == 24
    assert "n_tokens" in FEATURE_NAMES and "programming_hit" in FEATURE_NAMES
    assert "trivial_op_hit" in FEATURE_NAMES and "bracket_data_hit" in FEATURE_NAMES


def test_empty_question_safe():
    f = featurize("")
    assert f["n_chars"] == 0.0 and f["n_tokens"] == 0.0
    assert all(v == v for v in f.values())  # no NaN


def test_code_and_math_detected():
    f = featurize("Fix this: ```python\nint(x['a'])\n``` Traceback KeyError, solve x^2 - 1 = 0?")
    assert f["has_code_fence"] == 1.0
    assert f["debugging_hit"] == 1.0
    assert f["math_symbol_hits"] >= 2.0


def test_domain_hits_fire():
    assert featurize("Write a Python function that parses CSV")["programming_hit"] == 1.0
    assert featurize("Summarize this essay in two sentences")["writing_hit"] == 1.0
    assert featurize("It depends, the request is vague")["ambiguous_hit"] == 1.0


def test_trivial_op_and_bracket_hits():
    f = featurize("If I have a list of numbers [3, 1, 4, 1, 5], sort them descending?")
    assert f["trivial_op_hit"] == 1.0
    assert f["bracket_data_hit"] == 1.0
    g = featurize("Prove every even number > 2 is the sum of two primes.")
    assert g["trivial_op_hit"] == 0.0
    assert g["bracket_data_hit"] == 0.0


def test_featurize_is_fast():
    qs = ["What is the capital of France?"] * 200
    t0 = time.perf_counter()
    for q in qs:
        featurize(q)
    dt = (time.perf_counter() - t0) / len(qs)
    assert dt < 0.005  # 5ms/query budget keeps router cheaper than savings
