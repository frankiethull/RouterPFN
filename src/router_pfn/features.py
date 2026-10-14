"""Deterministic question features for RouterPFN (MIT).

v0.1 posture: zero model calls, zero network. Every feature derives
from the raw question string so extraction cost never erases routing
savings (charter Phase 5 guardrail).
"""

from __future__ import annotations

import math
import re

FEATURE_NAMES: list[str] = [
    "n_chars",
    "n_tokens",
    "n_sentences",
    "avg_token_len",
    "n_question_marks",
    "n_exclamations",
    "has_code_fence",
    "n_code_spans",
    "math_symbol_hits",
    "digit_ratio",
    "uppercase_ratio",
    "readability_proxy",
    "knowledge_hit",
    "reasoning_hit",
    "math_hit",
    "programming_hit",
    "debugging_hit",
    "ml_hit",
    "writing_hit",
    "analysis_hit",
    "multistep_hit",
    "ambiguous_hit",
    "trivial_op_hit",
    "bracket_data_hit",
]

_MATH_SYMS = re.compile(r"[+\-*/=^<>∑∫√%±×÷]")
_CODE_SPAN = re.compile(r"`[^`]+`")
_SENT_SPLIT = re.compile(r"[.!?]+\s+")
# Inline data that looks technical but asks for a trivial operation
# (charter Phase 10 "deceptive/simple-looking" bucket, e.g. sorting a
# 5-element list). Helps the router distrust surface complexity.
_BRACKET_DATA = re.compile(r"\[[^\]]*\d[^\]]*\]")
_TRIVIAL_OP = re.compile(
    r"\b(sort(ing|ed)?|count|list|name|define|convert|"
    r"what (is|are) the (capital|formula|symbol|color|chemical)|"
    r"translate\b.{0,40}\binto\b)",
    re.I,
)

_DOMAIN_PATTERNS: dict[str, re.Pattern] = {
    "knowledge_hit": re.compile(
        r"\b(who|when|where|capital|history|century|president|king|queen|war)\b",
        re.I,
    ),
    "reasoning_hit": re.compile(
        r"\b(if|then|because|therefore|however|prove|logic|puzzle|riddle)\b", re.I
    ),
    "math_hit": re.compile(
        r"\b(solve|equation|integral|derivative|theorem|calculate|probability)\b", re.I
    ),
    "programming_hit": re.compile(
        r"\b(function|class|python|code|algorithm|compile|api|regex|sql)\b", re.I
    ),
    "debugging_hit": re.compile(
        r"\b(error|traceback|bug|fails?|exception|stack|debug|fix)\b", re.I
    ),
    "ml_hit": re.compile(
        r"\b(model|dataset|train|accuracy|regression|classifier|embedding|llm)\b", re.I
    ),
    "writing_hit": re.compile(
        r"\b(essay|poem|story|rewrite|summar|email|letter|blog)\b", re.I
    ),
    "analysis_hit": re.compile(
        r"\b(analy|compare|contrast|evaluate|critique|tradeoff|pros|cons)\b", re.I
    ),
    "multistep_hit": re.compile(
        r"\b(first|then|next|finally|step[- ]by[- ]step|plan|multi[- ]step)\b", re.I
    ),
    "ambiguous_hit": re.compile(
        r"\b(maybe|perhaps|ambiguous|unclear|vague|depends|it depends)\b", re.I
    ),
}


def featurize(question: str) -> dict[str, float]:
    """Featurize one question string into a flat {name: value} dict."""
    q = question or ""
    n_chars = len(q)
    tokens = q.split()
    n_tokens = len(tokens)
    avg_token_len = (sum(len(t) for t in tokens) / n_tokens) if n_tokens else 0.0
    n_sentences = len([s for s in _SENT_SPLIT.split(q) if s.strip()]) or (1 if q else 0)
    n_digits = sum(c.isdigit() for c in q)
    n_upper = sum(c.isupper() for c in q)
    feats: dict[str, float] = {
        "n_chars": float(n_chars),
        "n_tokens": float(n_tokens),
        "n_sentences": float(n_sentences),
        "avg_token_len": float(avg_token_len),
        "n_question_marks": float(q.count("?")),
        "n_exclamations": float(q.count("!")),
        "has_code_fence": 1.0 if "```" in q else 0.0,
        "n_code_spans": float(len(_CODE_SPAN.findall(q))),
        "math_symbol_hits": float(len(_MATH_SYMS.findall(q))),
        "digit_ratio": float(n_digits / n_chars) if n_chars else 0.0,
        "uppercase_ratio": float(n_upper / n_chars) if n_chars else 0.0,
        "readability_proxy": float(avg_token_len * math.log1p(n_tokens)),
    }
    for name, pat in _DOMAIN_PATTERNS.items():
        feats[name] = 1.0 if pat.search(q) else 0.0
    feats["trivial_op_hit"] = 1.0 if _TRIVIAL_OP.search(q) else 0.0
    feats["bracket_data_hit"] = 1.0 if _BRACKET_DATA.search(q) else 0.0
    return feats


def featurize_many(questions: list[str]) -> list[dict[str, float]]:
    return [featurize(q) for q in questions]
