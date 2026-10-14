"""Synthetic question-corpus generator for RouterPFN (MIT).

v0.1 posture: template-based, license-clean by construction, covering
the charter's 10 domains with a spread of intended difficulties 1–5.
Optional `--teacher` paraphrase / re-judge via a local Ollama model is
a new expression of the template idea (not a copy of third-party
text); offline runs work with zero network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DOMAINS = [
    "knowledge",
    "reasoning",
    "mathematics",
    "programming",
    "debugging",
    "ml_data_science",
    "writing",
    "analysis",
    "multi_step",
    "ambiguous",
]

_TEMPLATES: dict[int, list[tuple[str, str]]] = {
    1: [
        ("knowledge", "What is the capital of {place}?"),
        ("writing", "Write a one-sentence greeting for {topic}."),
        ("knowledge", "Who wrote {work}?"),
    ],
    2: [
        ("reasoning", "If {a} then {b}; given {a}, what follows in one step?"),
        ("writing", "Summarize this paragraph in two sentences: {topic}."),
        ("knowledge", "Explain {topic} to a beginner in under 100 words."),
    ],
    3: [
        ("mathematics", "Solve for x: {eq}. Show your steps."),
        ("programming", "Write a Python function that {task}. Include a docstring."),
        ("analysis", "Compare {a} and {b}: list two tradeoffs each."),
    ],
    4: [
        ("debugging", "This code fails with {error}: `{code}`. Diagnose and fix it."),
        ("ml_data_science", "Design an evaluation for {task} with metric justification."),
        ("multi_step", "Plan then execute: {task} in three ordered steps with checks."),
    ],
    5: [
        ("reasoning", "Prove or refute: {claim}. Consider edge cases and counterexamples."),
        ("mathematics", "Prove {claim} and analyze the boundary conditions."),
        ("ambiguous", "The request is underspecified: {topic}. Resolve ambiguity, then solve."),
    ],
}

_FILLERS = {
    "place": ["France", "Japan", "Brazil", "Egypt"],
    "topic": ["photosynthesis", "the steam engine", "a product launch", "tides"],
    "work": ["Hamlet", "the Odyssey", "Moby Dick"],
    "a": ["it rains", "the cache is warm", "the tests pass"],
    "b": ["the ground is wet", "latency drops", "we can ship"],
    "eq": ["2x + 3 = 11", "x^2 - 5x + 6 = 0"],
    "task": ["deduplicates a CSV by email", "retries flaky HTTP calls", "parses log timestamps"],
    "error": [
        "ValueError: invalid literal",
        "KeyError: 'user_id'",
        "IndexError: list index out of range",
    ],
    "code": ["int(x['a'])", "rows[10]['id']", "parse_date(s)"],
    "claim": ["every even number > 2 is the sum of two primes", "this sort runs in O(n)"],
}


def _fill(template: str, rng: random.Random) -> str:
    out = template
    for key, options in _FILLERS.items():
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", rng.choice(options))
    return out


def generate(n: int = 500, seed: int = 0) -> list[dict]:
    """Generate n synthetic rows with balanced difficulty + domain spread."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(n):
        difficulty = (i % 5) + 1  # round-robin keeps the 1–5 balance exact
        domain, template = rng.choice(_TEMPLATES[difficulty])
        question = _fill(template, rng)
        digest = hashlib.md5(f"{seed}|{i}|{question}".encode()).hexdigest()[:8]
        rows.append(
            {
                "id": f"synth-v0-{digest}",
                "question": question,
                "domain": domain,
                "difficulty": difficulty,
                "judge": "template",
            }
        )
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate RouterPFN synthetic corpus")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/raw/synth-v0.jsonl")
    ap.add_argument("--teacher", type=str, default="",
                    help="optional local Ollama model for paraphrase/re-judge (offline when empty)")
    args = ap.parse_args()
    rows = generate(n=args.n, seed=args.seed)
    if args.teacher:
        print(f"note: --teacher {args.teacher} paraphrase is a v0.2 hook (templates kept as-is)")
    write_jsonl(rows, Path(args.out))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
