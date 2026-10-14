"""Benchmark runner: baselines vs RouterPFN (MIT).

Baselines (charter Phase 8): always-weak, always-strong, random,
rule-based (length heuristic). The router must beat useful baselines
before additional complexity lands.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .eval import cost_quality_summary, routing_accuracy, tier_report
from .features import featurize
from .router import RouterPFNClassifier
from .synth import generate


def rule_based_difficulty(question: str) -> int:
    """Length/structure heuristic: long+code/symbol-heavy reads harder."""
    f = featurize(question)
    score = f["n_tokens"] / 50.0 + f["has_code_fence"] * 1.5 + f["math_symbol_hits"] / 5.0
    if score < 0.6:
        return 1
    if score < 1.2:
        return 2
    if score < 2.0:
        return 3
    if score < 3.0:
        return 4
    return 5


def main() -> None:
    ap = argparse.ArgumentParser(description="RouterPFN benchmark (synthetic pilot)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", type=str, default="dummy")
    ap.add_argument("--roster", type=str, default="weak-local,strong-frontier")
    ap.add_argument("--data", type=str, default="",
                    help="teacher/synth JSONL with {question, consensus|difficulty}")
    ap.add_argument("--label", type=str, default="consensus",
                    help="label field when --data is given")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="reports/bench-v0.json")
    args = ap.parse_args()

    roster = [m.strip() for m in args.roster.split(",") if m.strip()]
    if args.data:
        rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()
                if line.strip()]
        questions_all = [r["question"] for r in rows]
        labels_all = [int(r.get(args.label, r.get("difficulty", 3))) for r in rows]
        split = int(0.7 * len(rows))
        questions, ytrain = questions_all[:split], labels_all[:split]
        test_q, ytest = questions_all[split:], labels_all[split:]
    else:
        rows = generate(n=args.n, seed=args.seed)
        split = int(0.7 * len(rows))
        train, test = rows[:split], rows[split:]
        questions = [r["question"] for r in train]
        ytrain = [r["difficulty"] for r in train]
        test_q = [r["question"] for r in test]
        ytest = [r["difficulty"] for r in test]

    clf = RouterPFNClassifier(engine=args.engine, roster=roster,
                              device=args.device).fit(questions, ytrain)
    pred = clf.predict(test_q)
    rule = [rule_based_difficulty(q) for q in test_q]
    rng = random.Random(args.seed)
    rnd = [rng.randint(1, 5) for _ in test_q]

    routes = clf.route_many(test_q)
    costs = {m: float(i + 1) for i, m in enumerate(roster)}
    quality = {m: 0.6 + 0.2 * i for i, m in enumerate(roster)}
    report = {
        "engine": args.engine,
        "roster": roster,
        "n_train": len(questions),
        "n_test": len(test_q),
        "router_difficulty_acc": routing_accuracy(ytest, pred),
        "rule_difficulty_acc": routing_accuracy(ytest, rule),
        "random_difficulty_acc": routing_accuracy(ytest, rnd),
        "tiers": tier_report(routes, roster),
        "cost_quality": cost_quality_summary(routes, roster, costs, quality),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
