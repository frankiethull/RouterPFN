"""Engine shootout for RouterPFN (MIT).

Compares backends on IDENTICAL splits: accuracy + fit latency +
first vs repeated predict latency (the kv_cache question).
Kumo is attempted and its blockage recorded, never silently skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .router import RouterPFNClassifier

CONFIGS: list[dict] = [
    {"name": "dummy", "engine": "dummy"},
    {"name": "tabicl-kv", "engine": "tabicl", "device": "cpu", "kv_cache": True},
    {"name": "tabicl-nokv", "engine": "tabicl", "device": "cpu", "kv_cache": False},
    {"name": "fiorino", "engine": "fiorino", "device": "cpu"},
    {"name": "kumo", "engine": "kumo", "device": "cpu"},
]


def run_config(questions, ytrain, test_q, ytest, roster, cfg: dict) -> dict:
    name = cfg["name"]
    out: dict = {"name": name}
    try:
        kwargs = {k: v for k, v in cfg.items() if k != "name"}
        clf = RouterPFNClassifier(roster=roster, **kwargs).fit(questions, ytrain)
    except Exception as exc:
        out["status"] = f"blocked: {type(exc).__name__}: {str(exc)[:200]}"
        return out
    t0 = time.perf_counter()
    clf = RouterPFNClassifier(roster=roster, **kwargs).fit(questions, ytrain)
    out["fit_s"] = round(time.perf_counter() - t0, 2)
    try:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            P = clf.predict_proba(test_q)
            times.append(time.perf_counter() - t0)
        pred = np.array([1, 2, 3, 4, 5])[np.asarray(P).argmax(axis=1)]
        out.update({
            "status": "ok",
            "predict_s_first": round(times[0], 2),
            "predict_s_2nd": round(times[1], 2),
            "predict_s_3rd": round(times[2], 2),
            "accuracy": round(float((pred == np.asarray(ytest)).mean()), 4),
            "simplex_ok": bool(np.allclose(np.asarray(P).sum(axis=1), 1.0, atol=1e-6)),
        })
    except Exception as exc:
        out["status"] = f"predict-blocked: {type(exc).__name__}: {str(exc)[:200]}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="RouterPFN engine shootout")
    ap.add_argument("--data", default="data/raw/teacher-v1.jsonl")
    ap.add_argument("--label", default="consensus")
    ap.add_argument("--roster", default="weak-local,strong-frontier")
    ap.add_argument("--out", default="reports/shootout.json")
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()
            if line.strip()]
    questions_all = [r["question"] for r in rows]
    labels_all = [int(r.get(args.label, r.get("difficulty", 3))) for r in rows]
    split = int(0.7 * len(rows))
    questions, ytrain = questions_all[:split], labels_all[:split]
    test_q, ytest = questions_all[split:], labels_all[split:]
    roster = [m.strip() for m in args.roster.split(",") if m.strip()]
    report = {"n_train": len(questions), "n_test": len(test_q), "results": []}
    for cfg in CONFIGS:
        print(f"== {cfg['name']} ==", flush=True)
        res = run_config(questions, ytrain, test_q, ytest, roster, cfg)
        print(f"   {res}", flush=True)
        report["results"].append(res)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
