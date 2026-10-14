"""Laya-derived labels for RouterPFN (MIT).

Laya's `score` carries strong rank signal (r≈0.7 vs LLM consensus on
v1) but compressed absolute range (~1.6–3.6, argmax degenerate at 3).
This module maps it onto usable 1–5 labels two ways:

- laya_argmax: argmax of the Laya distribution + 1 (raw, degenerate).
- laya_calibrated: affine fit of laya_expected onto consensus, fit on
  TRAIN ONLY, rounded + clipped to 1–5. Two parameters — the LLM
  labels serve only as a ruler, and test labels never touch the fit.

The same 70/30 in-order split as bench.py is replicated here so the
fit sees exactly the training rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def train_test_split(rows: list[dict], frac: float = 0.7) -> tuple[list[dict], list[dict]]:
    split = int(frac * len(rows))
    return rows[:split], rows[split:]


def fit_laya_calibration(train: list[dict]) -> tuple[float, float]:
    """Least-squares affine map laya_expected -> consensus on train rows."""
    xs = [r["laya_expected"] for r in train if "laya_expected" in r]
    ys = [float(r["consensus"]) for r in train if "laya_expected" in r]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs) or 1e-9
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom
    return a, my - a * mx


def apply_laya_labels(rows: list[dict], a: float, b: float) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        proba = r.get("laya_proba", {})
        if proba:
            r["laya_argmax"] = int(max(proba, key=lambda k: proba[k]))
        if "laya_expected" in r:
            r["laya_calibrated"] = min(5, max(1, int(round(a * r["laya_expected"] + b))))
        out.append(r)
    return out


def agreement(rows: list[dict], field: str, ref: str = "consensus") -> float:
    pairs = [(r[field], r[ref]) for r in rows if field in r and ref in r]
    return sum(1 for x, y in pairs if x == y) / len(pairs) if pairs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill Laya-derived labels")
    ap.add_argument("--inp", default="data/raw/teacher-v1.jsonl")
    ap.add_argument("--out", default="data/parsed/teacher-v1-laya.jsonl")
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.inp).read_text().splitlines()
            if line.strip()]
    train, test = train_test_split(rows)
    a, b = fit_laya_calibration(train)
    labeled = apply_laya_labels(rows, a, b)
    _, test_labeled = train_test_split(labeled)
    print(f"affine: consensus = {a:.3f} * laya_expected + {b:.3f} (fit n={len(train)})")
    print(f"laya_argmax vs consensus (test):      {agreement(test_labeled, 'laya_argmax'):.3f}")
    print(f"laya_calibrated vs consensus (test): {agreement(test_labeled, 'laya_calibrated'):.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in labeled) + "\n")
    print(f"wrote {len(labeled)} rows to {out}")


if __name__ == "__main__":
    main()
