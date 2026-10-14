"""Seed corpus for RouterPFN (MIT): the zero-data onboarding path.

The package ships `data/seed-v1.jsonl` (stratified, slim schema) so a
user with a single question and no labels can fit a router against
OUR corpus in seconds:

    clf = RouterPFNClassifier.from_seed(engine="tabicl", roster=[...])
    clf.route("...")

Seed selection (deterministic): per consensus level, take the N
lowest-spread rows (judge agreement first), ties broken by id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SEED_N_PER_LEVEL = 40
SEED_VERSION = "v1"


def seed_path() -> Path:
    return Path(__file__).resolve().parent / "data" / f"seed-{SEED_VERSION}.jsonl"


def build_seed(rows: list[dict], n_per_level: int = SEED_N_PER_LEVEL,
               seed: int = 0) -> list[dict]:
    """Stratified seed: N lowest-spread rows per consensus level.

    `seed` is accepted for API stability but selection is fully
    deterministic (spread, then id) — no sampling involved.
    """
    _ = seed
    by_level: dict[int, list[dict]] = {d: [] for d in (1, 2, 3, 4, 5)}
    for r in rows:
        by_level[int(r.get("consensus", r.get("difficulty", 3)))].append(r)
    seed_rows: list[dict] = []
    for d in (1, 2, 3, 4, 5):
        cell = sorted(
            by_level[d],
            key=lambda r: (float(r.get("spread", 9.0)), str(r.get("id", ""))),
        )
        for r in cell[:n_per_level]:
            seed_rows.append({
                "question": r["question"],
                "difficulty": int(r.get("consensus", r.get("difficulty", 3))),
                "domain": r.get("domain", ""),
            })
    return seed_rows


def load_seed() -> list[dict]:
    """Load the bundled seed corpus (slim schema)."""
    path = seed_path()
    if not path.exists():
        raise FileNotFoundError(
            f"bundled seed corpus missing at {path}; rebuild with seed.py")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed_questions_labels() -> tuple[list[str], list[int]]:
    rows = load_seed()
    return [r["question"] for r in rows], [int(r["difficulty"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the bundled seed corpus")
    ap.add_argument("--corpus", default="data/raw/teacher-v1.jsonl")
    ap.add_argument("--out", default=str(seed_path()))
    ap.add_argument("--n-per-level", type=int, default=SEED_N_PER_LEVEL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.corpus).read_text().splitlines()
            if line.strip()]
    seed_rows = build_seed(rows, args.n_per_level, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in seed_rows) + "\n")
    from collections import Counter
    print(f"wrote {len(seed_rows)} seed rows to {out} "
          f"{dict(sorted(Counter(r['difficulty'] for r in seed_rows).items()))}")


if __name__ == "__main__":
    main()
