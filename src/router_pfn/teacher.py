"""LLM question generation + difficulty judging for RouterPFN (MIT).

Pipeline (all local, no keys, reproducible):
  1. GENERATE: a handful of Ollama LLMs each write questions per
     (domain × intended-difficulty) cell.
  2. RATE: Laya `score` (typed decision model, not prose) gives a
     calibrated 1–5 distribution, plus one LLM judge gives an
     independent 1–5. Consensus = rounded mean; spread is recorded
     for disagreement analysis (charter Phase 10).
  3. FIT (elsewhere): TabICL learns features → consensus difficulty.

Why Laya as rater: it returns typed, calibrated decisions in one
forward pass (~33ms, no parsing, no hallucinated prose) — the right
tool for "rate 1–5". Caveat (upstream, Sep 2026): ordinal scoring is
Laya's weaker area, so we keep the LLM judge alongside and measure
agreement instead of trusting either alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import requests

from .synth import DOMAINS

# Sub-domains (phase-A expansion). Each maps to one of the charter's 10
# parent domains so charter reporting stays intact while the generator
# gets more specific briefs.
SUB_DOMAINS: dict[str, str] = {
    "code_review": "programming",
    "data_analysis": "analysis",
    "longform_summary": "writing",
    "multilingual": "knowledge",
    "creative": "writing",
    "science": "knowledge",
}
ALL_DOMAINS: list[str] = DOMAINS + list(SUB_DOMAINS)


def parent_of(domain: str) -> str:
    return SUB_DOMAINS.get(domain, domain)


# Per-model prompt suffixes. olmo-3:7b yielded ~0 keepers in the first
# pilot (chatty/meta output) — it gets the strict variant.
MODEL_SUFFIX: dict[str, str] = {
    "olmo-3:7b": " Respond with ONLY the questions, one per line. No other text of any kind.",
}

OLLAMA_URL = "http://localhost:11434"

DIFFICULTY_DESC = {
    1: "trivial: single factual answer, no reasoning needed",
    2: "easy: short explanation or summary of one idea",
    3: "moderate: multi-part, needs worked steps or comparison",
    4: "hard: debugging, design, or proof needing careful reasoning",
    5: "expert: open-ended, ambiguous, or proof-level with edge cases",
}

LAYA_CRITERIA = [f"Level {d}: {DIFFICULTY_DESC[d]}" for d in range(1, 6)]

GENERATE_PROMPT = """\
Write {k} diverse, realistic questions a user might ask an AI assistant.
Domain: {domain}. Target difficulty {d}/5 ({desc}).

Rules: one question per line, no numbering or bullets, no extra text, \
each question 1-4 sentences. Vary the topics; avoid repeating yourself. \
Begin immediately with the first question — no preamble, no meta-commentary \
about the task, no mention of difficulty levels."""

JUDGE_PROMPT = """\
Rate how difficult this user question is for a language model to answer well, 1-5:
1 = trivial factual, single-step answer
2 = easy, needs a short explanation or summary
3 = moderate, multi-part or needs worked steps
4 = hard, debugging/design/proof needing careful reasoning
5 = expert, open-ended/ambiguous/proof-level

Question: {question}

Return ONLY this JSON, nothing else: {{"difficulty": N}}"""


def ollama_chat(model: str, content: str, *, num_predict: int = 400,
                temperature: float = 0.8, timeout: int = 300) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return (((r.json().get("message") or {}).get("content")) or "").strip()


_META_PATTERNS = re.compile(
    r"difficulty|generate.{0,20}question|as an ai|here are|i (should|need|will)|"
    r"target difficulty|/5 \(|realistic questions",
    re.I,
)


def _clean_lines(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^[\s>]*(\d+[.)\-:]|[-*•])\s+", "", line).strip().strip('"')
        if len(line.split()) >= 4 and not _META_PATTERNS.search(line):
            out.append(line)
    return out


def generate_questions(model: str, domain: str, difficulty: int, k: int,
                       seed: int) -> list[str]:
    prompt = GENERATE_PROMPT.format(
        k=k, domain=domain, d=difficulty, desc=DIFFICULTY_DESC[difficulty])
    prompt += MODEL_SUFFIX.get(model, "")
    rng = random.Random(f"{seed}|{model}|{domain}|{difficulty}")
    # Mild prompt shuffle so the handful don't all write the same question.
    opener = rng.choice(["", "Vary difficulty cues naturally. ", "Prefer concrete scenarios. "])
    return _clean_lines(ollama_chat(model, opener + prompt))


def llm_judge(model: str, question: str) -> int:
    text = ollama_chat(model, JUDGE_PROMPT.format(question=question),
                       num_predict=60, temperature=0.1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in judge reply: {text[:200]!r}")
    d = int(json.loads(m.group(0)).get("difficulty", 0))
    if d not in (1, 2, 3, 4, 5):
        raise ValueError(f"judge returned out-of-range difficulty: {text[:200]!r}")
    return d


_LAYA_ROUTER = None


def laya_rate(question: str) -> dict:
    """Rate via Laya typed `score` (lazy import; single forward pass).

    The Router is cached module-wide so the checkpoint loads once, not
    per question.
    """
    global _LAYA_ROUTER
    if _LAYA_ROUTER is None:
        from laya import Router

        # CPU: ollama holds the GPU during corpus builds; ModernBERT-large
        # rates in ms on CPU and this avoids the GPU-OOM retry dance.
        _LAYA_ROUTER = Router(device="cpu")
    res = _LAYA_ROUTER.predict(
        question,
        {"difficulty": {
            "type": "score",
            "instructions": "How difficult is this user question for a language model?",
            "criteria": LAYA_CRITERIA,
        }},
    )
    ans = res["answers"]["difficulty"]
    # Laya score levels are 0-indexed ("0".."4"); map onto difficulty 1–5.
    proba = {str(int(k) + 1): float(v) for k, v in ans["probabilities"].items()}
    expected = sum(int(d) * p for d, p in proba.items())
    return {"expected": float(expected), "proba": proba}


def _norm(text: str) -> str:
    """Normalized form for exact-dup detection."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _near_dup(candidate: str, seen_tokens: list[set[str]], thresh: float = 0.85) -> bool:
    """Token-set Jaccard near-dup check against kept questions."""
    cand = set(_norm(candidate).split())
    if not cand:
        return True
    for toks in seen_tokens:
        if not toks:
            continue
        inter = len(cand & toks)
        union = len(cand | toks)
        if union and inter / union >= thresh:
            return True
    return False


def rate_question(q: str, domain: str, intended: int, gen: str,
                  judges: list[str], seed: int, use_laya: bool = True) -> dict:
    """Run Laya + LLM judges on one question; return the consensus row."""
    row: dict = {
        "id": "t-" + hashlib.md5(f"{seed}|{gen}|{q}".encode()).hexdigest()[:8],
        "question": q,
        "domain": domain,
        "parent": parent_of(domain),
        "intended_difficulty": intended,
        "generator": gen,
    }
    if use_laya:
        try:
            lr = laya_rate(q)
            row["laya_expected"] = lr["expected"]
            row["laya_proba"] = lr["proba"]
        except Exception as exc:
            print(f"  ! laya rate: {exc}")
            row["laya_error"] = str(exc)[:200]
    llm_ratings: dict[str, int] = {}
    for judge in judges:
        try:
            llm_ratings[judge] = llm_judge(judge, q)
        except Exception as exc:
            print(f"  ! judge {judge}: {exc}")
    row["llm_ratings"] = llm_ratings
    # Consensus: median of LLM judges is primary. Laya's expected score
    # compresses to ~2-3 on the micro-pilot (upstream-known ordinal
    # weakness), so it is recorded as an auxiliary signal, not averaged in.
    if llm_ratings:
        vals = sorted(llm_ratings.values())
        consensus = float(vals[len(vals) // 2])
    else:
        consensus = float(intended)
    row["consensus"] = min(5, max(1, int(round(consensus))))
    all_signals = list(map(float, llm_ratings.values()))
    if "laya_expected" in row:
        all_signals.append(row["laya_expected"])
    row["spread"] = round(
        max(all_signals) - min(all_signals), 3) if all_signals else 0.0
    return row


def build_corpus(generators: list[str], judges: list[str], k_per_cell: int,
                 seed: int, use_laya: bool = True,
                 domains: list[str] | None = None,
                 seen: set[str] | None = None,
                 seen_tokens: list[set[str]] | None = None,
                 on_row=None) -> list[dict]:
    rows: list[dict] = []
    seen = seen if seen is not None else set()
    seen_tokens = seen_tokens if seen_tokens is not None else []
    for gen in generators:
        for domain in (domains or ALL_DOMAINS):
            for difficulty in (1, 2, 3, 4, 5):
                try:
                    questions = generate_questions(gen, domain, difficulty, k_per_cell, seed)
                except Exception as exc:
                    print(f"  ! generate {gen}/{domain}/{difficulty}: {exc}")
                    continue
                for q in questions[:k_per_cell]:
                    norm = _norm(q)
                    if norm in seen or _near_dup(q, seen_tokens):
                        print(f"  - dup dropped ({domain}/{difficulty}/{gen}): {q[:60]}")
                        continue
                    seen.add(norm)
                    seen_tokens.append(set(norm.split()))
                    row = rate_question(q, domain, difficulty, gen, judges, seed, use_laya)
                    rows.append(row)
                    if on_row is not None:
                        on_row(row)
                    print(f"  + [{row['consensus']}] ({domain}/{difficulty}/{gen}) {q[:80]}")
    return rows


def consensus_counts(rows: list[dict]) -> dict[int, int]:
    counts = {d: 0 for d in (1, 2, 3, 4, 5)}
    for r in rows:
        counts[int(r.get("consensus", 3))] += 1
    return counts


def top_up(rows: list[dict], generators: list[str], judges: list[str],
           target_per_level: int, seed: int, use_laya: bool,
           seen: set[str], seen_tokens: list, on_row=None,
           max_cells: int = 60) -> list[dict]:
    """Fill consensus-poor levels: generate cells aimed at the poorest level
    until every level has >= target rows. Generation aims one step BELOW
    the poor level (judges systematically rate up: intended-1 lands 1-2,
    intended-2 lands 2-4), except L5 which aims at 5 directly."""
    AIM_BELOW = {1: 1, 2: 1, 3: 2, 4: 3, 5: 5}
    rng = random.Random(f"topup|{seed}")
    for i in range(max_cells):
        counts = consensus_counts(rows)
        poor = [d for d in (1, 2, 3, 4, 5) if counts[d] < target_per_level]
        if not poor:
            print(f"top-up complete: {counts}")
            break
        level = min(poor, key=lambda d: counts[d])
        aim = AIM_BELOW[level]
        gen = rng.choice(generators)
        domain = rng.choice(ALL_DOMAINS)
        try:
            questions = generate_questions(gen, domain, aim, 2, seed + 1000 + i)
        except Exception as exc:
            print(f"  ! topup generate {gen}/{domain}/{aim}->{level}: {exc}")
            continue
        for q in questions[:2]:
            norm = _norm(q)
            if norm in seen or _near_dup(q, seen_tokens):
                print(f"  - dup dropped (topup/{domain}/{aim}->{level}/{gen}): {q[:60]}")
                continue
            seen.add(norm)
            seen_tokens.append(set(norm.split()))
            row = rate_question(q, domain, aim, gen, judges, seed + 1000 + i, use_laya)
            rows.append(row)
            if on_row is not None:
                on_row(row)
            print(f"  + [{row['consensus']}] (topup/{domain}/{aim}->{level}/{gen}) {q[:80]}")
    return rows


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-generate + Laya/LLM-judge corpus")
    ap.add_argument("--generators", default="ornith:9b,olmo-3:7b,gemma3:4b")
    ap.add_argument("--judges", default="gemma3:4b,ornith:9b")
    ap.add_argument("--no-laya", action="store_true")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domains", default="",
                    help="comma subset of domains (default: all charter + sub-domains)")
    ap.add_argument("--resume", action="store_true",
                    help="keep existing --out rows and skip their questions")
    ap.add_argument("--target-per-level", type=int, default=0,
                    help="top-up loop quota per consensus level (0 = off)")
    ap.add_argument("--topup-cells", type=int, default=60)
    ap.add_argument("--topup-only", action="store_true",
                    help="skip the broad pass; only top-up (requires --resume)")
    ap.add_argument("--out", default="data/raw/teacher-v1.jsonl")
    args = ap.parse_args()
    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None
    use_laya = not args.no_laya

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(out) if args.resume else []
    seen = {_norm(r["question"]) for r in rows}
    seen_tokens = [set(_norm(r["question"]).split()) for r in rows]
    if rows:
        print(f"resumed with {len(rows)} existing rows")

    with open(out, "a" if args.resume else "w") as f:
        def on_row(row: dict) -> None:
            f.write(json.dumps(row) + "\n")
            f.flush()

        rows = rows + ([] if args.topup_only else build_corpus(
            gens, judges, args.k, args.seed, use_laya,
            domains=domains, seen=seen,
            seen_tokens=seen_tokens, on_row=on_row))
        if args.target_per_level > 0:
            rows = top_up(rows, gens, judges, args.target_per_level,
                          args.seed, use_laya, seen, seen_tokens,
                          on_row=on_row, max_cells=args.topup_cells)
    counts = consensus_counts(rows)
    agreed = sum(1 for r in rows if r.get("spread", 9) <= 1.0)
    print(f"total {len(rows)} rows in {out} ({agreed} spread<=1.0) levels={counts}")


if __name__ == "__main__":
    main()
