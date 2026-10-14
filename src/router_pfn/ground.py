"""Empirical grounding for RouterPFN (MIT) — charter Phase 4.

Replaces judge opinions with measurements: run WEAK vs STRONG models
on each question, grade both answers (LLM grader + Laya noul), and
relabel by weakest-model-that-succeeds. Also captures real
tokens/latency per answer (charter Phase 3 metrics).

Execution is STAGED by model to avoid VRAM swaps on the 8GB card:
  A. all weak answers   (weak resident)
  B. all strong answers (strong resident)
  C. all grades         (grader resident; Laya is CPU)
Each stage writes incrementally with --resume support.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434"

GRADE_PROMPT = """\
Question: {question}

Proposed answer: {answer}

Does the proposed answer correctly solve the question? A correct answer
addresses what was asked with no critical errors; minor omissions are OK,
wrong facts or missing the point are not.

Return ONLY this JSON, nothing else: {{"success": true/false, "quality": 0.0-1.0}}
(quality: 1.0 perfect, 0.7 minor gaps, 0.4 major gaps but on-topic, 0.0 wrong)
"""

_LAYA_ROUTER = None


def ollama_answer(model: str, question: str, num_predict: int,
                  temperature: float = 0.3) -> dict:
    t0 = time.perf_counter()
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        },
        timeout=900,
    )
    r.raise_for_status()
    body = r.json()
    text = (((body.get("message") or {}).get("content")) or "").strip()
    return {
        "text": text,
        "tokens": int(body.get("eval_count") or 0),
        "prompt_tokens": int(body.get("prompt_eval_count") or 0),
        "seconds": round(time.perf_counter() - t0, 1),
        "truncated": body.get("done_reason") == "length",
    }


def grade_llm(grader: str, question: str, answer: str) -> dict:
    text = ollama_answer(grader, GRADE_PROMPT.format(question=question, answer=answer[:3000]),
                         num_predict=120, temperature=0.1)["text"]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in grade reply: {text[:200]!r}")
    d = json.loads(m.group(0))
    return {"success": bool(d.get("success", False)),
            "quality": min(1.0, max(0.0, float(d.get("quality", 0.0))))}


def grade_laya(question: str, answer: str) -> float:
    """Laya noul P(answer solves question). CPU; router cached."""
    global _LAYA_ROUTER
    if _LAYA_ROUTER is None:
        from laya import Router

        _LAYA_ROUTER = Router(device="cpu")
    state = f"Question: {question}\nAnswer: {answer[:2000]}"
    res = _LAYA_ROUTER.predict(
        state,
        {"solved": {"type": "noul",
                    "instructions": "The answer correctly and completely solves the question."}},
    )
    return float(res["answers"]["solved"]["noul"])


def weakest_tier(weak_ok: bool, strong_ok: bool) -> str:
    """Weakest model that succeeds. both_failed → strong (safe default)."""
    if weak_ok:
        return "weak"
    return "strong"


def pick_subset(rows: list[dict], n_per_level: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_level: dict[int, list[dict]] = {d: [] for d in (1, 2, 3, 4, 5)}
    for r in rows:
        by_level[int(r.get("consensus", 3))].append(r)
    subset = []
    for d in (1, 2, 3, 4, 5):
        pool = by_level[d][:]
        rng.shuffle(pool)
        subset.extend(pool[:n_per_level])
    rng.shuffle(subset)
    return subset


def load_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())}


def save_records(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records.values():
            f.write(json.dumps(r) + "\n")


def run_stage(records: dict[str, dict], stage: str, weak: str, strong: str,
              grader: str, weak_np: int, strong_np: int, use_laya: bool,
              path: Path) -> dict[str, dict]:
    for rid, rec in records.items():
        try:
            if stage == "answers_weak" and "weak" not in rec:
                rec["weak"] = ollama_answer(weak, rec["question"], weak_np)
                print(f"  A weak {rid} ({rec['weak']['tokens']} tok, {rec['weak']['seconds']}s)")
            elif stage == "answers_strong" and "strong" not in rec:
                rec["strong"] = ollama_answer(strong, rec["question"], strong_np)
                s = rec["strong"]
                print(f"  B strong {rid} ({s['tokens']} tok, {s['seconds']}s)")
            elif stage == "grades":
                for side in ("weak", "strong"):
                    slot = rec.get(side, {})
                    if side not in rec or not slot.get("text"):
                        continue
                    if "grade" not in slot:
                        slot["grade"] = grade_llm(grader, rec["question"], slot["text"])
                    if use_laya and "laya" not in slot:
                        slot["laya"] = round(grade_laya(rec["question"], slot["text"]), 4)
                    g = slot["grade"]
                    print(f"  C {side} {rid}: ok={g['success']} "
                          f"q={g['quality']} laya={slot.get('laya')}")
        except Exception as exc:
            print(f"  ! {stage} {rid}: {str(exc)[:150]}")
        save_records(path, records)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Empirical grounding pilot")
    ap.add_argument("--corpus", default="data/raw/teacher-v1.jsonl")
    ap.add_argument("--n-per-level", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weak", default="gemma3:4b")
    ap.add_argument("--strong", default="ornith:9b")
    ap.add_argument("--grader", default="ornith:9b")
    ap.add_argument("--weak-np", type=int, default=300)
    ap.add_argument("--strong-np", type=int, default=512)
    ap.add_argument("--no-laya", action="store_true")
    ap.add_argument("--stages", default="answers_weak,answers_strong,grades")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="data/raw/ground-v0.jsonl")
    args = ap.parse_args()

    corpus = [json.loads(line) for line in Path(args.corpus).read_text().splitlines()
              if line.strip()]
    subset = pick_subset(corpus, args.n_per_level, args.seed)
    print(f"subset: {len(subset)} questions")
    out = Path(args.out)
    records = load_records(out) if args.resume else {}
    for r in subset:
        records.setdefault(r["id"], {"id": r["id"], "question": r["question"],
                                     "domain": r["domain"], "consensus": r["consensus"]})
    save_records(out, records)
    for stage in [s.strip() for s in args.stages.split(",") if s.strip()]:
        print(f"== stage {stage} ==")
        records = run_stage(records, stage, args.weak, args.strong, args.grader,
                            args.weak_np, args.strong_np, not args.no_laya, out)
    n_weak = sum(1 for r in records.values() if "weak" in r)
    n_strong = sum(1 for r in records.values() if "strong" in r)
    n_graded = sum(1 for r in records.values()
                   if r.get("weak", {}).get("grade") and r.get("strong", {}).get("grade"))
    print(f"done: weak={n_weak} strong={n_strong} graded={n_graded} / {len(records)}")


if __name__ == "__main__":
    main()
