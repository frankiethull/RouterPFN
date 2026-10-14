# RouterPFN — Materials

License-first ledger. v0.1 trains on **synthetic data only**
(templates + optional local teacher), so there is no third-party
dataset to license. This file records engine deps + data provenance.

## Engine dependency ledger (verify license at pin time, do not trust notes)

| Engine | Pip package | License (recheck before pinning) | Role |
|---|---|---|---|
| TabPFN | `tabpfn` | recheck (Prior Labs) | primary v0.1 |
| TabICL | `tabicl` | recheck | adapter |
| Kumo-Tabular | `kumo-tabular` (or per-docs name) | recheck | adapter |
| Fiorino | `fiorino` from `github.com/frankiethull/fiorino` (`python/` subdir, not on PyPI yet) | MIT | 4th engine |
| dummy | none | — | tests / offline |

Pin checklist (before any benchmark claim):
- [x] Installed 2026-10-11 via `uv pip install`: `tabicl==2.2.0`
      (BSD-3-Clause, Soda/Inria — permissive, gate passes),
      `laya==0.4.2` (Apache-2.0, Convai Innovations — gate passes).
      Checkpoints auto-download from HF on first use:
      TabICL `tabicl-classifier-v2-20260212.ckpt` (default),
      Laya `convaiinnovations/laya` (auto-routed `english`).
- [ ] `predict_proba` returns a valid `(n, 5)` simplex on the synth pilot.

## Synthetic corpus v0 (license-clean by construction)

- Generator: `synth.py` templates across the charter's 10 domains
  (knowledge, reasoning, math, programming, debugging, ML/DS, writing,
  analysis, multi-step, ambiguous). Optional local-teacher paraphrase /
  1–5 judging behind `--teacher`; teacher output is a new expression of
  a template idea, not a copy of third-party text.
- Labels: `difficulty` in {1..5} (LLM-judged synth posture for v0.1;
  empirical weakest-model-that-succeeds becomes v0.2 ground truth).
- Location: `data/{raw,parsed,splits}/` (gitignored
  raw, committed splits manifest only). No redistribution of external sets.
- No NC source used anywhere (train / calib / eval).

## Teacher pilot v0 (2026-10-11, running)

- Generators (handful): `ornith:9b` + `olmo-3:7b` (both fit the 8GB
  card). `gemma4:latest` (9.6GB) EXCLUDED from the pilot: heavy CPU
  offload made generation ~10min/cell with zero output in 7min —
  revisit on a bigger card or with a smaller quant.
- Judges: `gemma3:4b` + `ornith:9b` (overlap with generator role is
  recorded per row; self-rating ablation later). Laya `score`
  (CPU-forced, checkpoint `convaiinnovations/laya`) as auxiliary
  signal — NOT averaged into consensus.
- Finding (micro-pilot, n=19): Laya expected-score compresses to
  ~2.0–3.2 while the LLM judge spans 1–5 and tracks intended
  difficulty. Matches the upstream Sep-2026 caveat on ordinal scoring.
  Consensus = median of LLM judges; Laya expected + full spread kept
  for disagreement analysis. Laya also warns its temperatures are
  uncalibrated — confidence values are not trusted in v0.1.
- Prompt hygiene: generators needed "no preamble" instruction + a
  meta-response filter (first micro run leaked "I need to generate…"
  rows); filter drops lines matching difficulty/meta talk.

## Teacher corpus v1 — final (2026-10-11)

- **732 rows**, `data/raw/teacher-v1.jsonl`.
  Consensus levels {1:87, 2:60, 3:264, 4:260, 5:61} — all quotas met
  via aim-below top-up (targets 60/level, 150-cell cap, exited early
  on completion). Generators {ornith:243+topups, gemma3:4b:233+topups,
  olmo:10}. Dual LLM judges on ~90% of rows; Laya on 100%.
- Level-2 finding: judges systematically rate up (intended-2 lands
  3-4), so L2 was filled by aiming intended-1 (`AIM_BELOW` map).
  Zero-cell parents from the broad pass (debugging/L1,
  ml_data_science/L1+L5, multi_step/L2) were covered in top-up.
- Bench (TabICL CPU, kv_cache, 512/220): difficulty acc **0.609**
  vs rule 0.205 vs random 0.191. Routing 178 weak / 42 strong,
  cost −40% vs always-strong. Quality delta still under the assumed
  0.6/0.8 mapping — empirical grounding (Phase 4) remains open.
  Acc is lower than v0's 0.667 on 24 test rows; v1's test is 9×
  larger, 3 generators, 16 domains — harder task, honest number.

## Laya-labels study (2026-10-11)

- Laya signal on 732 rows: argmax degenerate (73% land on 3, zero
  5s), but expected-value is monotonic in consensus (2.27 → 2.39 →
  2.69 → 2.99 → 3.04) with r=0.705. Calibration affine (train-only):
  consensus = 2.093 × expected − 2.477 → `laya_calibrated` in
  `data/parsed/teacher-v1-laya.jsonl` (`labels.py`).
- Scoreboard, test n=220 (tier split 176 weak / 44 strong):

  | setup | exact acc vs consensus | tier agree vs consensus |
  |---|---|---|
  | TabICL on consensus | 0.609 | 0.882 |
  | TabICL on laya-cal | 0.486 | 0.868 |
  | Laya-direct calibrated | 0.427 | 0.827 |
  | Laya-direct argmax | 0.459 | 0.782 |
  | always-weak | — | 0.800 |

- Read: the TFM trained on Laya labels routes MORE like the LLM
  consensus (0.868) than Laya itself does (0.827) — the tabular
  router generalizes the decision model's signal through question
  features instead of echoing its compression. Both trail the
  consensus-trained router (0.882), as expected.
- Caveat: 44 strong test rows makes ±0.03 noise on tier numbers;
  quality side still assumed, not empirical.

## Empirical grounding pilot (2026-10-11) — charter Phase 4

- `ground.py`: 150-question stratified subset (30/level), weak
  `gemma3:4b` vs strong `ornith:9b`, staged by model (A weak, B
  strong, C grades), grader `ornith:9b` + Laya noul (CPU), resume +
  incremental. 128/150 both-graded (`data/raw/ground-v0.jsonl`).
- Measured success (grader), by consensus level —
  L1 weak 100%, L2 76%, L3 48% (strong 86%), L4 21% (strong 92%),
  L5 8% (strong 54%; 46% of L5 stump strong too). Opinion labels
  validate against measurements.
- Caveats: weak hit the 300-token cap in 63% of answers (some fails
  are verbosity, not inability — recorded via `truncated`);
  Laya-noul range 0.72–0.89 on ALL real answers (fails score 0.79+),
  so success = LLM grader only, Laya kept as aux. Laya-grader
  agreement 0.54 ≈ coin flip as a gate.
- DECISION-GRADE RESULT (measured success + measured seconds, n=128):
  | policy | success | time/q |
  |---|---|---|
  | always-weak (local) | 0.539 | 3.7s |
  | always-strong | 0.875 | 6.7s |
  | router-followed | 0.766 | 6.4s |

  Tier agreement router vs MEASURED tiers: **0.820** (105/128).
  Confusion: 17 weak-routed-needing-strong (failures), 6
  strong-routed-weak-could-do (waste).
- Cost reading: latency gap is only 1.8×, so wall-clock savings are
  thin (4%) — but weak-local tokens are ~free, so the dollar bill
  falls ~60% (37.5% strong fraction) while retaining 87.5% of
  frontier success. The router is a money saver, not a time saver.

## Engine shootout (2026-10-11, `shootout.py`, CPU, 512/220)
| engine | acc | fit | predict 1st/2nd/3rd |
|---|---|---|---|
| dummy | 0.136 | 0.06s | 0.02s |
| tabicl-kv | 0.609 | 4.21s | 1.58 / 2.91 / 1.74s |
| tabicl-nokv | 0.609 | 0.61s | 2.94 / 3.50 / 3.40s |
| fiorino (ours, CPU) | 0.614 | 0.18s | 0.74 / 0.91 / 0.58s |
| kumo | BLOCKED | — | — |

- Fiorino matches TabICLv2 on accuracy (0.614 vs 0.609) at ~3× the
  predict speed — the nano model holds its own here. (fit_s gap is
  first-touch checkpoint download, not architecture.)
- kv_cache shows no measurable win on CPU at this scale (noisy
  repeats); the kv story needs a GPU re-run. Defaults stay ON —
  harmless on CPU, helps on GPU.
- Kumo blocked: `structured-data-models` is an empty 0.0.0a0
  placeholder with no classifier API; weights OpenMDW-1.1 (gate).
  `engines.py` raises an explicit error; re-probe on release.

- Generators: `olmo-3:7b` (strict suffix fixed its yield), `ornith:9b`,
  `gemma3:4b` (judge overlap recorded per row). k=2, seed 1.
- Domains: charter 10 + 6 sub-domains (`code_review→programming`,
  `data_analysis→analysis`, `longform_summary→writing`,
  `multilingual→knowledge`, `creative→writing`, `science→knowledge`),
  each row carrying both `domain` and charter `parent`.
- Scale guards new in this push: exact+near-dup (Jaccard 0.85) filter,
  `--resume` with incremental flushed writes, `python -u` streaming
  logs (v0's buffered log hid completion for an hour).
- Stratification: `--target-per-level 40 --topup-cells 60` fills
  consensus-poor levels after the broad pass (quotas on consensus,
  never on intended difficulty).
- Phase-A final config (after two restarts): generators
  `ornith:9b,gemma3:4b`, judges `gemma3:4b,ornith:9b`, k=3, seed 1,
  `--resume` over the 10 olmo rows. Rationale: peak residency
  gemma3:4b+olmo ≈ 7.8GB fits the 8GB card; the gemma3:4b+ornith judge
  pairing thrashed at ~8.9GB (≈84s/row). gemma4 excluded (VRAM
  thrash, revisited later). First attempt (olmo gen + dual judges
  incl. ornith) was killed for the same reason.
- Judge finding: `olmo-3:7b` returns EMPTY replies to the rating
  prompt on both `/api/chat` and `/api/generate` (yet generates
  questions fine) — dropped as judge, kept as generator. Rows rated
  during the olmo-judge window carry gemma3:4b+Laya only; backfill
  with ornith is a later option, recorded here.

- Corpus: 77 rows (75 ornith:9b, 2 olmo-3:7b — olmo's output mostly
  filtered as meta/short; needs a per-model prompt pass). Domains
  balanced 5–9; consensus skewed mid (3:34, 4:27, 1:8, 5:5, 2:3).
- TabICL (CPU, kv_cache on), 53 train / 24 test: difficulty acc
  **0.667** vs rule 0.25 vs random 0.167. Routing split 11 weak /
  13 strong (healthy, not collapsed). Cost −23% vs always-strong.
- Caveats: n=77 tiny; level 2 nearly absent; quality mapping is
  assumed (0.6/0.8), NOT empirical — charter Phase 4 grounding
  (weakest-model-that-succeeds) still open. No success claimed.

## Seed corpus + zero-data onboarding (2026-10-12)

- `seed.py` builds `data/seed-v1.jsonl`: 200 rows, 40/level, lowest
  judge-spread first (deterministic), slim schema
  `{question, difficulty, domain}` — 49KB, ships inside the wheel.
- `RouterPFNClassifier.from_seed(engine, roster)` fits on the bundle:
  one line, no user data. Verified live (TabICL/CUDA): trivial
  question → weak at 0.9999; terse halting-proof phrasing → difficulty
  2 → weak (known short-but-difficult bucket, still open).
- L5 seed rows are all contested (zero spread≤1.0 at L5) — best
  available, flagged, not hidden.
- Wheel verified (`uv build`): `routerpfn-0.1.0-py3-none-any.whl`,
  20 files, `router_pfn/data/seed-v1.jsonl` included. `from_seed`
  works from the installed layout (no repo paths).

## GPU shootout + N-model + features (2026-10-12)

- GPU (RTX 4070 8GB, 220 test rows), kv_cache verdict: TabICL-kv
  0.08s vs no-kv 0.30s per repeated predict — **~4× faster**, same
  acc 0.609. Fiorino-CUDA 0.614 acc at ~0.08s. Both route 220
  questions in under 0.1s (sub-ms/question). Defaults: kv ON.
- N-model rosters: `difficulty_to_tier` generalized to 2–5 models
  (`(d-1)*N//5` reproduces the old 2/3 cuts exactly; N=5 = identity).
  Tested N=2 (177/43), N=3 (68/142/10), N=5 (49/9/119/33/10) —
  m2 thin (9) reflects real L2 scarcity, not a bug.
- Features 22→24: `trivial_op_hit` + `bracket_data_hit` for the
  deceptive-simple bucket. Sort-descending probe moved pred 3→2
  (truth 1); re-bench acc **0.609→0.632**.
- Live PoC (gemma3:4b + ornith:9b, 5 fresh questions, real ollama
  calls): easy→weak correct; "prove halting undecidable" predicted
  difficulty 3 → routed weak — a real miss (terse-proof phrasing
  lacks surface complexity; charter "short but difficult" bucket).
  Recorded as the next error-analysis target, not hidden.

## Explicitly out of scope for v0.1

- Third-party benchmark suites (MMLU, GSM8K, HumanEval, etc.) as
  training data. May appear as *background context only* with per-set
  license checks if ever touched — currently untouched.
- Publishing model weights. Publishable artifacts: pip adapter +
  optional versioned synthetic dataset.
