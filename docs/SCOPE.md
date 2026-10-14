# RouterPFN Expert — Project Scope

**Codename:** RouterPFN (TabPFN-extension adapter family)

## What this is

RouterPFN is a **TabPFN extension package / adapter** for LLM routing — not
a new HuggingFace checkpoint. A small decision system around frozen
tabular foundation models:

```
QUESTION → deterministic FEATURES → TABULAR FM (1-5 difficulty) → POLICY → MODEL ID
```

The core classifier predicts portable **difficulty 1–5**. A separate
`tier_mapper + policy` layer maps that distribution onto whatever model
roster the user actually has (2–5 models, opaque id strings such as
`opencode/muse-spark-1.3-free` or `ollama/gemma4`).

Default mapping: N=2 → `{1,2,3}→weak, {4,5}→strong`;
N=3 → `{1,2}→cheap, {3,4}→standard, {5}→frontier`;
N=5 → identity (difficulty d → model d-1).
All bounds configurable via `config.py` / env — never hard-coded.

## What this is NOT

- **Not** a new tabular FM pretrain. Engines (TabPFN, TabICL,
  Kumo-Tabular, Fiorino) stay frozen; `fit` stores in-context rows only.
- **Not** another LLM. Outputs are typed: difficulty distribution +
  routed model id. No chat, no prose.
- **Not** a fixed 3-model router. The roster is user-supplied; the
  classifier never sees model ids during training.
- **Not** an embedding/LLM-feature pipeline in v0.1. Inference features
  are deterministic and CPU-cheap so extraction never erases routing
  savings (charter Phase 5 guardrail).

## Engine options (`--engine`)

| Engine | Package | Status in v0.1 |
|---|---|---|
| `dummy` | none (tests/offline) | always available |
| `tabpfn` | `tabpfn` (Prior Labs) | lazy import, primary |
| `tabicl` | `tabicl` | lazy import |
| `kumo` | `kumo-tabular` | lazy import |
| `fiorino` | `fiorino` from `github.com/frankiethull/fiorino` (python/ subdir, not on PyPI yet) | lazy import |

Missing optional dep → clear error naming the pip package, never a
silent fallback.

## Pipelines

```
synth (templates + optional local teacher → 1-5 judged) → features
  → engine.fit → router.route → eval (vs always-weak/always-strong)
```

Composable modules under `src/router_pfn/`, config via `config.py`.
Single-package rule: the importable package is `router_pfn`, and this
repo is its one and only home — no mirrors, no second copies.

## Guardrails (hard, from charter)

- MIT for everything shipped.
- Strict train/val/test separation; never optimize against test.
- No model-evaluation info leaks into inference features.
- No expensive LLM calls for features in v0.1.
- Never claim success from routing accuracy alone — headline metric is
  **quality retained vs always-strong + cost saved**.
- Synthetic eval data is license-clean by construction (templates +
  local teacher); no NC sources, no redistribution of third-party sets.
