# RouterPFN

**RouterPFN** — a TabPFN-extension *adapter* (pip package, MIT) that routes an
incoming LLM question to the cheapest capable model from *your* roster. Not a
checkpoint, not an LLM, not a HuggingFace release.

```
QUESTION → deterministic features → tabular FM (difficulty 1–5) → policy → model id
```

The classifier predicts portable **difficulty 1–5**. A separate policy layer
maps that onto whatever models you actually have — 2 or 3 opaque ids such as
`opencode/muse-spark-1.3-free` or `ollama/gemma4`. Default mapping: N=2 →
`{1,2,3}→weak, {4,5}→strong`; N=3 → `{1,2}→cheap, {3,4}→standard, {5}→frontier`.

## Install

```bash
pip install RouterPFN                 # core (dummy engine, offline)
pip install "RouterPFN[tabicl]"       # TabICL backend (recommended)
pip install "RouterPFN[laya]"         # Laya decision-model rater for corpus builds
# Fiorino is not on PyPI yet — it lives at frankiethull/fiorino:
pip install "routerpfn[fiorino] @ git+https://github.com/frankiethull/RouterPFN.git"
# or directly:
pip install "fiorino @ git+https://github.com/frankiethull/fiorino.git#subdirectory=python"
```

Or install everything from GitHub (gets the fiorino extra too):

```bash
pip install "routerpfn[all] @ git+https://github.com/frankiethull/RouterPFN.git"
```

Requires Python ≥ 3.10. Tabular engines stay frozen: `fit` stores the
in-context table, `predict` answers in forward passes. Nothing is trained.

## Use

```python
from router_pfn import RouterPFNClassifier

roster = ["ollama/gemma3:4b", "opencode/muse-spark-1.3-free"]

# No data? Fit on the bundled 200-question seed corpus in one line.
clf = RouterPFNClassifier.from_seed(engine="tabicl", roster=roster)
print(clf.route("Prove that ..."))  # {"model": ..., "confidence": ..., ...}

# Your own labels later:
clf = RouterPFNClassifier(engine="tabicl", roster=roster)
clf.fit(questions, difficulties)   # difficulties in {1..5}
```

Engines (`--engine` / `engine=`): `dummy` (always available, tests),
`tabicl` (KV-cached repeated inference), `fiorino`, `tabpfn`. Kumo is
explicitly blocked (placeholder package + custom license) — see `docs/MATERIALS.md`.

## CLI

```bash
python -m router_pfn.synth --n 500                        # template corpus (offline)
python -m router_pfn.teacher --generators o1,o2 --judges j1  # LLM + Laya-judged corpus (needs ollama)
python -m router_pfn.ground --n-per-level 30              # empirical weak-vs-strong grounding
python -m router_pfn.bench --data data/raw/corpus.jsonl --engine tabicl
python -m router_pfn.shootout                             # backend comparison on identical splits
python -m router_pfn.publish --version v1 --tarball       # file-based release bundle (no registry)
```

## Measured (single 8GB card, 732-row synth corpus, 128 grounded)

* Difficulty accuracy **0.63** vs 0.21 rule baseline; tier agreement **0.82
  against measured** weakest-that-succeeds labels.
* Following the router: **0.77 success at ~40% of the frontier bill**
  (always-weak 0.54, always-strong 0.88). A money saver, not a time saver.
* GPU: TabICL kv_cache ≈ 4× faster repeated inference (0.08s vs 0.30s per
  220 questions); Fiorino matches TabICL accuracy (0.614 vs 0.609) at the
  same GPU speed. Roster support is N-model generic (2–5 models tested).
  See `docs/MATERIALS.md` for the full ledger.

## Layout

```
src/router_pfn/   features engines router policy synth teacher ground
                  labels eval bench shootout publish config
tests/            contract + unit tests (dummy engine: fully offline)
docs/             scope, charter, materials/license ledger
```

## License

MIT. No model weights ship here. Engine checkpoints download from their
own upstreams on first use (see `docs/MATERIALS.md` for the license ledger).
