"""Engine adapters for RouterPFN (MIT).

Every engine exposes fit(X, y)/predict_proba(X) over difficulty labels
{1..5} and returns a valid (n, 5) simplex. Real backends (tabpfn,
tabicl, kumo, fiorino) import lazily so `dummy` works with zero
optional deps. Missing dep -> clear ImportError naming the pip package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIFFICULTY_LEVELS: list[int] = [1, 2, 3, 4, 5]

_ENGINES: dict[str, str] = {
    "dummy": "no-install-needed",
    "tabpfn": "tabpfn",
    "tabicl": "tabicl",
    "kumo": "kumo-tabular",
    # Not on PyPI yet: frankiethull/fiorino (python/ subdir, dist `fiorino`).
    "fiorino": "fiorino @ git+https://github.com/frankiethull/fiorino.git#subdirectory=python",
}


def available_engines() -> list[str]:
    return sorted(_ENGINES)


class Engine:
    """Minimal backend contract shared by all engines."""

    name = "base"

    def fit(self, X: pd.DataFrame, y) -> Engine:
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class DummyEngine(Engine):
    """Offline baseline: class-frequency prior (tests, no deps)."""

    name = "dummy"

    def fit(self, X: pd.DataFrame, y) -> DummyEngine:
        y = np.asarray(y)
        self.classes_ = np.array(DIFFICULTY_LEVELS)
        counts = np.array([(y == c).sum() for c in DIFFICULTY_LEVELS], dtype=float)
        total = counts.sum()
        self.prior_ = counts / total if total > 0 else np.full(5, 0.2)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.tile(self.prior_, (n, 1))


class _SklearnTabularEngine(Engine):
    """Generic wrapper: any sklearn classifier over the 5 difficulty classes."""

    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y) -> _SklearnTabularEngine:
        self.estimator.fit(X, np.asarray(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        est = self.estimator
        proba = np.asarray(est.predict_proba(X))
        # Align columns to DIFFICULTY_LEVELS via est.classes_.
        out = np.zeros((len(X), len(DIFFICULTY_LEVELS)))
        for j, c in enumerate(est.classes_):
            if c in DIFFICULTY_LEVELS:
                out[:, DIFFICULTY_LEVELS.index(c)] = proba[:, j]
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums


def get_engine(name: str, **kwargs) -> Engine:
    """Resolve an engine by name; lazy-imports real backends."""
    key = (name or "dummy").lower()
    if key == "dummy":
        return DummyEngine()
    if key == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError("engine 'tabpfn' needs pip package 'tabpfn'") from e
        return _SklearnTabularEngine("tabpfn", TabPFNClassifier(**kwargs))
    if key == "tabicl":
        try:
            from tabicl import TabICLClassifier
        except ImportError as e:
            raise ImportError("engine 'tabicl' needs pip package 'tabicl'") from e
        # kv_cache defaults ON: the router's access pattern is repeated
        # predict() calls against one fitted context — exactly what
        # TabICL's KV cache accelerates.
        kwargs.setdefault("kv_cache", True)
        return _SklearnTabularEngine("tabicl", TabICLClassifier(**kwargs))
    if key == "kumo":
        # NVIDIA Kumo-Tabular (Sep 2026). BLOCKED as of 2026-10-11:
        # `structured-data-models` on PyPI is an empty 0.0.0a0
        # placeholder (no importable classifier), and the weights are
        # OpenMDW-1.1 (custom, not MIT/Apache-2.0). Re-probe when the
        # package ships a real API; verify license in MATERIALS.md first.
        raise ImportError(
            "engine 'kumo' is blocked: 'structured-data-models' is an "
            "empty placeholder with no classifier API (checked 2026-10-11), "
            "and Kumo-Tabular weights are OpenMDW-1.1 (license gate)"
        )
    if key == "fiorino":
        try:
            from fiorino import FiorinoClassifier
        except ImportError:
            try:
                # Forward-compat: `fiorino-tab` if/when it lands on PyPI.
                from fiorino_tab import FiorinoClassifier
            except ImportError as e:
                raise ImportError(
                    "engine 'fiorino' needs package 'fiorino' from GitHub "
                    "(not on PyPI yet): "
                    "pip install "
                    "'fiorino @ git+https://github.com/frankiethull/fiorino.git"
                    "#subdirectory=python' "
                    "or pip install 'routerpfn[fiorino] "
                    "@ git+https://github.com/frankiethull/RouterPFN.git'"
                ) from e
        return _SklearnTabularEngine("fiorino", FiorinoClassifier(**kwargs))
    raise ValueError(f"unknown engine {name!r}; expected one of {available_engines()}")
