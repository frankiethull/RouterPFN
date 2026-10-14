"""RouterPFNClassifier — sklearn estimator over difficulty 1–5 (MIT).

v0.1: frozen tabular-FM posture. fit() featurizes nothing itself —
callers pass either raw question strings (auto-featurized) or a
feature DataFrame. The engine stores the in-context table; no
gradients. route() maps the difficulty distribution onto the user's
roster via policy.apply_policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .engines import DIFFICULTY_LEVELS, Engine, get_engine
from .features import FEATURE_NAMES, featurize
from .policy import VALID_POLICIES, apply_policy


def _to_feature_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if isinstance(X, (list, tuple, np.ndarray)) and len(X) > 0 and isinstance(
        X[0], str
    ):
        return pd.DataFrame([featurize(q) for q in X], columns=FEATURE_NAMES)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be question strings or a 2D feature array")
    cols = FEATURE_NAMES[: X.shape[1]] if X.shape[1] <= len(FEATURE_NAMES) else [
        f"f{i}" for i in range(X.shape[1])
    ]
    return pd.DataFrame(X, columns=cols)


class RouterPFNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        engine: str = "dummy",
        policy: str = "argmax",
        roster: list[str] | None = None,
        tier_bounds: list[int] | None = None,
        confidence_threshold: float = 0.6,
        random_state: int = 0,
        **engine_kwargs,
    ):
        self.engine = engine
        self.policy = policy
        self.roster = roster
        self.tier_bounds = tier_bounds
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.engine_kwargs = engine_kwargs

    # -- fit ---------------------------------------------------------
    def fit(self, X, y):
        if self.policy not in VALID_POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}")
        Xf = _to_feature_frame(X)
        y = np.asarray(y)
        bad = set(np.unique(y)) - set(DIFFICULTY_LEVELS)
        if bad:
            raise ValueError(f"y must be difficulty levels 1–5, got {sorted(bad)}")
        self.classes_ = np.array(DIFFICULTY_LEVELS)
        self.feature_names_in_ = np.array(Xf.columns.tolist(), dtype=object)
        self.n_features_in_ = Xf.shape[1]
        roster = list(self.roster) if self.roster else ["weak-local", "strong-frontier"]
        if not 2 <= len(roster) <= 5:
            raise ValueError("roster must hold 2–5 models")
        self.roster_ = roster
        self._engine: Engine = get_engine(self.engine, **self.engine_kwargs)
        self._engine.fit(Xf, y)
        return self

    # -- sklearn API -------------------------------------------------
    def predict_proba(self, X):
        check_is_fitted(self)
        Xf = _to_feature_frame(X)
        proba = np.asarray(self._engine.predict_proba(Xf), dtype=float)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums

    def predict(self, X):
        check_is_fitted(self)
        proba = self.predict_proba(X)
        return np.array(DIFFICULTY_LEVELS)[proba.argmax(axis=1)]

    # -- routing -----------------------------------------------------
    def route(self, question: str) -> dict:
        """Route one question string to a model id in the roster."""
        check_is_fitted(self)
        proba = self.predict_proba([question])[0]
        model_id, confidence, escalated = apply_policy(
            proba,
            self.roster_,
            policy=self.policy,
            tier_bounds=self.tier_bounds,
            confidence_threshold=self.confidence_threshold,
        )
        return {
            "model": model_id,
            "difficulty_proba": {
                str(d): float(p) for d, p in zip(DIFFICULTY_LEVELS, proba, strict=True)
            },
            "predicted_difficulty": int(np.array(DIFFICULTY_LEVELS)[proba.argmax()]),
            "confidence": float(confidence),
            "escalated": bool(escalated),
            "policy": self.policy,
        }

    def route_many(self, questions: list[str]) -> list[dict]:
        return [self.route(q) for q in questions]

    @classmethod
    def from_seed(cls, engine: str = "dummy",
                  roster: list[str] | None = None, **kwargs) -> RouterPFNClassifier:
        """Fit a router on the bundled seed corpus — no user data needed.

        Zero-data onboarding: the package ships 200 stratified,
        judge-agreement-ranked questions so one caller line gets a
        working router. Bring your own corpus later via fit().
        """
        from .seed import seed_questions_labels

        questions, labels = seed_questions_labels()
        return cls(engine=engine, roster=roster, **kwargs).fit(questions, labels)
