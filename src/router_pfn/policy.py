"""Tier mapping + routing policies (MIT).

The classifier predicts difficulty 1–5 (portable). This module maps a
difficulty distribution onto the user's roster (2–5 opaque model ids)
and applies the decision policy. Classifier ≠ policy (charter Phase 7).
"""

from __future__ import annotations

import numpy as np

VALID_POLICIES = ("argmax", "threshold", "conservative", "cost_aware")


def difficulty_to_tier(
    difficulty: int, roster_size: int, tier_bounds: list[int] | None = None
) -> int:
    """Map a 1–5 difficulty to a tier index in [0, roster_size).

    Default (no bounds): tier = min(N-1, (d-1)*N//5), which reproduces
    the classic cuts — N=2: {1,2,3}→0, {4,5}→1; N=3: {1,2}→0, {3,4}→1,
    {5}→2 — and extends to N=4 ({1,2}→0, {3}→1, {4}→2, {5}→3) and N=5
    (identity: difficulty d → model d-1).
    Explicit tier_bounds: N-1 ascending cut points (difficulty <= cut
    stays in the lower tier).
    """
    if not 2 <= roster_size <= 5:
        raise ValueError("roster must hold 2–5 models")
    if tier_bounds:
        cuts = sorted(tier_bounds)
        if len(cuts) != roster_size - 1:
            raise ValueError(
                f"need {roster_size - 1} cut points for {roster_size} models, "
                f"got {len(cuts)}"
            )
        for tier, cut in enumerate(cuts):
            if difficulty <= cut:
                return tier
        return roster_size - 1
    return min(roster_size - 1, (difficulty - 1) * roster_size // 5)


def apply_policy(
    proba: np.ndarray,
    roster: list[str],
    policy: str = "argmax",
    tier_bounds: list[int] | None = None,
    confidence_threshold: float = 0.6,
    costs: list[float] | None = None,
) -> tuple[str, float, bool]:
    """Map one (5,) difficulty distribution to (model_id, confidence, escalated).

    - argmax: expected-difficulty tier via probability mass per tier.
    - threshold: escalate one tier up when max tier-mass < threshold.
    - conservative: escalate when top-two tier masses are within 0.1.
    - cost_aware: escalate when P(needs stronger) * cost_gap justifies it;
      falls back to threshold behaviour when costs are absent.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected {VALID_POLICIES}")
    if not 2 <= len(roster) <= 5:
        raise ValueError("roster must hold 2–5 models")
    p = np.asarray(proba, dtype=float)
    if p.shape != (5,) or not np.isfinite(p).all() or p.sum() <= 0:
        raise ValueError("proba must be a finite (5,) simplex")
    p = p / p.sum()

    tier_mass = np.zeros(len(roster))
    for i, d in enumerate((1, 2, 3, 4, 5)):
        tier_mass[difficulty_to_tier(d, len(roster), tier_bounds)] += p[i]
    top = int(tier_mass.argmax())
    confidence = float(tier_mass[top])
    escalated = False

    if policy == "threshold" and confidence < confidence_threshold:
        if top < len(roster) - 1:
            top += 1
            escalated = True
    elif policy == "conservative":
        order = np.argsort(tier_mass)[::-1]
        if len(order) > 1 and tier_mass[order[0]] - tier_mass[order[1]] < 0.1:
            if top < len(roster) - 1:
                top += 1
                escalated = True
    elif policy == "cost_aware":
        if costs is not None and len(costs) == len(roster):
            stronger_mass = float(tier_mass[top + 1 :].sum())
            gap = max(costs[top + 1] - costs[top], 0.0) if top < len(roster) - 1 else 0.0
            if top < len(roster) - 1 and stronger_mass * (1.0 + gap) > (
                1.0 - confidence_threshold
            ):
                top += 1
                escalated = True
        elif confidence < confidence_threshold and top < len(roster) - 1:
            top += 1
            escalated = True
    return roster[top], confidence, escalated
