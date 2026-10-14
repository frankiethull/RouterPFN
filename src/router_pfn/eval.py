"""Evaluation harness for RouterPFN (MIT).

Headline comparison (charter Phase 9): ROUTER vs ALWAYS-STRONG —
quality retained + cost saved. Routing accuracy alone is never the
success claim.
"""

from __future__ import annotations

import numpy as np


def routing_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def tier_report(routes: list[dict], roster: list[str]) -> dict:
    counts = {m: 0 for m in roster}
    for r in routes:
        counts[r["model"]] = counts.get(r["model"], 0) + 1
    total = len(routes) or 1
    return {
        "counts": counts,
        "fractions": {m: c / total for m, c in counts.items()},
        "escalation_rate": float(sum(1 for r in routes if r.get("escalated")) / total),
    }


def cost_quality_summary(
    routes: list[dict],
    roster: list[str],
    costs: dict[str, float],
    quality: dict[str, float],
) -> dict:
    """Compare router cost/quality against always-weak / always-strong.

    costs: model_id -> relative cost per query. quality: model_id ->
    assumed task quality (0..1) for the evaluated slice. v0.1 takes
    these as inputs; v0.2 grounds them empirically (charter Phase 4).
    """
    n = len(routes) or 1
    router_cost = sum(costs[r["model"]] for r in routes) / n
    router_quality = sum(quality[r["model"]] for r in routes) / n
    strong, weak = roster[-1], roster[0]
    return {
        "router_cost": router_cost,
        "router_quality": router_quality,
        "always_strong_cost": costs[strong],
        "always_strong_quality": quality[strong],
        "always_weak_cost": costs[weak],
        "always_weak_quality": quality[weak],
        "cost_saved_vs_strong": costs[strong] - router_cost,
        "quality_retained_vs_strong": router_quality - quality[strong],
    }
