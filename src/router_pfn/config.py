"""RouterPFN expert config: paths, env-overridable defaults (MIT).

Roster entries are opaque model-id strings (e.g.
"opencode/muse-spark-1.3-free", "ollama/gemma4"). The router never calls
the LLM itself in v0.1 — it returns the id to call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EXPERT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = EXPERT_DIR / "reports"
MODELS_DIR = EXPERT_ROOT / "models" / "router-pfn"


def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    return default if raw is None or raw == "" else raw


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _parse_roster(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return ["weak-local", "strong-frontier"]
    try:
        val = json.loads(raw)
        if isinstance(val, list) and all(isinstance(m, str) for m in val):
            return val
    except json.JSONDecodeError:
        pass
    return [m.strip() for m in raw.split(",") if m.strip()]


def _parse_bounds(raw: str, n: int) -> list[int]:
    """Parse e.g. "3" (N=2 cut) or "2,4" (N=3 cuts) into cut points.

    Empty → [] meaning "use the formula default" (policy.difficulty_to_tier).
    Explicit bounds must hold exactly N-1 cuts.
    """
    if not 2 <= n <= 5:
        raise ValueError("roster must hold 2–5 models")
    if raw.strip() == "":
        return []
    try:
        cuts = [int(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        cuts = []
    if len(cuts) != n - 1:
        raise ValueError(f"need {n - 1} cut points for {n} models, got {len(cuts)}")
    return sorted(cuts)


@dataclass(frozen=True)
class RouterConfig:
    """Defaults; all settable via env or ctor params (mirrors JabularConfig)."""

    engine: str = field(default_factory=lambda: _env_str("ROUTER_ENGINE", "dummy"))
    policy: str = field(default_factory=lambda: _env_str("ROUTER_POLICY", "argmax"))
    confidence_threshold: float = field(
        default_factory=lambda: _env_float("ROUTER_CONFIDENCE_THRESHOLD", 0.6)
    )
    random_state: int = field(default_factory=lambda: _env_int("ROUTER_SEED", 0))
    max_rows: int = field(default_factory=lambda: _env_int("ROUTER_MAX_ROWS", 2048))

    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    reports_dir: Path = field(default_factory=lambda: REPORTS_DIR)
    models_dir: Path = field(default_factory=lambda: MODELS_DIR)

    @property
    def roster(self) -> list[str]:
        return _parse_roster(_env_str("ROUTER_ROSTER_JSON", ""))

    @property
    def tier_bounds(self) -> list[int]:
        return _parse_bounds(
            _env_str("ROUTER_TIER_BOUNDS", ""), len(self.roster)
        )


HF_DATASET_REPO = os.getenv("ROUTER_HF_DATASET", "Nanite-Labs/router-pfn-synth-v0")


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
