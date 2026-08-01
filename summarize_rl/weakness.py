"""Weakness-driven continual RL: real-time failure accumulation (Phase 2).

The trainers already compute a `RewardBreakdown` per rollout. This module turns
that free signal into a persistent, per-rollout failure log with (almost) zero
extra compute: `FailureLog.record` just compares each axis against a threshold
and appends the failures to JSONL. Diagnosis (Phase 3) reads the log back.

Axis-sign convention (critical): in `rewards.compute_reward` the content axes
(`faithfulness`, `coverage`, `key_sentence`, `contrast`) are *added* to the
reward — low is bad — while the penalty axes (`hallucination`, `copy_penalty`,
`length_penalty`) are stored as positive magnitudes and *subtracted* — high is
bad. `FailureLog` flips the comparison for the penalty axes accordingly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .branches import Example
from .rewards import RewardBreakdown, _FACT_RE

# Penalty axes: value is a positive magnitude, higher == worse, so a rollout
# fails the axis when its value is ABOVE the threshold (the mirror of the
# content axes, which fail below it).
PENALTY_AXES = {"hallucination", "copy_penalty", "length_penalty"}

# Default thresholds. `key_sentence` is deliberately excluded: when key-sentence
# scoring is disabled the field is 0.0 for every rollout, which would flag the
# whole corpus. Callers enable it explicitly (e.g. when config.reward.w_keysent
# > 0) via the `thresholds` argument.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.5,   # content: fail if <
    "coverage": 0.4,       # content: fail if <
    "hallucination": 0.3,  # penalty: fail if >
    "copy_penalty": 0.5,   # penalty: fail if >
}

# Terrain archetypes mirrored from data/gen_test_scenarios.py, used to profile
# which environments failures cluster in.
_TERRAIN_KEYWORDS = ("도시", "산악", "삼림", "하천", "해안", "사막")


@dataclass
class FailureRecord:
    step: int
    example_id: str
    axis: str
    score: float
    summary: str
    meta: dict = field(default_factory=dict)


def default_thresholds(reward_config) -> dict[str, float]:
    """`DEFAULT_THRESHOLDS`, plus `key_sentence` iff key-sentence scoring is on.

    `key_sentence` is only a meaningful signal when `w_keysent > 0`; otherwise the
    field is a constant 0.0 and would mark every rollout as failing.
    """
    thr = dict(DEFAULT_THRESHOLDS)
    if getattr(reward_config, "w_keysent", 0.0) > 0:
        thr["key_sentence"] = 0.4
    return thr


def _is_failure(axis: str, value: float, threshold: float) -> bool:
    if axis in PENALTY_AXES:
        return value > threshold
    return value < threshold


def extract_meta(example: Example) -> dict:
    """Scenario metadata used by Phase 4 to steer synthesis without copying text.

    Mines structural characteristics (numeric density, echelon count, terrain,
    length) so weak-axis synthesis reflects *what kind* of scenario fails, not
    the failed text itself.
    """
    src = example.source or ""
    numeric = [m for m in _FACT_RE.findall(src) if any(ch.isdigit() for ch in m)]
    heads = {t.head for t in example.triplets}
    terrain = [kw for kw in _TERRAIN_KEYWORDS if kw in src]
    return {
        "numeric_tokens": len(numeric),
        "n_units": len(heads),
        "n_keyfacts": len(example.keyfacts),
        "source_len": len(src),
        "terrain": terrain,
    }


class FailureLog:
    """Accumulate rollout-level failures to JSONL during training.

    The trainer calls `record(step, example, rollout_text, breakdown)` per
    rollout; only the axes below/above their thresholds are written. In-memory
    counts feed TensorBoard (`weakness/fail_rate_<axis>`); the JSONL survives
    crashes and is read by the diagnoser, possibly from a separate process.
    """

    def __init__(self, path: str, thresholds: dict[str, float] | None = None):
        self.path = path
        self.thresholds = dict(thresholds) if thresholds is not None else dict(DEFAULT_THRESHOLDS)
        self._counts: dict[str, int] = {}
        self._total_rollouts = 0

    def record(self, step: int, example: Example, rollout_text: str,
               bd: RewardBreakdown) -> None:
        self._total_rollouts += 1
        meta = None
        for axis, thr in self.thresholds.items():
            val = float(getattr(bd, axis))
            if _is_failure(axis, val, thr):
                self._counts[axis] = self._counts.get(axis, 0) + 1
                if meta is None:
                    meta = extract_meta(example)
                self._append(FailureRecord(
                    step=step,
                    example_id="" if example.id is None else str(example.id),
                    axis=axis,
                    score=val,
                    summary=rollout_text,
                    meta=meta,
                ))

    def failure_rates(self) -> dict[str, float]:
        n = max(1, self._total_rollouts)
        return {axis: c / n for axis, c in self._counts.items()}

    def tb_scalars(self) -> dict[str, float]:
        """Failure rates keyed for `TensorBoardLogger.log_scalars`."""
        return {f"weakness/fail_rate_{a}": v for a, v in self.failure_rates().items()}

    def _append(self, rec: FailureRecord) -> None:
        import os
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def load_window(path: str, window_steps: int, now: int) -> list[FailureRecord]:
    """Load failure records with `step >= now - window_steps` (inclusive)."""
    import os
    if not os.path.exists(path):
        return []
    lo = now - window_steps
    out: list[FailureRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("step", 0) < lo:
                continue
            out.append(FailureRecord(
                step=d["step"], example_id=d.get("example_id", ""),
                axis=d["axis"], score=d.get("score", 0.0),
                summary=d.get("summary", ""), meta=d.get("meta", {}),
            ))
    return out
