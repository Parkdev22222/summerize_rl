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
import os
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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def record(self, step: int, example: Example, rollout_text: str,
               bd: RewardBreakdown) -> None:
        self._total_rollouts += 1
        failing = []
        for axis, thr in self.thresholds.items():
            val = float(getattr(bd, axis))
            if _is_failure(axis, val, thr):
                self._counts[axis] = self._counts.get(axis, 0) + 1
                failing.append((axis, val))
        if not failing:
            return
        # One extract_meta + one file open per failing rollout (not per axis):
        # keeps the per-rollout training hook near-zero-overhead.
        meta = extract_meta(example)
        ex_id = "" if example.id is None else str(example.id)
        self._append([
            FailureRecord(step=step, example_id=ex_id, axis=axis, score=val,
                          summary=rollout_text, meta=meta)
            for axis, val in failing
        ])

    def failure_rates(self) -> dict[str, float]:
        n = max(1, self._total_rollouts)
        return {axis: c / n for axis, c in self._counts.items()}

    def tb_scalars(self) -> dict[str, float]:
        """Failure rates keyed for `TensorBoardLogger.log_scalars`."""
        return {f"weakness/fail_rate_{a}": v for a, v in self.failure_rates().items()}

    def _append(self, recs: list[FailureRecord]) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Phase 3: diagnosis. Aggregate the windowed failure log, triage a sample with
# the multi-axis judge, and confirm weak axes under three gates (absolute rate,
# consecutive rounds, judge agreement), with RAW-ceiling reclassification.
# ---------------------------------------------------------------------------

# Reward-breakdown axis -> the diagnostic judge axis that corroborates it.
# copy_penalty has no semantic analog, so it is trusted from lexical evidence.
_REWARD_TO_JUDGE = {
    "hallucination": "hallucination",
    "coverage": "omission",
    "key_sentence": "omission",
    "faithfulness": "entity",
}


def axis_failure_rates(records: list[FailureRecord], total_rollouts: int) -> dict[str, float]:
    """Per-axis failure rate = (failures on that axis) / total rollouts in window."""
    n = max(1, total_rollouts)
    counts: dict[str, int] = {}
    for r in records:
        counts[r.axis] = counts.get(r.axis, 0) + 1
    return {axis: c / n for axis, c in counts.items()}


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


@dataclass
class MetaProfile:
    """Aggregated failure metadata, used by Phase 4 to steer synthesis intensity."""

    by_axis: dict[str, dict]
    overall_numeric_median: float

    def severity(self, axis: str) -> float:
        """Synthesis intensity for `axis`: how numeric-dense its failing scenarios
        are relative to the corpus, clamped to [0.5, 2.0] (1.0 = no signal)."""
        info = self.by_axis.get(axis)
        if not info or self.overall_numeric_median <= 0:
            return 1.0
        ratio = info["numeric_median"] / self.overall_numeric_median
        return max(0.5, min(2.0, ratio))


def failure_meta_profile(records: list[FailureRecord]) -> MetaProfile:
    by_axis: dict[str, dict] = {}
    all_numeric = [float(r.meta.get("numeric_tokens", 0)) for r in records]
    for axis in {r.axis for r in records}:
        rows = [r for r in records if r.axis == axis]
        by_axis[axis] = {
            "count": len(rows),
            "numeric_median": _median([float(r.meta.get("numeric_tokens", 0)) for r in rows]),
            "source_len_median": _median([float(r.meta.get("source_len", 0)) for r in rows]),
            "n_units_median": _median([float(r.meta.get("n_units", 0)) for r in rows]),
        }
    return MetaProfile(by_axis=by_axis, overall_numeric_median=_median(all_numeric))


def triage_with_judge(records, source_lookup, multi_judge, k_per_axis=20,
                      confirm_below=0.5) -> dict[str, float]:
    """Re-score up to k representative failures per axis; return the confirm rate.

    A failure is *confirmed* when the judge's corresponding axis also scores it
    bad (score < confirm_below). Axes with no judge analog (copy_penalty) are
    trusted from lexical evidence (confirm rate 1.0). Returns {} if no judge.
    """
    if multi_judge is None:
        return {}
    out: dict[str, float] = {}
    axes = {r.axis for r in records}
    for axis in axes:
        judge_axis = _REWARD_TO_JUDGE.get(axis)
        if judge_axis is None:
            out[axis] = 1.0
            continue
        sample = [r for r in records if r.axis == axis][:k_per_axis]
        confirmed = 0
        scored = 0
        for r in sample:
            source = (source_lookup or {}).get(r.example_id)
            if source is None:
                continue
            v = multi_judge.score_axes(source, r.summary)
            if not v or judge_axis not in v:
                continue
            scored += 1
            if v[judge_axis] < confirm_below:
                confirmed += 1
        out[axis] = (confirmed / scored) if scored else 0.0
    return out


@dataclass
class DiagnosisReport:
    weak_axes: list[str]
    backbone_limited: list[str]
    rates: dict[str, float]
    meta_stats: MetaProfile
    confirmed: dict[str, float]
    candidates: list[str]

    def converged(self, min_rate: float = 0.15) -> bool:
        """True when every axis's failure rate is below `min_rate`."""
        return all(r < min_rate for r in self.rates.values())

    def to_dict(self) -> dict:
        """Persistable view (drops the derived meta_stats/confirmed objects)."""
        return {"weak_axes": self.weak_axes, "backbone_limited": self.backbone_limited,
                "candidates": self.candidates, "rates": self.rates}

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosisReport":
        """Restore a report from state.json. meta_stats/confirmed aren't persisted
        (the resume path only needs candidates for the consecutive-round rule)."""
        return cls(
            weak_axes=d.get("weak_axes", []), backbone_limited=d.get("backbone_limited", []),
            rates=d.get("rates", {}), meta_stats=MetaProfile(by_axis={}, overall_numeric_median=0.0),
            confirmed={}, candidates=d.get("candidates", []),
        )


@dataclass
class HeldoutReport:
    """Mean per-(judge)-axis score over the 30-scenario held-out set."""

    axis_scores: dict[str, float]

    def regressed(self, prev: "HeldoutReport | None", margin: float = 0.05) -> bool:
        """True if any axis dropped by more than `margin` vs the previous round."""
        if prev is None:
            return False
        for axis, now in self.axis_scores.items():
            before = prev.axis_scores.get(axis)
            if before is not None and (before - now) > margin:
                return True
        return False


class WeaknessDiagnoser:
    """Confirm weak axes from the windowed failure log under three gates."""

    def __init__(self, min_rate: float = 0.15, consecutive: int = 2,
                 min_confirm: float = 0.5, k_per_axis: int = 20):
        self.min_rate = min_rate
        self.consecutive = consecutive
        self.min_confirm = min_confirm
        self.k_per_axis = k_per_axis

    def diagnose(self, records, total_rollouts, *, source_lookup=None,
                 multi_judge=None, history=None, raw_axis_rates=None) -> DiagnosisReport:
        history = history or []
        rates = axis_failure_rates(records, total_rollouts)
        meta_stats = failure_meta_profile(records)
        candidates = [a for a, r in rates.items() if r >= self.min_rate]
        confirmed = triage_with_judge(records, source_lookup, multi_judge,
                                       k_per_axis=self.k_per_axis)

        weak: list[str] = []
        for axis in candidates:
            if not self._consecutive_ok(axis, history):
                continue
            if confirmed and confirmed.get(axis, 0.0) < self.min_confirm:
                continue
            weak.append(axis)

        # RAW ceiling: if the frozen base fails the axis too, no amount of policy
        # training / synthetic data can fix it -> exclude from synthesis targets.
        backbone_limited: list[str] = []
        if raw_axis_rates:
            kept = []
            for axis in weak:
                if raw_axis_rates.get(axis, 0.0) >= self.min_rate:
                    backbone_limited.append(axis)
                else:
                    kept.append(axis)
            weak = kept

        return DiagnosisReport(
            weak_axes=weak, backbone_limited=backbone_limited, rates=rates,
            meta_stats=meta_stats, confirmed=confirmed, candidates=candidates,
        )

    def _consecutive_ok(self, axis: str, history: list) -> bool:
        need = self.consecutive - 1
        if need <= 0:
            return True
        recent = history[-need:]
        if len(recent) < need:
            return False
        return all(axis in rep.candidates for rep in recent)


def breakdown_failures(bd: RewardBreakdown, thresholds: dict[str, float]) -> set[str]:
    """The set of axes this breakdown fails, with penalty-axis sign handling.

    Reused to compute RAW-base per-axis failure rates for the RAW-ceiling gate.
    """
    return {axis for axis, thr in thresholds.items()
            if _is_failure(axis, float(getattr(bd, axis)), thr)}


def eval_heldout(summarizer, examples, multi_judge) -> "HeldoutReport":
    """Summarize each held-out example and average the judge's per-axis scores.

    Used both to cross-check that a weak axis is genuinely weak (not just a
    training-data artifact) and to detect round-over-round regression.
    """
    from .judge import AXES as _JUDGE_AXES

    sums: dict[str, float] = {a: 0.0 for a in _JUDGE_AXES}
    counts: dict[str, int] = {a: 0 for a in _JUDGE_AXES}
    for ex in examples:
        res = summarizer.summarize(ex.source, query=ex.query, triplets=ex.triplets)
        scores = multi_judge.score_axes(ex.source, res.text)
        if not scores:
            continue
        for axis, val in scores.items():
            if axis in sums:
                sums[axis] += val
                counts[axis] += 1
    axis_scores = {a: (sums[a] / counts[a]) for a in _JUDGE_AXES if counts[a] > 0}
    return HeldoutReport(axis_scores=axis_scores)


def load_window(path: str, window_steps: int, now: int) -> list[FailureRecord]:
    """Load failure records with `step >= now - window_steps` (inclusive)."""
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
