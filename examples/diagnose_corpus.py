"""Corpus-level weakness diagnosis over an accumulated failure log.

Unlike examples/diagnose.py (a single-example RAW/NEUTRAL/TRAINED view), this
aggregates a whole training run's `--failure-log` JSONL into per-axis failure
rates and confirmed weak axes using summarize_rl.weakness.WeaknessDiagnoser.

Offline (no model needed):
    python -m examples.diagnose_corpus --failure-log runs/x/failures.jsonl \
        --total-rollouts 4000 --window-steps 1000

The diagnoser's lexical aggregation and confirmation gates run without a
backbone. Judge triage / held-out cross-checks (which need a real model) are
wired in the continual orchestrator (examples/run_continual.py).
"""

from __future__ import annotations

import argparse

from summarize_rl.weakness import WeaknessDiagnoser, load_window


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--failure-log", required=True, help="failures.jsonl from a --failure-log run")
    p.add_argument("--total-rollouts", type=int, required=True,
                   help="rollouts in the window (denominator for failure rates)")
    p.add_argument("--window-steps", type=int, default=10**9,
                   help="only count failures with step >= now-window (default: all)")
    p.add_argument("--now", type=int, default=None,
                   help="current step (default: max step in the log)")
    p.add_argument("--min-rate", type=float, default=0.15)
    p.add_argument("--consecutive", type=int, default=1,
                   help="consecutive-round rule (1 for a one-shot diagnosis)")
    args = p.parse_args()

    # Load the whole log once; derive `now` from it when unset, then window
    # in memory (no second file read/parse).
    all_recs = load_window(args.failure_log, window_steps=10**9, now=10**9)
    now = args.now if args.now is not None else max((r.step for r in all_recs), default=0)
    lo = now - args.window_steps
    records = [r for r in all_recs if r.step >= lo]

    diag = WeaknessDiagnoser(min_rate=args.min_rate, consecutive=args.consecutive)
    report = diag.diagnose(records, args.total_rollouts, history=[])

    print(f"failures in window: {len(records)}  total rollouts: {args.total_rollouts}")
    print("per-axis failure rate:")
    for axis, r in sorted(report.rates.items(), key=lambda kv: -kv[1]):
        mark = "  <-- candidate" if r >= args.min_rate else ""
        print(f"  {axis:>14}: {r:.3f}{mark}")
    print(f"candidates (>= {args.min_rate}): {report.candidates}")
    print(f"converged: {report.converged(args.min_rate)}")


if __name__ == "__main__":
    main()
