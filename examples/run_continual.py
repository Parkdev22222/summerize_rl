"""Weakness-driven continual RL orchestrator (Phase 5).

Loops: train n steps (FailureLog on) -> diagnose the windowed failure log ->
cross-check + regression-guard on the 30 held-out scenarios -> synthesize
weak-axis data (backbone-limited axes excluded) -> merge with a replay ratio ->
repeat until the weak axes converge. Round boundaries checkpoint the policy and
a resumable state.json.

The round loop (`run_continual`) is backbone-agnostic and driven by injected
objects, so it runs on MockBackend in tests. The CLI (`main`) reuses
examples.train_real.build_trainer, adding `--model mock` for a CPU smoke run:

    python -m examples.run_continual --model mock --data data/test_scenarios_ko.jsonl \
        --limit 6 --rounds 2 --steps-per-round 6 --num-samples 3 \
        --max-new-tokens 6 --min-new-tokens 2 --out-dir runs/continual_mock --logdir none
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from summarize_rl.branches import Example, Triplet
from summarize_rl.grpo import GRPOTrainer
from summarize_rl.infer import Summarizer
from summarize_rl.judge import MultiAxisJudge
from summarize_rl.policy import WeightPolicy
from summarize_rl.rewards import compute_reward
from summarize_rl.weakness import (
    HeldoutReport,
    WeaknessDiagnoser,
    breakdown_failures,
    default_thresholds,
    eval_heldout,
    load_window,
)

from data.gen_weakness_scenarios import (
    generate as gen_weakness,
    synthesis_budget,
    synthesis_params_from,
)


def record_to_example(rec: dict) -> Example:
    """A synthetic-scenario JSONL record -> Example (carrying id/keyfacts/split)."""
    triplets = [Triplet(*t) for t in rec.get("triplets", []) if len(t) == 3]
    rid = rec.get("id")
    return Example(
        source=rec.get("source_text") or "",
        triplets=triplets,
        id=None if rid is None else str(rid),
        keyfacts=list(rec.get("keyfacts") or []),
        meta={"split": rec.get("split")},
    )


def merge_datasets(base: list, synth_pool: list, ratio: float) -> list:
    """Base + a capped slice of the synthetic pool so synth fraction ~= ratio."""
    base = list(base)
    if not synth_pool or ratio <= 0:
        return base
    target = int(round(ratio * len(base) / max(1e-9, 1.0 - ratio)))
    take = min(len(synth_pool), max(1, target))
    return base + list(synth_pool[:take])


def _rollouts_per_step(trainer) -> int:
    if isinstance(trainer, GRPOTrainer):
        return trainer.config.grpo.group_size
    return trainer.config.train.num_samples


def raw_failure_rates(summarizer, examples, thresholds, config) -> dict[str, float]:
    """Per-axis failure rate of the RAW frozen-LLM baseline (for the RAW ceiling).

    An axis the base model itself fails is a backbone limit, not a policy gap, so
    the diagnoser will keep it out of the synthesis targets.
    """
    counts: dict[str, int] = {}
    n = 0
    for ex in examples:
        text = summarizer.baseline_summary(ex.source, query=ex.query, triplets=ex.triplets)
        bd = compute_reward(summary=text, source=ex.source, triplets=ex.triplets,
                            active_terms=[], config=config)
        n += 1
        for axis in breakdown_failures(bd, thresholds):
            counts[axis] = counts.get(axis, 0) + 1
    return {axis: c / max(1, n) for axis, c in counts.items()}


def _save_state(out_dir: str, round_k: int, history: list, synth_ratio: float) -> None:
    state = {
        "round": round_k,
        "synth_ratio": synth_ratio,
        "history": [
            {"weak_axes": r.weak_axes, "backbone_limited": r.backbone_limited,
             "candidates": r.candidates, "rates": r.rates}
            for r in history
        ],
    }
    with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)


class _PriorRound:
    """Restored past round: exposes `.candidates` for the consecutive-round rule
    and the other saved fields so state.json can be re-serialized on resume."""

    def __init__(self, weak_axes, backbone_limited, candidates, rates):
        self.weak_axes = weak_axes
        self.backbone_limited = backbone_limited
        self.candidates = candidates
        self.rates = rates


def _load_state(out_dir: str):
    path = os.path.join(out_dir, "state.json")
    if not os.path.exists(path):
        return 0, [], None
    with open(path, encoding="utf-8") as fh:
        state = json.load(fh)
    history = [
        _PriorRound(r.get("weak_axes", []), r.get("backbone_limited", []),
                    r.get("candidates", []), r.get("rates", {}))
        for r in state.get("history", [])
    ]
    return state.get("round", 0), history, state.get("synth_ratio")


def run_continual(*, trainer, summarizer, base_examples, heldout_examples,
                  multi_judge, diagnoser, out_dir, rounds, steps_per_round,
                  ga=1, logger=None, synth_ratio=0.3, thresholds=None,
                  raw_ceiling=True, raw_sample=8, regression_margin=0.05,
                  seed=4242, resume=False) -> list:
    """Run the continual-RL loop; returns the per-round DiagnosisReport history."""
    os.makedirs(out_dir, exist_ok=True)
    config = trainer.config
    thresholds = thresholds or (
        trainer.failure_log.thresholds if trainer.failure_log
        else default_thresholds(config.reward)
    )
    fail_path = trainer.failure_log.path if trainer.failure_log else None

    history: list = []
    start_round = 1
    if resume:
        done, history, saved_ratio = _load_state(out_dir)
        start_round = done + 1
        if saved_ratio is not None:
            synth_ratio = saved_ratio
        ckpt = os.path.join(out_dir, f"round_{done}.pt")
        if done and os.path.exists(ckpt):
            trainer.load_checkpoint(ckpt)

    dataset = list(base_examples)
    all_synth: list = []
    prev_heldout: HeldoutReport | None = None
    rps = _rollouts_per_step(trainer)

    for k in range(start_round, rounds + 1):
        # 1) train this round (FailureLog accumulates per rollout)
        for s in range(steps_per_round):
            start = (s * ga) % len(dataset)
            micro = [dataset[(start + i) % len(dataset)] for i in range(ga)]
            m = trainer.train_step(micro)
            if logger:
                logger.log_metrics(m, m.step)
                if trainer.failure_log:
                    logger.log_scalars(trainer.failure_log.tb_scalars(), m.step)

        now = trainer._global_step
        total_rollouts = steps_per_round * ga * rps

        # 2) diagnose the window
        records = load_window(fail_path, steps_per_round, now) if fail_path else []
        source_lookup = {ex.id: ex.source for ex in dataset if ex.id is not None}
        raw_rates = (
            raw_failure_rates(summarizer, base_examples[:raw_sample], thresholds, config)
            if raw_ceiling and raw_sample else None
        )
        report = diagnoser.diagnose(
            records, total_rollouts, source_lookup=source_lookup,
            multi_judge=multi_judge, history=history, raw_axis_rates=raw_rates,
        )

        # 3) held-out cross-check + regression guard
        if multi_judge is not None:
            heldout = eval_heldout(summarizer, heldout_examples, multi_judge)
        else:
            heldout = HeldoutReport(axis_scores={})
        with open(os.path.join(out_dir, f"round_{k}_heldout.json"), "w", encoding="utf-8") as fh:
            json.dump(heldout.axis_scores, fh, ensure_ascii=False, indent=2)
        if heldout.regressed(prev_heldout, margin=regression_margin):
            synth_ratio *= 0.5
        prev_heldout = heldout

        # 4) synthesize weak-axis data (backbone_limited already excluded) + merge
        if report.weak_axes:
            budget = synthesis_budget(report, len(base_examples))
            params = synthesis_params_from(report)
            new = gen_weakness(budget, report.weak_axes, k, seed=seed + k, params=params)
            with open(os.path.join(out_dir, f"synth_r{k}.jsonl"), "w", encoding="utf-8") as fh:
                for rec in new:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            all_synth.extend(record_to_example(r) for r in new)
            dataset = merge_datasets(base_examples, all_synth, synth_ratio)

        # 5) checkpoint + resumable state + TB round curves
        trainer.save_checkpoint(os.path.join(out_dir, f"round_{k}.pt"))
        if logger:
            logger.log_scalars(
                {"continual/round": float(k),
                 **{f"heldout/{a}": v for a, v in heldout.axis_scores.items()}},
                now,
            )
        history.append(report)
        _save_state(out_dir, k, history, synth_ratio)

        if report.converged(diagnoser.min_rate):
            break

    return history


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_cli_pieces(args):
    """Assemble trainer/summarizer/judge/diagnoser for the CLI from train_real."""
    from examples.train_real import build_trainer

    if not args.failure_log:
        args.failure_log = os.path.join(args.out_dir, "failures.jsonl")
    trainer, backend, cfg, logger, examples, _flog = build_trainer(args)

    # Summarizer shares the trainer's live policy so held-out eval reflects
    # current weights.
    summarizer = Summarizer(backend, trainer.policy, cfg, glossary=trainer.glossary)
    multi_judge = MultiAxisJudge(backend, cfg.reward) if args.judge else None

    heldout = examples
    if args.heldout and os.path.exists(args.heldout):
        from examples.train_real import load_corpus
        heldout = load_corpus(args.heldout, args.query, args.heldout_limit)

    diagnoser = WeaknessDiagnoser(min_rate=args.min_rate, consecutive=args.consecutive)
    return trainer, backend, cfg, logger, examples, summarizer, multi_judge, heldout, diagnoser


def main() -> None:
    from examples.train_real import build_arg_parser  # shared flags

    p = build_arg_parser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--steps-per-round", type=int, default=200)
    p.add_argument("--out-dir", default="runs/continual")
    p.add_argument("--synth-ratio", type=float, default=0.3)
    p.add_argument("--min-rate", type=float, default=0.15)
    p.add_argument("--consecutive", type=int, default=2)
    p.add_argument("--judge", action="store_true", help="enable multi-axis judge triage / held-out scoring")
    p.add_argument("--heldout", default="data/test_scenarios_ko.jsonl", help="held-out cross-check set")
    p.add_argument("--heldout-limit", type=int, default=None)
    p.add_argument("--no-raw-ceiling", dest="raw_ceiling", action="store_false")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    (trainer, backend, cfg, logger, examples, summarizer,
     multi_judge, heldout, diagnoser) = _build_cli_pieces(args)

    print(f"continual: rounds={args.rounds} steps/round={args.steps_per_round} "
          f"examples={len(examples)} heldout={len(heldout)} judge={bool(multi_judge)}")

    history = run_continual(
        trainer=trainer, summarizer=summarizer, base_examples=examples,
        heldout_examples=heldout, multi_judge=multi_judge, diagnoser=diagnoser,
        out_dir=args.out_dir, rounds=args.rounds, steps_per_round=args.steps_per_round,
        ga=max(1, args.grad_accum), logger=logger, synth_ratio=args.synth_ratio,
        raw_ceiling=args.raw_ceiling, seed=args.seed, resume=args.resume,
    )
    logger.close()
    for k, rep in enumerate(history, 1):
        print(f"round {k}: weak={rep.weak_axes} backbone_limited={rep.backbone_limited} "
              f"converged={rep.converged(args.min_rate)}")


if __name__ == "__main__":
    main()
