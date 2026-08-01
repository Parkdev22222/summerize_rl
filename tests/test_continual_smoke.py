"""End-to-end smoke of the continual-RL orchestrator on MockBackend.

Runs 2 rounds x a few steps and asserts the loop completes without crashing and
emits every expected artifact (failure log, per-round held-out json, round
checkpoints, synthetic data, resumable state). The saturating threshold + the
consecutive-round rule mean a weak axis is confirmed on round 2, so synthesis
actually fires.
"""

import json
import os

import torch

from summarize_rl.branches import Example, Triplet
from summarize_rl.config import Config, DecodeConfig, GRPOConfig, PolicyConfig, TrainConfig
from summarize_rl.glossary import Glossary
from summarize_rl.infer import Summarizer
from summarize_rl.llm_backend import MockBackend
from summarize_rl.policy import WeightPolicy
from summarize_rl.train import SCSTTrainer
from summarize_rl.weakness import FailureLog, WeaknessDiagnoser

from examples.run_continual import merge_datasets, record_to_example, run_continual


def _cfg():
    cfg = Config()
    cfg.policy = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    cfg.decode = DecodeConfig(max_new_tokens=6, min_new_tokens=2, eos_token_id=1)
    cfg.train = TrainConfig(num_samples=3, total_steps=100, grad_accum_steps=1, seed=0)
    cfg.grpo = GRPOConfig(group_size=3)
    return cfg


def _examples(n=4):
    return [Example(f"산악 지형에서 {i}중대 120명이 고지를 점령했다.",
                    [Triplet(f"{i}중대", "규모", "120명")], id=str(i)) for i in range(n)]


def test_two_round_continual_produces_all_artifacts(tmp_path):
    cfg = _cfg()
    backend = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    policy = WeightPolicy(cfg.policy)
    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    # Saturating coverage threshold => every rollout fails coverage => a strong
    # candidate; consecutive=2 confirms it on round 2 and triggers synthesis.
    flog = FailureLog(os.path.join(out, "failures.jsonl"), thresholds={"coverage": 2.0})
    trainer = SCSTTrainer(policy, backend, cfg, glossary=Glossary({"기동": ["이동"]}),
                          generator=torch.Generator().manual_seed(0), failure_log=flog)
    summarizer = Summarizer(backend, policy, cfg, glossary=Glossary({"기동": ["이동"]}))
    diagnoser = WeaknessDiagnoser(min_rate=0.15, consecutive=2)

    base = _examples()
    history = run_continual(
        trainer=trainer, summarizer=summarizer, base_examples=base,
        heldout_examples=base, multi_judge=None, diagnoser=diagnoser,
        out_dir=out, rounds=2, steps_per_round=4, ga=1,
        thresholds={"coverage": 2.0}, raw_ceiling=False, seed=1,
    )

    assert len(history) == 2
    assert os.path.exists(os.path.join(out, "failures.jsonl"))
    for k in (1, 2):
        assert os.path.exists(os.path.join(out, f"round_{k}_heldout.json"))
        assert os.path.exists(os.path.join(out, f"round_{k}.pt"))
    assert os.path.exists(os.path.join(out, "state.json"))
    # Round 2 confirms coverage weak -> synthetic data written.
    assert "coverage" in history[1].weak_axes
    assert os.path.exists(os.path.join(out, "synth_r2.jsonl"))
    with open(os.path.join(out, "state.json")) as fh:
        state = json.load(fh)
    assert state["round"] == 2


def test_resume_continues_from_saved_state(tmp_path):
    def _make():
        cfg = _cfg()
        backend = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
        policy = WeightPolicy(cfg.policy)
        flog = FailureLog(os.path.join(out, "failures.jsonl"), thresholds={"coverage": 2.0})
        trainer = SCSTTrainer(policy, backend, cfg, glossary=Glossary({"기동": ["이동"]}),
                              generator=torch.Generator().manual_seed(0), failure_log=flog)
        summarizer = Summarizer(backend, policy, cfg, glossary=Glossary({"기동": ["이동"]}))
        return trainer, summarizer

    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    base = _examples()
    diagnoser = WeaknessDiagnoser(min_rate=0.15, consecutive=2)

    t1, s1 = _make()
    run_continual(trainer=t1, summarizer=s1, base_examples=base, heldout_examples=base,
                  multi_judge=None, diagnoser=diagnoser, out_dir=out, rounds=2,
                  steps_per_round=4, thresholds={"coverage": 2.0}, raw_ceiling=False)

    # Resume for one more round; must not crash re-serializing restored history.
    t2, s2 = _make()
    history = run_continual(trainer=t2, summarizer=s2, base_examples=base,
                            heldout_examples=base, multi_judge=None, diagnoser=diagnoser,
                            out_dir=out, rounds=3, steps_per_round=4,
                            thresholds={"coverage": 2.0}, raw_ceiling=False, resume=True)
    assert len(history) == 3  # 2 restored + 1 new
    assert os.path.exists(os.path.join(out, "round_3.pt"))
    with open(os.path.join(out, "state.json")) as fh:
        assert json.load(fh)["round"] == 3


def test_merge_datasets_respects_ratio_cap():
    base = [Example(f"s{i}", []) for i in range(10)]
    synth = [Example(f"x{i}", []) for i in range(10)]
    merged = merge_datasets(base, synth, ratio=0.3)
    n_synth = len(merged) - len(base)
    # synth fraction ~= 0.3 -> about 4 synth added to 10 base (4/14 ~= 0.29)
    assert 3 <= n_synth <= 5
    assert merge_datasets(base, synth, ratio=0.0) == base


def test_record_to_example_roundtrips_fields():
    rec = {"id": 20001, "source_text": "본문", "triplets": [["A", "r", "B"]],
           "keyfacts": ["k"], "split": "synthetic_r1"}
    ex = record_to_example(rec)
    assert ex.source == "본문" and ex.id == "20001"
    assert ex.triplets[0].tail == "B" and ex.keyfacts == ["k"]
    assert ex.meta["split"] == "synthetic_r1"
