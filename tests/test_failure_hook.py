"""Trainer -> FailureLog integration: both trainers record every rollout failure.

Uses MockBackend and a saturating threshold (coverage < 2.0 is always true) so
the hook is guaranteed to fire once per rollout, proving the plumbing without
depending on the mock's exact reward values.
"""

import torch

from summarize_rl.branches import Example, Triplet
from summarize_rl.config import Config, DecodeConfig, GRPOConfig, PolicyConfig, TrainConfig
from summarize_rl.glossary import Glossary
from summarize_rl.grpo import GRPOTrainer
from summarize_rl.llm_backend import MockBackend
from summarize_rl.policy import WeightPolicy
from summarize_rl.train import SCSTTrainer
from summarize_rl.weakness import FailureLog, load_window


def _cfg(**train_overrides):
    cfg = Config()
    cfg.policy = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    cfg.decode = DecodeConfig(max_new_tokens=6, min_new_tokens=2, eos_token_id=1)
    cfg.train = TrainConfig(num_samples=4, total_steps=10, grad_accum_steps=1,
                            seed=0, **train_overrides)
    cfg.grpo = GRPOConfig(group_size=4)
    return cfg


def _example(rid="10000"):
    return Example(
        source="적 부대가 이동 중이며 고지를 점령했다.",
        triplets=[Triplet("적부대", "행동", "이동")],
        query="요약하시오.", id=rid,
    )


def test_scst_hook_records_each_rollout(tmp_path):
    cfg = _cfg()
    log = FailureLog(str(tmp_path / "f.jsonl"), thresholds={"coverage": 2.0})
    trainer = SCSTTrainer(
        WeightPolicy(cfg.policy), MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1),
        cfg, glossary=Glossary({"기동": ["이동"]}),
        generator=torch.Generator().manual_seed(0), failure_log=log,
    )
    trainer.train_step([_example()])
    recs = load_window(str(tmp_path / "f.jsonl"), window_steps=100, now=trainer._global_step)
    assert len(recs) == cfg.train.num_samples          # one per rollout
    assert all(r.axis == "coverage" and r.example_id == "10000" for r in recs)
    assert log.failure_rates()["coverage"] == 1.0


def test_grpo_hook_records_each_rollout(tmp_path):
    cfg = _cfg()
    log = FailureLog(str(tmp_path / "f.jsonl"), thresholds={"coverage": 2.0})
    trainer = GRPOTrainer(
        WeightPolicy(cfg.policy), MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1),
        cfg, glossary=Glossary({"기동": ["이동"]}),
        generator=torch.Generator().manual_seed(0), failure_log=log,
    )
    trainer.train_step([_example()])
    recs = load_window(str(tmp_path / "f.jsonl"), window_steps=100, now=trainer._global_step)
    assert len(recs) == cfg.grpo.group_size
    assert all(r.axis == "coverage" for r in recs)


def test_no_failure_log_is_harmless(tmp_path):
    cfg = _cfg()
    trainer = SCSTTrainer(
        WeightPolicy(cfg.policy), MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1),
        cfg, glossary=Glossary({"기동": ["이동"]}),
        generator=torch.Generator().manual_seed(0),
    )
    trainer.train_step([_example()])  # must not raise with failure_log unset
