import copy
import os

import torch

from summarize_rl.branches import Example, Triplet
from summarize_rl.config import Config, DecodeConfig, PolicyConfig, TrainConfig
from summarize_rl.decoder import combine_logits
from summarize_rl.glossary import Glossary
from summarize_rl.llm_backend import MockBackend, StepOutput
from summarize_rl.policy import WeightPolicy, Weights
from summarize_rl.train import SCSTTrainer, build_scheduler


def _config(**train_overrides):
    cfg = Config()
    cfg.policy = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    cfg.decode = DecodeConfig(max_new_tokens=6, min_new_tokens=2, eos_token_id=1)
    cfg.train = TrainConfig(
        num_samples=4, total_steps=10, grad_accum_steps=1, seed=0, **train_overrides
    )
    return cfg


def _example():
    return Example(
        source="적 부대가 이동 중이며 고지를 점령했다.",
        triplets=[Triplet("적부대", "행동", "이동"), Triplet("적부대", "점령", "고지")],
        query="요약하시오.",
    )


def _trainer(cfg, backend=None):
    policy = WeightPolicy(cfg.policy)
    backend = backend or MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    glossary = Glossary({"기동": ["이동"], "점령": ["점령"]})
    gen = torch.Generator().manual_seed(0)
    return SCSTTrainer(policy, backend, cfg, glossary=glossary, generator=gen)


def test_llm_logits_are_constant_wrt_policy():
    # combine_logits must detach LLM logits: a leaf requiring grad gets no grad.
    logits = torch.randn(4, 5, requires_grad=True)
    step = StepOutput(logits=logits, hidden=torch.zeros(4, 2))
    w = Weights(
        a=torch.tensor([0.5], requires_grad=True),
        b=torch.tensor([0.3], requires_grad=True),
        c=torch.tensor([0.3], requires_grad=True),
        d=torch.tensor([0.4], requires_grad=True),
    )
    combine_logits(step, w).sum().backward()
    assert logits.grad is None  # frozen LLM receives no gradient


def test_train_step_updates_policy():
    cfg = _config()
    trainer = _trainer(cfg)
    before = copy.deepcopy(trainer.policy.state_dict())
    trainer.train_step([_example()])
    after = trainer.policy.state_dict()
    changed = any(
        not torch.equal(before[k], after[k]) for k in before
    )
    assert changed, "policy parameters did not change after a train step"


def test_train_step_returns_metrics_in_range():
    cfg = _config()
    trainer = _trainer(cfg)
    m = trainer.train_step([_example()])
    assert 0 <= m.weight_a <= 1
    assert abs((m.weight_b + m.weight_c + m.weight_d) - 1.0) < 1e-3
    assert 0.0 <= m.faithfulness <= 1.0
    assert 0.0 <= m.coverage <= 1.0
    assert 0.0 <= m.term_usage <= 1.0
    assert m.grad_norm >= 0.0
    assert m.step == 1


def test_grad_accumulation_batch():
    cfg = _config()
    trainer = _trainer(cfg)
    m = trainer.train_step([_example(), _example()])
    assert m.step == 1  # one optimizer step for the whole micro-batch


def test_greedy_baseline_runs():
    cfg = _config(baseline="greedy", reward_norm=False)
    trainer = _trainer(cfg)
    m = trainer.train_step([_example()])
    assert m.step == 1


def test_scheduler_warmup_then_decay():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_scheduler(opt, warmup_steps=5, total_steps=20)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    # warmup rising
    assert lrs[1] < lrs[4]
    # decays afterward
    assert lrs[5] > lrs[-1]


def test_checkpoint_roundtrip(tmp_path):
    cfg = _config()
    trainer = _trainer(cfg)
    trainer.train_step([_example()])
    path = os.path.join(tmp_path, "ckpt.pt")
    trainer.maybe_update_best(0.5)
    trainer.save_checkpoint(path, is_best=True)
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(tmp_path, "best.pt"))

    # New trainer loads and matches.
    trainer2 = _trainer(cfg)
    trainer2.load_checkpoint(path)
    s1 = trainer.policy.state_dict()
    s2 = trainer2.policy.state_dict()
    for k in s1:
        assert torch.equal(s1[k], s2[k])
    assert trainer2._global_step == trainer._global_step


def test_maybe_update_best():
    cfg = _config()
    trainer = _trainer(cfg)
    assert trainer.maybe_update_best(1.0) is True
    assert trainer.maybe_update_best(0.5) is False
    assert trainer.maybe_update_best(2.0) is True
