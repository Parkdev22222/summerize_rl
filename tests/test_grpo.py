import copy
import math
import os

import torch

from summarize_rl.branches import Example, Triplet
from summarize_rl.config import Config, DecodeConfig, GRPOConfig, PolicyConfig, TrainConfig
from summarize_rl.glossary import Glossary
from summarize_rl.grpo import GRPOTrainer
from summarize_rl.llm_backend import MockBackend


def _config(**grpo_overrides):
    cfg = Config()
    cfg.policy = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    cfg.decode = DecodeConfig(max_new_tokens=6, min_new_tokens=2, eos_token_id=1)
    cfg.train = TrainConfig(total_steps=10, grad_accum_steps=1, lr=1e-2, seed=0)
    cfg.grpo = GRPOConfig(group_size=4, inner_epochs=2, **grpo_overrides)
    return cfg


def _example():
    return Example(
        source="적 부대가 이동 중이며 고지를 점령했다.",
        triplets=[Triplet("적부대", "행동", "이동"), Triplet("적부대", "점령", "고지")],
        query="요약하시오.",
    )


def _trainer(cfg):
    from summarize_rl.policy import WeightPolicy

    policy = WeightPolicy(cfg.policy)
    backend = MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)
    glossary = Glossary({"기동": ["이동"], "점령": ["점령"]})
    gen = torch.Generator().manual_seed(0)
    return GRPOTrainer(policy, backend, cfg, glossary=glossary, generator=gen)


def test_grpo_step_updates_policy():
    trainer = _trainer(_config())
    before = copy.deepcopy(trainer.policy.state_dict())
    trainer.train_step([_example()])
    after = trainer.policy.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_reference_policy_is_frozen():
    trainer = _trainer(_config())
    ref_before = copy.deepcopy(trainer.ref_policy.state_dict())
    for p in trainer.ref_policy.parameters():
        assert not p.requires_grad
    trainer.train_step([_example()])
    # reference must not move even as the trained policy does
    for k, v in trainer.ref_policy.state_dict().items():
        assert torch.equal(ref_before[k], v)


def test_llm_stays_frozen():
    trainer = _trainer(_config())
    trainer.train_step([_example()])
    model = trainer.backend.model if hasattr(trainer.backend, "model") else None
    if model is not None:
        for p in model.parameters():
            assert not p.requires_grad


def test_metrics_in_range():
    trainer = _trainer(_config())
    m = trainer.train_step([_example()])
    assert 0.0 <= m.weight_a <= 1.0
    assert abs((m.weight_b + m.weight_c + m.weight_d) - 1.0) < 1e-3
    assert m.kl >= -1e-6  # k3 estimator is non-negative
    assert 0.0 <= m.clip_frac <= 1.0
    assert m.grad_norm >= 0.0
    assert m.step == 1


def test_group_advantages_are_normalized():
    trainer = _trainer(_config())
    adv = trainer._group_advantages([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(sum(adv), 0.0, abs_tol=1e-6)  # mean-zero
    # unit-ish std (eps makes it slightly under 1)
    var = sum(a * a for a in adv) / len(adv)
    assert 0.9 < var <= 1.0


def test_zero_variance_group_gives_finite_advantage():
    trainer = _trainer(_config())
    adv = trainer._group_advantages([5.0, 5.0, 5.0, 5.0])
    assert all(math.isfinite(a) and abs(a) < 1e-3 for a in adv)


def test_checkpoint_roundtrip(tmp_path):
    cfg = _config()
    trainer = _trainer(cfg)
    trainer.train_step([_example()])
    path = os.path.join(tmp_path, "grpo.pt")
    trainer.maybe_update_best(0.5)
    trainer.save_checkpoint(path, is_best=True)
    assert os.path.exists(path)
    assert os.path.exists(os.path.join(tmp_path, "best.pt"))

    trainer2 = _trainer(cfg)
    trainer2.load_checkpoint(path)
    s1, s2 = trainer.policy.state_dict(), trainer2.policy.state_dict()
    for k in s1:
        assert torch.equal(s1[k], s2[k])
    assert trainer2._global_step == trainer._global_step
