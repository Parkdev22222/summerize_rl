"""Summarizer inference tests (MockBackend, no model download).

Covers the checkpoint round-trip the REPL relies on: train -> save -> load ->
summarize, plus the bare-state-dict path, a clear missing-file error, and greedy
determinism.
"""

import os

import pytest
import torch

from summarize_rl.branches import Example, Triplet
from summarize_rl.config import Config, DecodeConfig, PolicyConfig, TrainConfig
from summarize_rl.glossary import Glossary
from summarize_rl.infer import Summarizer, SummaryResult
from summarize_rl.llm_backend import MockBackend
from summarize_rl.policy import WeightPolicy
from summarize_rl.train import SCSTTrainer


def _config():
    cfg = Config()
    cfg.policy = PolicyConfig(llm_hidden_size=8, hidden_dim=16)
    cfg.decode = DecodeConfig(max_new_tokens=6, min_new_tokens=2, eos_token_id=1)
    cfg.train = TrainConfig(num_samples=3, total_steps=2, grad_accum_steps=1, seed=0)
    return cfg


def _backend():
    return MockBackend(vocab_size=16, hidden_size=8, eos_token_id=1)


def _glossary():
    return Glossary({"기동": ["이동"], "점령": ["점령"]})


def _summarizer(cfg):
    policy = WeightPolicy(cfg.policy)
    return Summarizer(_backend(), policy, cfg, glossary=_glossary())


SOURCE = "적 부대가 이동 중이며 고지를 점령했다."


def test_summarize_returns_result_with_active_terms():
    cfg = _config()
    s = _summarizer(cfg)
    result = s.summarize(SOURCE)
    assert isinstance(result, SummaryResult)
    assert isinstance(result.text, str) and result.text != ""
    # Glossary gates on the source: "이동" -> 기동, "점령" -> 점령.
    assert set(result.active_terms) == {"기동", "점령"}
    # Mean weights obey the policy constraints: b + c + d ~= 1.
    a, b, c, d = result.mean_weights
    assert 0.0 <= a <= 1.0
    assert b + c + d == pytest.approx(1.0, abs=1e-4)


def test_greedy_is_deterministic():
    cfg = _config()
    s = _summarizer(cfg)
    assert s.summarize(SOURCE).text == s.summarize(SOURCE).text


def test_checkpoint_round_trip_train_save_load(tmp_path):
    cfg = _config()
    # Train a couple of steps and save a real training checkpoint.
    policy = WeightPolicy(cfg.policy)
    gen = torch.Generator().manual_seed(0)
    trainer = SCSTTrainer(policy, _backend(), cfg, glossary=_glossary(), generator=gen)
    example = Example(source=SOURCE, triplets=[Triplet("적부대", "행동", "이동")], query="요약하시오.")
    trainer.train_step([example])
    ckpt = os.path.join(str(tmp_path), "best.pt")
    trainer.save_checkpoint(ckpt)

    # Fresh summarizer loads those trained weights and produces a summary.
    fresh = WeightPolicy(cfg.policy)
    s = Summarizer(_backend(), fresh, cfg, glossary=_glossary())
    step = s.load_checkpoint(ckpt)
    assert step == trainer._global_step

    # Loaded policy must match the trained policy's parameters exactly.
    for k, v in trainer.policy.state_dict().items():
        assert torch.allclose(fresh.state_dict()[k], v)

    assert s.summarize(SOURCE).text != ""


def test_load_bare_state_dict(tmp_path):
    cfg = _config()
    policy = WeightPolicy(cfg.policy)
    path = os.path.join(str(tmp_path), "bare.pt")
    torch.save(policy.state_dict(), path)

    s = Summarizer(_backend(), WeightPolicy(cfg.policy), cfg)
    assert s.load_checkpoint(path) == 0
    for k, v in policy.state_dict().items():
        assert torch.allclose(s.policy.state_dict()[k], v)


def test_missing_checkpoint_raises():
    cfg = _config()
    s = _summarizer(cfg)
    with pytest.raises(FileNotFoundError):
        s.load_checkpoint("/no/such/checkpoint.pt")


def test_unrecognized_checkpoint_raises(tmp_path):
    cfg = _config()
    path = os.path.join(str(tmp_path), "junk.pt")
    torch.save({"not_policy": 123}, path)
    s = _summarizer(cfg)
    with pytest.raises(ValueError):
        s.load_checkpoint(path)


def test_query_override_changes_branches():
    cfg = _config()
    s = _summarizer(cfg)
    # Different instruction -> different Q/XQ/etc. prompt text -> (generally)
    # different decode. At minimum the call succeeds and returns a summary.
    r = s.summarize(SOURCE, query="핵심만 한 문장으로 요약하시오.")
    assert r.text != ""
